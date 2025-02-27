import datetime
import logging
from math import log
from django.utils import timezone

from django.conf import settings
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

from .models import PromptFilter, User, Channel, Video, Comment, FilterPrediction
from . import updates
from . import utils


logger = logging.getLogger(__name__)


class YoutubeAPI:

    def __init__(self, credentials):
        if credentials:
            if isinstance(credentials, dict):
                self.credentials = credentials
            else:
                self.credentials = utils.credentials_to_dict(credentials)

            if self.credentials.get('token', 'FAKE_TOKEN') == 'FAKE_TOKEN':
                # if the token is fake, we will not be able to use the private youtube API
                self.private_youtube = None
            else:
                filtered_credentials = {k: v for k, v in self.credentials.items() if k != 'myChannelId'}
                self.private_youtube = build('youtube', 'v3', credentials=Credentials(**filtered_credentials))
        else:
            self.credentials = {}
            self.private_youtube = None

        self.youtube = build('youtube', 'v3', developerKey='AIzaSyBBdr6RyGr0fRnjAoHzn-NXRmwy_tiYL5A')

    def retrieve_account_with_handle(self, handle):
        # Step 1: Search for the channel using the handle
        search_response = self.youtube.search().list(
            q=handle,
            part='id,snippet',
            type='channel',
            maxResults=1
        ).execute()
        
        if not search_response.get('items'):
            return None
        
        channel_id = search_response['items'][0]['id']['channelId']
        
        # Step 2: Get detailed channel information
        channel_response = self.youtube.channels().list(
            id=channel_id,
            part='snippet'
        ).execute()
        
        if not channel_response.get('items'):
            return None
        
        channel_info = channel_response['items'][0]['snippet']
        
        return {
            'username': handle,
            'avatar': channel_info['thumbnails']['default']['url'],
            'channel': {
                'name': channel_info['title'],
                'id': channel_id
            }
        }
            

    def retrieve_channels(self):
        channel = self.youtube.channels().list(mine=True, part='snippet').execute()
        return channel

    def retrieve_account(self):
        account_info = self.private_youtube.channels().list(mine=True, part='snippet').execute()
        logger.info(f'Account info: {account_info}')
        snippet = account_info['items'][0]['snippet']
        channel = snippet['title'] # Leijie Wang
        username = snippet['customUrl'] #@Leijiewang
        avatar = snippet['thumbnails']['default']['url']
        channelID = account_info['items'][0]['id'] #UC1yBKRuGpC1tSM73A0ZjYjQ
        return {
            'username': username,
            'avatar': avatar,
            'channel': {
                'name': channel,
                'id': channelID
            }
        }

    def retrieve_video_statistics(self, video_id):
        video_details_request = self.youtube.videos().list(
            part='statistics',
            id=f'{video_id}'
        )
        video_details_response = video_details_request.execute()
        video_details_item = video_details_response['items'][0]
        return video_details_item['statistics']['commentCount']
    
    def retrieve_videos(self, channel_id, video_num=5, published_after=None):
        if published_after:
            published_after = published_after.isoformat('T').replace('+00:00', 'Z')
        
        request = self.youtube.search().list(
            part='snippet',
            channelId=channel_id,
            type='video',
            order='date',
            publishedAfter=published_after,
        )
        response = request.execute()
        logger.info(f'There are {len(response["items"])} videos in the channel')
        videos = []
        for item in response['items']:
            if len(videos) >= video_num:
                break
            video_id = item['id']['videoId']
            if self.retrieve_video_statistics(video_id) == 0:
                logger.info(f'Video {video_id} has no comments')
                continue
            video_title = item['snippet']['title']
            video_description = item['snippet']['description']
            video_image = item['snippet']['thumbnails']['default']['url']
            video_published = item['snippet']['publishTime']
            video_link = f'https://www.youtube.com/watch?v={video_id}'
            video, created = Video.objects.update_or_create(
                id=video_id,
                channel_id=channel_id,
                defaults={
                    'title': video_title,
                    'description': video_description,
                    'thumbnail': video_image,
                    'posted_at': video_published,
                    'video_link': video_link
                }
            )
            videos.append(video)
        return videos

    def __retrieve_replies(self, parent_comment):
        replies = []
        if parent_comment.total_replies > 0:
            # print(f'This comment has {parent_comment.total_replies} replies')
            replies = []
            reply_request = self.youtube.comments().list(
                part='snippet',
                parentId=parent_comment.id,
            )
            while reply_request:
                reply_response = reply_request.execute()
                for reply_item in reply_response.get('items', []):
                    reply_comment = self.__process_comment(reply_item, parent_comment.video.id, parent_comment)
                    replies.append(reply_comment)
                # Handle pagination
                reply_request = self.youtube.comments().list_next(reply_request, reply_response)
        # if len(replies) > 10:
        #     logger.debug(f'Extracted {len(replies)} replies from the comment {parent_comment.id}')

    def __process_comment(self, comment_item, video_id, parent_comment=None):
        comment_snippet = comment_item
        if 'topLevelComment' in comment_snippet:
            # comments that are not replies have this additional level
            comment_snippet = comment_snippet['topLevelComment']
        comment_id = comment_snippet['id']

        comment_snippet = comment_snippet['snippet']
        # print(comment_snippet)
        user = comment_snippet['authorDisplayName']
        content = comment_snippet['textOriginal']
        user_image = comment_snippet['authorProfileImageUrl']
        posted_at = comment_snippet['publishedAt']
        likes = comment_snippet['likeCount']
        # there are no dislikes counts here
        if 'totalReplyCount' in comment_item:
            total_replies = comment_item['totalReplyCount']
        else:
            total_replies = 0

        user, _ = User.objects.update_or_create(
            username=user,
            defaults={
                'avatar': user_image
            }
        )

        comment, _ = Comment.objects.update_or_create(
            video_id=video_id,
            id=comment_id,
            defaults={
                'parent': parent_comment,
                'user': user,
                'content': content,
                'posted_at': posted_at,
                'likes': likes,
                'total_replies': total_replies
            }
        )
        self.__retrieve_replies(comment)
        return comment

    def retrieve_comments(self, video_id, comment_num=100, published_after=None):
        if published_after:
            published_after = published_after.isoformat('T') + 'Z'
        
        # https://developers.google.com/youtube/v3/docs/commentThreads/list
        comment_request = self.youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            order='time',
            # publishedAfter=published_after # This is not supported
        )
        try:
            comment_response = comment_request.execute()
        except HttpError as error:
            print(f'An HTTP error occurred:\n{error}')
            return []

        # logger.info(f'There are {len(comment_response["items"])} comments in the video')
        comments = []
        for comment_item in comment_response['items']:
            comment = self.__process_comment(comment_item['snippet'], video_id)
            if published_after and comment.posted_at < published_after:
                # we have reached the first comment after the cutoff time
                return comments
            comments.append(comment)
        
        while ('nextPageToken' in comment_response) and (len(comments) < comment_num):
            comment_request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=comment_response['nextPageToken']
            )
            comment_response = comment_request.execute()
            for comment_item in comment_response['items']:
                comment = self.__process_comment(comment_item['snippet'], video_id)
                children_comments = Comment.objects.filter(parent=comment)
                if published_after and comment.posted_at < published_after:
                    # we have reached the first comment after the cutoff time
                    return comments
                comments.append(comment)
                comments.extend(children_comments)
        return comments
        
    def synchronize(self, user, restart=False):
        now_synchronized = datetime.datetime.now()
        channel = user.channel
        videos = Video.objects.filter(channel=channel).all()
        # delete all existing videos and comments
        if restart:
            videos.delete()

        new_videos = self.retrieve_videos(channel.id, published_after=user.last_sync)
        logger.info(f'Found {len(new_videos)} new videos for the channel {user.username}')
        videos = Video.objects.filter(channel=channel).all()

        total_new_comments = 0
        for video in videos:
            new_comments = self.retrieve_comments(video.id, published_after=user.last_sync)
            logger.info(f'Extracted {len(new_comments)} new comments from the video {video.title}')
            total_new_comments += len(new_comments)
        logger.info(f'Found {total_new_comments} new comments for the channel {user.username}')
        
        filters = PromptFilter.objects.filter(channel=channel).all()
        for filter in filters:
            logger.info(f'Updated predictions for the filter {filter.name}')
            updates.update_predictions(filter, 'new', now_synchronized=now_synchronized)
        
        user.second_last_sync = user.last_sync
        user.last_sync = now_synchronized
        user.save()

    def create_account(self, oauth=True, handle=None):
        """
            Create a new account for the user.
            If oauth is True, retrieve account information using OAuth.
            If oauth is False, retrieve account information using the handle.
        """

        if oauth:
            account_info = self.retrieve_account()
        else:
            account_info = self.retrieve_account_with_handle(handle)
            self.credentials = utils.populate_fake_credentials(account_info['channel']['id'])
        logger.info('Account info: %s', account_info)
        user, created = User.objects.update_or_create(
            username=account_info['username'],
            defaults={
                'oauth_credentials': self.credentials,
                'avatar': account_info['avatar'],  # Make sure 'avatar' is in account_info if needed
                'moderation_access': oauth,
            }
        )

        channel, created = Channel.objects.update_or_create(
            owner=user,
            id=account_info['channel']['id'],  # Use channel ID as the unique key
            defaults={
                'name': account_info['channel']['name'],
            }
        )
        if created:
            # only when the channel is newly created, we need to populate the filters
            utils.populate_filters(channel)
        return user, channel

    def delete_comment(self, comment_id):
        # https://developers.google.com/youtube/v3/docs/comments/delete#try-it
        if self.private_youtube is None:
            logger.info(f'No credentials provided, cannot delete comment {comment_id}')
            return
        
        if settings.DJANGO_ENV == 'production':
            request = self.private_youtube.comments().delete(
                id=comment_id
            )
            response = request.execute()
            logger.info(f'Deleting comment {comment_id}: {response}')
            return response
        elif settings.DJANGO_ENV == 'debug':
            logger.info(f'Deleting comment {comment_id} in debug mode')
            return

    def execute_action_on_prediction(self, prediction):
        filter = prediction.filter
        if filter.action == 'delete':
            comment = prediction.comment
            self.delete_comment(comment.id)
            comment.status = 'deleted'
            comment.save()