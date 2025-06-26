from calendar import c
import datetime
import logging
import os

from django.conf import settings
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from litellm import moderation

from .models import CommentStatus, PromptFilter, User, Channel, Video, Comment, FilterPrediction
from . import updates
from . import utils


logger = logging.getLogger(__name__)
"""
    When we are testing, we do not want to fetch too many comments from a video
    But in the actual deployment, we want to make sure that every comment is fetched.
"""
COMMENTS_CAP_PER_VIDEO = os.getenv("COMMENTS_CAP_PER_VIDEO", "True") == "True"

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
        """
            Generator function that paginates over YouTube Search results,
            yielding one Video object at a time.

            Args:
                channel_id (str): The ID of the YouTube channel to retrieve videos from.
                video_num (int, optional): The maximum number of videos to retrieve. Defaults to 5.
                When set to None, it retrieves all videos.
                published_after (datetime, optional): Only retrieve videos published after this date.
        """
        def _process_video(video_item):
            video_id = video_item['id']['videoId']
            # skip videos that have no comments
            if self.retrieve_video_statistics(video_id) == 0:
                logger.info(f'Video {video_id} has no comments')
                return None

            video_title = video_item['snippet']['title']
            video_description = video_item['snippet']['description']
            video_image = video_item['snippet']['thumbnails']['default']['url']
            video_published = video_item['snippet']['publishTime']
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
            return video


        if published_after:
            published_after = published_after.isoformat('T').replace('+00:00', 'Z')
        
        page_token = None
        total_fetched = 0

        while True:
            # the results are sorted in reverse chronological order
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                publishedAfter=published_after,
                pageToken=page_token,
            )
            response = request.execute()
            items = response.get('items', [])
            if not items:
                logger.info(f'No items found for channel {channel_id}')
                break

            for item in response['items']:
                if video_num is not None and total_fetched >= video_num:
                    break
                
                video = _process_video(item)
                if video is None:
                    # skip videos that have no comments
                    continue
                
                total_fetched += 1
                yield video

            # If we've met the limit or there's no next page, exit.
            if (video_num is not None and total_fetched >= video_num) or 'nextPageToken' not in response:
                break

            # Otherwise, move to the next page
            page_token = response['nextPageToken']

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

    def retrieve_comments(self, video_id, comment_num=500, published_after=None):
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
        
        while ('nextPageToken' in comment_response) and (comment_num is None or len(comments) < comment_num):
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
        
    def synchronize(self, user, restart=False, max_new_comments=500):
        """
            Synchronize the user's YouTube channel with the database.
            If restart is True, delete all existing videos and comments.
            If max_new_comments is set, stop synchronizing when the number of new comments exceeds this limit.
        """

        
        now_synchronized = datetime.datetime.now()
        channel = user.channel

        existing_videos = Video.objects.filter(channel=channel).all()
        # delete all existing videos and comments
        if restart:
            existing_videos.delete()
            existing_videos = []
            logger.info(f'Deleted all existing videos for the channel {user.username}')
        
        
        total_new_comments = 0
        apply_restrictions = COMMENTS_CAP_PER_VIDEO or user.username == '@CNN'
        if apply_restrictions:
            # we only want to fetch at most 5 existing videos if we are using COMMENTS_CAP_PER_VIDEO
            existing_videos = existing_videos[:1]
        for video in existing_videos:
            # in case this video has too many comments, we will only retrive new comments after the last sync
            # this is helpful to avoid fetch all comments for a video that has been synchronized before
            # because of running multiple times
            if apply_restrictions:
                comment_num = 1
            else:
                comment_num = None

            new_comments = self.retrieve_comments(video.id, comment_num=comment_num, published_after=user.last_sync)
            logger.info(f'Fetched {len(new_comments)} new comments from existing video {video.title}')
            total_new_comments += len(new_comments)
            

        new_videos_count = 0
        for new_video in self.retrieve_videos(channel.id, video_num=None, published_after=user.last_sync):
            new_videos_count += 1
            # if the user has just created the account, we will only fetch at most max_new_comments new comments
            # otherwise, we will not limit the number of new comments
            if apply_restrictions:
                comment_num = 1
            elif user.last_sync is None:
                comment_num = 200
            else:
                comment_num = None
            new_comments = self.retrieve_comments(new_video.id, comment_num=comment_num, published_after=user.last_sync)
            logger.info(f'Fetched {len(new_comments)} new comments from new video {new_video.title}')
            total_new_comments += len(new_comments)
            # if the user has just created the account, we will only fetch at most max_new_comments new comments
            # but if the user has been using the system for a while, we should not limit the number of new comments
            if user.last_sync is None and total_new_comments > max_new_comments:
                logger.info(f'Stopped synchronizing after {max_new_comments} new comments')
                break
            if apply_restrictions and new_videos_count > 2:
                # if the user has just created the account, we will only fetch at most 5 new videos
                logger.info(f'Stopped synchronizing after {new_videos_count} new videos because we set COMMENTS_CAP_PER_VIDEO to True')
                break

        logger.info(f'Found {new_videos_count} new videos for {user.username} after {user.last_sync}')
        logger.info(f'Total of {total_new_comments} new comments for {user.username}')

        filters = PromptFilter.objects.filter(channel=channel).all()
        for filter in filters:
            logger.info(f'Updated predictions for the filter {filter.name}')
            updates.update_predictions(filter, 'new', now_synchronized=now_synchronized)
        
        user.second_last_sync = user.last_sync
        user.last_sync = now_synchronized
        user.save()
        return {
            'newCommentsCount': total_new_comments,
            'newVideosCount': new_videos_count
        }

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
            id=account_info['channel']['id'],  # Use channel ID as the unique key
            defaults={
                'name': account_info['channel']['name'],
                'owner': user,
            }
        )
        # if created:
        #     # only when the channel is newly created, we need to populate the filters
        #     utils.populate_filters(channel)
        return user, channel

    def __delete_comment(self, comment_id):
        # https://developers.google.com/youtube/v3/docs/comments/delete#try-it
        """
            We don't want this function to be called directly.
            Because some checks are provided in the execute_action_on_comment method.
            This function is only called when the user has provided the credentials
        """

        request = self.private_youtube.comments().delete(
            id=comment_id
        )
        response = request.execute()
        logger.info(f'Deleting comment {comment_id}: {response}')
        return response

    def __moderate_comments(self, comment, publish=False):
        new_status = 'published' if publish else 'heldForReview'
        request = self.private_youtube.comments().setModerationStatus(
            id=comment.id, moderationStatus=new_status
        )
        response = request.execute()
        logger.info(f'{new_status} comment {comment.id}: {response}')
        return response


    def execute_action_on_comment(self, comment):
        # because the final action should be affected by various filters.
        new_action = comment.determine_status()
        
        if new_action == comment.status:
            # if the action is the same as the current status, we do not need to execute it
            logger.debug(f'No action needed for comment {comment.id}, current status: {comment.status}, new action: {new_action}')
            return
        
        """
            When the action is different from the current status, we will execute the action.
            In a debug mode, we will simply fake the action.
            In a production mode, we will actually execute the action, given the user provides the credentials.
        """
        if self.private_youtube is None:
            if settings.DJANGO_ENV == 'production':
                logger.info(f'No credentials provided, cannot actually execute action on comment {comment.id}')
                return
            else:
                logger.debug(f'Fake executing comment {comment.id} for {new_action}')
        else:
            if new_action == CommentStatus.DELETED:
                self.__delete_comment(comment.id)
            elif new_action == CommentStatus.PUBLISHED:
                self.__moderate_comments(comment, publish=True)
            elif new_action == CommentStatus.REVIEW:
                self.__moderate_comments(comment, publish=False)
            else:
                raise ValueError(f'Unknown action: {new_action}')
            
        comment.status = new_action
        comment.save()
            