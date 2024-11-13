from math import log
import re
from googleapiclient.discovery import build
import logging
from .models import User, Channel, Video, Comment
from . import utils

logger = logging.getLogger(__name__)
class YoutubeAPI:

    def __init__(self, credentials):
        if isinstance(credentials, dict):
            self.credentials = credentials
        else:
            self.credentials = utils.credentials_to_dict(credentials)
        # self.youtube = build('youtube', 'v3', credentials=credentials)
        self.youtube = build('youtube', 'v3', developerKey='AIzaSyBBdr6RyGr0fRnjAoHzn-NXRmwy_tiYL5A')

    def retrieve_channels(self):
        channel = self.youtube.channels().list(mine=True, part='snippet').execute()
        return channel

    def retrieve_account(self):
        account_info = self.youtube.channels().list(mine=True, part='snippet').execute()
        username = account_info['items'][0]['snippet']['title']
        channel = account_info['items'][0]['snippet']['customUrl']
        channelID = account_info['items'][0]['id']
        return {
            'username': username,
            'channel': {
                'name': channel,
                'channel_id': channelID
            }
        }

    def retrieve_videos(self, channel_id, video_num=5):
        request = self.youtube.search().list(
            part='snippet',
            channelId=channel_id,
            type='video',
            order='date'
        )
        response = request.execute()
        logger.info(f'There are {len(response["items"])} videos in the channel')
        videos = []
        for item in response['items'][:video_num]:
            video_id = item['id']['videoId']
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
            print(f'This comment has {parent_comment.total_replies} replies')
            replies = []
            reply_request = self.youtube.comments().list(
                part='snippet',
                parentId=parent_comment.id,
            )
            while reply_request:
                reply_response = reply_request.execute()
                for reply_item in reply_response.get('items', [])[:1]:
                    reply_comment = self.__process_comment(reply_item, parent_comment.video.id, parent_comment)
                    replies.append(reply_comment)
                # Handle pagination
                reply_request = self.youtube.comments().list_next(reply_request, reply_response)
        if len(replies) > 2:
            logger.info(f'Extracted {len(replies)} replies from the comment {parent_comment.id}')

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

    def retrieve_comments(self, video_id, comment_num=100):
        comment_request = self.youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
        )
        comment_response = comment_request.execute()
        logger.info(f'There are {len(comment_response["items"])} comments in the video')
        comments = []
        for comment_item in comment_response['items']:
            comment = self.__process_comment(comment_item['snippet'], video_id)
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
                comments.append(comment)
        logger.info(f'Extracted {len(comments)} comments from the video {video_id}')
        return comments
        
    def initialize_comments(self, channel, restart=False):
        videos = Video.objects.filter(channel=channel).all()
        # delete all existing videos and comments
        if restart:
            videos.delete()
        elif videos.exists():
            logger.info(f'Found {len(videos)} videos for the channel {channel.name}')
            return

        videos = self.retrieve_videos(channel.id)
        for video in videos:
            self.retrieve_comments(video.id)

    def create_account(self):

        account_info = self.retrieve_account()

        user = User(username=account_info['username'], oauth_credentials=self.credentials)
        user.save()

        channel = Channel(owner=user, **account_info['channel'])
        channel.save()

        self.initialize_comments(channel)
        return user, channel
    
