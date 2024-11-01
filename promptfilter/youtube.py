from googleapiclient.discovery import build
from .models import User, Channel
from . import utils

class YoutubeAPI:

    def __init__(self, credentials):
        self.credentials = utils.credentials_to_dict(credentials)
        self.youtube = build('youtube', 'v3', credentials=credentials)

    def retrieve_channels(self):
        channel = self.youtube.channels().list(mine=True, part='snippet').execute()
        return channel

    def retrieve_account_info(self):
        account_info = self.youtube.channels().list(mine=True, part='snippet').execute()
        username = account_info['items'][0]['snippet']['title']
        channel = account_info['items'][0]['snippet']['customUrl']
        return {
            'username': username,
            'channel': channel
        }

    def create_account(self):
        account_info = self.retrieve_account_info()

        user = User(username=account_info['username'], oauth_credentials=self.credentials)
        user.save()

        channel = Channel(owner=user, name=account_info['channel'])
        channel.save()
        return user, channel
