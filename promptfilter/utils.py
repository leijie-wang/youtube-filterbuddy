import logging
import random
from datetime import timedelta, datetime
from django.utils import timezone
from .models import User, Channel, PromptFilter, Comment, FilterPrediction
from .llm_filter import LLMFilter

logger = logging.getLogger(__name__)

def credentials_to_dict(credentials):
  return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

def random_time(zerotime=None):
    if zerotime is None:
        zerotime = timezone.now() - timedelta(days=30)
    days_after = random.randint(0, 30)
    random_time = zerotime + timedelta(days=days_after, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    return random_time

def populate_test_users():
    user = User(
        username='TheYoungTurks', 
        avatar='https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
        oauth_credentials={
            'token': 'FAKE_TOKEN',
            'refresh_token': "FAKE_REFRESH_TOKEN",
            'token_uri': 'https://127.0.0.1',
            'client_id': 'FAKE_CLIENT_ID',
            'client_secret': 'FAKE_CLIENT_SECRET',
            'scopes': [],
            'myChannelId': 'UC1yBKRuGpC1tSM73A0ZjYjQ'
        }
    )
       
    user.save()
    print(f"User {user.username} has been successfully created!")
    print(f"There are {User.objects.all().count()} users in the database.")

    channel = Channel(owner=user, name='The Young Turks', id='UC1yBKRuGpC1tSM73A0ZjYjQ')
    channel.save()
    print(f"Channel {channel.name} has been successfully created!")
    print(f"There are {Channel.objects.all().count()} channels in the database.")
    populate_filters(channel)
    return user

def populate_filters(channel):
    if PromptFilter.objects.filter(channel=channel).exists():
        print("Prompt filters already exist for the channel.")
        return

    prompt_filters = [
        {"name": "Sexually Explicit Content", "description": "Comments that contain sexually explicit or inappropriate content not suitable for public viewing."},
        {"name": "Spam", "description": "Comments that are repetitive, irrelevant, or promotional in nature."},
        # {"name": "Off-Topic", "description": "Comments that are unrelated to the video content or discussion."},
    ]
    for info in prompt_filters:
        filter = PromptFilter(channel=channel, name=info['name'], description=info['description'])
        filter.save()
    print(f"{len(prompt_filters)} prompt filters have been successfully created!")
    print(f"There are {PromptFilter.objects.all().count()} prompt filters in the database.")

def predict_comments(filter, comments):
    """Predict the comments using the filter.
    
    Args:
        filter (dict): The filter information.
        comments (list): The list of comments to predict.
    """

    datasets = [comment['content'] for comment in comments]
    llm_filter = LLMFilter({
            'name': filter['name'],
            'description': filter['description'],
        }, debug=False
    )
    predictions = llm_filter.predict(datasets)
    for index, comment in enumerate(comments):
        # make sure it is true or false
        comment['prediction'] = predictions[index] == 1

    # summarize the predictions
    positive_num = sum(predictions)
    negative_num = len(predictions) - positive_num
    print(f'There are {positive_num} positive predictions and {negative_num} negative predictions.')
    return comments

def update_predictions(filter, mode):
    # randomly sample comments from the database to begin with
    comments = Comment.objects.filter(video__channel=filter.channel).order_by('posted_at')
    if not comments.exists():
        return None
    logger.info(f'Filter {filter.name} has {comments.count()} comments.')
    if mode == 'new' and filter.last_run:
        # select comments that appear after the last run
        comments = comments.filter(posted_at__gt=filter.last_run)
    else:
        comments = list(comments.all())
    filter.last_run = datetime.now()
    filter.save()

    comments = [comment.serialize() for comment in comments]
    comments_with_preds = predict_comments(filter.serialize(), comments)

    
    for comment in comments_with_preds:
        # update the prediction in the database
        FilterPrediction.objects.update_or_create(
            filter=filter,
            comment_id=comment['id'],
            defaults={'prediction': comment['prediction']}
        )
    # in addition identify the groundtruth attribute of each comment
    comments = []
    for comment in comments_with_preds:
        comments.append(
            FilterPrediction.objects.get(filter=filter, comment_id=comment['id']).serialize()
        )

    return comments

def determine_new_comments(comments, time_cutoff):
    """Determine the new comments based on the time cutoff.
    
    Args:
        comments (list): The list of comments to filter.
        time_cutoff (datetime): The time cutoff.
    """
    for comment in comments:
        comment['new'] = time_cutoff is None or comment['posted_at'] > time_cutoff
    return comments

def retrieve_predictions(filter):
    predictions = FilterPrediction.objects.filter(filter=filter).order_by('-comment__posted_at')
    logger.info(f"predictions: {len(predictions)}")
    comments = [prediction.serialize() for prediction in predictions]
    comments = determine_new_comments(comments, filter.channel.owner.second_last_sync)
    return comments