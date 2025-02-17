import logging
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import timedelta
from django.utils import timezone
from django.db.models import Q, F, Case, When, Value, IntegerField

from .models import User, Channel, PromptFilter, Comment, FilterPrediction


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


def populate_fake_credentials(channel_id):
    return {
        'token': 'FAKE_TOKEN',
        'refresh_token': "FAKE_REFRESH_TOKEN",
        'token_uri': 'https://127.0.0.1',
        'client_id': 'FAKE_CLIENT_ID',
        'client_secret': 'FAKE_CLIENT_SECRET',
        'scopes': [],
        'myChannelId': channel_id
    }

def populate_test_users():
    user = User(
        username='@TheYoungTurks', 
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

    # prompt_filters = [
    #     {"name": "Sexually Explicit Content", "description": "Comments that contain sexually explicit or inappropriate content not suitable for public viewing."},
    #     {"name": "Spam", "description": "Comments that are repetitive, irrelevant, or promotional in nature."},
    #     # {"name": "Off-Topic", "description": "Comments that are unrelated to the video content or discussion."},
    # ]
    prompt_filters = [
        {"name": "Political Hate Speech", "description": "Comments that use derogatory insults towards specific political groups"},
    ]
    
    for info in prompt_filters:
        filter = PromptFilter(channel=channel, name=info['name'], description=info['description'])
        filter.save()
    print(f"{len(prompt_filters)} prompt filters have been successfully created!")
    print(f"There are {PromptFilter.objects.all().count()} prompt filters in the database.")

def determine_new_comments(comments, time_cutoff):
    """Determine the new comments based on the time cutoff.
    
    Args:
        comments (list): The list of comments to filter.
        time_cutoff (datetime): The time cutoff.
    """
    for comment in comments:
        comment['new'] = time_cutoff is None or comment['posted_at'] > time_cutoff
    return comments

def determine_time_cutoff(filter):
    user = filter.channel.owner
    if filter.last_run and filter.last_run == user.last_sync:
        # if the filter was synchronized when the whole channel was synchronized,
        # then we want to highlight new comments that appear after the second last synchronization
        compare_time = user.second_last_sync
    else:
        # otherwise, all comments are new
        # because this filter should be just initialized.
        compare_time = None
    return compare_time

def retrieve_predictions(filter, whether_iterate):
    N = 200
    
    if not whether_iterate:
        # if the mode is not iteration, we want to retrieve all the predictions
        predictions = FilterPrediction.objects.filter(filter=filter).order_by('-comment__posted_at')
        logger.info(f"predictions: {len(predictions)}")
    else:
        # if the mode is iteration, we want to retrieve predictions that have groundtruths
        predictions = (
            FilterPrediction.objects
            .filter(filter=filter, groundtruth__isnull=False, prediction__isnull=False)
            # True sorts higher than False if we do descending order on the Boolean field
            .order_by('-prediction', '-comment__posted_at')
        )
        logger.info(f"predictions at the iteration mode: {len(predictions)}")
        # if there are too many predictions, we want to sample the first N predictions
        predictions = predictions[:N]
       
    
    comments = [prediction.serialize() for prediction in predictions]

    compare_time = determine_time_cutoff(filter)
    comments = determine_new_comments(comments, compare_time)
    return comments

def number_of_new_comments(filter):
    """
        We want to count the number of new comments since the second last synchronization
    """
    compare_time = determine_time_cutoff(filter)
    predicitions = FilterPrediction.objects.filter(filter=filter, prediction=True)
    if compare_time is not None:
        predicitions = predicitions.filter(comment__posted_at__gt=compare_time)
    return predicitions.count()

def recalculate_confidence(old_pred, new_pred, weight=1):
    """
    Recalculate the confidence of the prediction based on the new prediction.
    Args:
        old_pred (dict): The old prediction with keys 'prediction' and 'confidence'.
        new_pred (dict): The new prediction with keys 'prediction' and 'confidence'.
        weight (float): The weight to give to the new prediction.

    Returns:
        dict: The new prediction with recalculated confidence in a dictionary.
    """
    if old_pred['prediction'] is None:
        return new_pred['prediction'], new_pred['confidence']

    support_for_old = old_pred['confidence']
    support_for_other = 1 - old_pred['confidence']

    if new_pred['prediction'] == old_pred['prediction']:
        support_for_old += weight * new_pred['confidence']
        support_for_other += weight * (1 - new_pred['confidence'])
    else:
        support_for_other += weight * new_pred['confidence']
        support_for_old += weight * (1 - new_pred['confidence'])

    # Determine which prediction now has more support
    total_support = support_for_old + support_for_other

    if support_for_old >= support_for_other:
        updated_prediction = old_pred['prediction']
        updated_confidence = support_for_old / total_support
    else:
        updated_prediction = 1 - old_pred['prediction']
        updated_confidence = support_for_other / total_support

    return updated_prediction, updated_confidence

def retrieve_rubric_info(filter, rubric_index):
    # make sure rubric index as a string can be converted to an integer
    try:
        rubric_index = int(rubric_index)
    except ValueError:
        raise ValueError(f"Error: rubric_index '{rubric_index}' is not convertible to an integer.")

    if 'positives' not in filter or 'negatives' not in filter:
        raise KeyError("Error: 'rubric_filter' must contain both 'positives' and 'negatives' keys.")

    if rubric_index >= len(filter['positives']):
        return 'negatives', rubric_index - len(filter['positives'])
    else:
        return 'positives', rubric_index
    
def clean_comments(comments):
    new_comments = []
    for comment in comments or []:
        new_comments.append(
            {
                'content': comment['content'],
                'groundtruth': comment['groundtruth'],
                'reflection': comment.get('reflection', None)
            }
        )
    return new_comments

def eval_performance(now_comments, print_comments=False):
    # measure the performance in terms of accuracy, precision, recall, and F1 score
    # Extract ground truth and predictions
    y_true = [comment['groundtruth'] for comment in now_comments]
    y_pred = [comment['prediction'] for comment in now_comments]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    # print out the number of mistakes 
    mistakes = [comment for comment in now_comments if comment['groundtruth'] != comment['prediction']]
    false_positives = [comment for comment in mistakes if comment['groundtruth'] == 0 and comment['prediction'] == 1]
    false_negatives = [comment for comment in mistakes if comment['groundtruth'] == 1 and comment['prediction'] == 0]
    print(f"\nNumber of mistakes: {len(mistakes)}; False Positives: {len(false_positives)}; False Negatives: {len(false_negatives)}")

    if print_comments:
        # print first false positives
        print("\nFalse Positives:")
        for comment in now_comments:
            if comment['groundtruth'] == 0 and comment['prediction'] == 1:
                print(f"\tConfidence: {comment['confidence']}\tComment:\t{comment['content']}\n")
        # then print false negatives
        print("\nFalse Negatives:")
        for comment in now_comments:
            if comment['groundtruth'] == 1 and comment['prediction'] == 0:
                print(f"\tConfidence: {comment['confidence']}\tComment:\t{comment['content']}\n")
                # explanation = llm_buddy.explain_prediction(filter, comment)
                # print(f"\tExplanation:\t{explanation}\n")
        # then print true positives
        print("\nTrue Positives:")
        for comment in now_comments:
            if comment['groundtruth'] == 1 and comment['prediction'] == 1:
                print(f"\tConfidence: {comment['confidence']}\t{comment['content']}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def sample_diverse_embeddings(embeddings, k):
    n_samples = embeddings.shape[0]
    if k > n_samples:
        raise ValueError("k cannot be larger than the number of available samples.")
    selected_indices = [np.random.randint(n_samples)]
    min_distances = cdist(embeddings, embeddings[selected_indices], metric='euclidean').flatten()
    for _ in range(1, k):
        # Select the point with the maximum minimum distance to the selected set
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)
        
        # Update the minimum distances
        distances = cdist(embeddings, embeddings[[next_index]], metric='euclidean').flatten()
        min_distances = np.minimum(min_distances, distances)
    return selected_indices

def deduplicate_filters(filters):
    unique_content_sets = {}
    deduped_filters = []

    for filter_idx, filter_obj in enumerate(filters):
        # Validate that 'few_shots' exists and is a list
        if not hasattr(filter_obj, 'few_shots') or not isinstance(filter_obj.few_shots, list):
            raise ValueError(f"Filter at index {filter_idx} lacks a valid 'few_shots' attribute.")

        # Extract the set of 'content' values
        try:
            content_set = frozenset(item['content'] for item in filter_obj.few_shots)
        except KeyError as e:
            raise KeyError(f"Missing key in 'few_shots' for filter at index {filter_idx}: {e}")

        # Check if this content_set is already encountered
        if content_set not in unique_content_sets:
            unique_content_sets[content_set] = filter_obj
            deduped_filters.append(filter_obj)
        else:
            # Duplicate found; you can choose to log or handle duplicates here
            print(f"Duplicate found: Filter at index {filter_idx} is a duplicate of another filter.")

    return deduped_filters