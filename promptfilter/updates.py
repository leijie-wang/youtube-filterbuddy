from datetime import datetime
from math import log
from django.db.models import Q
import logging
from .models import Comment, FilterPrediction, MistakeCluster
from .backend_filter import BackendPromptFilter
from .youtube import YoutubeAPI

logger = logging.getLogger(__name__)

def update_predictions(filter, mode, now_synchronized=None, start_date=None, cached_predictions=None):
    """When we call this function, we want to also execute the corresponding action for each True prediction.

        @mode: we have various modes to make sure we do not waste too much computational resources
            'new': only update comments that have been posted after the last run;
            'all': update all comments;
            'initialize': when the user first creates the filter, we will randomly sample 100 comments to preview the filter;
            'iteration': when the user wants to iterate on their filters.
    """

    comments = filter.retrieve_update_comments(mode, start_date)

    # We should not update the last_run time in the iteration mode
    # TODO: think about whether we should update the last_run time in the initialize mode
    if mode not in ['initialize', 'iteration', 'refresh']:
        filter.last_run = datetime.now() if now_synchronized is None else now_synchronized
        filter.save()

    # we still want to update the last run even if the length of comments is 0; 
    # for instance, when the user selects future comments.
    if len(comments) == 0:
        return None
    logger.info(f'Filter {filter.name} has {len(comments)} comments at the mode {mode}.')
    comments = [comment.serialize() for comment in comments]
    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    comments_with_preds = backend_filter.predict_comments_consistently(comments, cached_predictions=cached_predictions)

    # update the predictions in the database and execute the corresponding action
    youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
    predictions = []
    logger.info(f'Filter {filter.name} has {len(comments_with_preds)} comments with predictions to update.')
    for comment in comments_with_preds:
        # update the prediction in the database;
        existing_prediction = FilterPrediction.objects.filter(
            filter=filter,
            comment_id=comment['id']
        ).first()
        
        update_defaults = {
            'prediction': comment['prediction'],
            'confidence': comment['confidence'],
            'explanation': ''
        }
        
        # if the prediction has changed, we will also empty the reflection; 
        # otherwise, we expect the old reflection to be still applicable
        if not existing_prediction or existing_prediction.prediction != comment['prediction']:
            update_defaults['reflection'] = ''

        prediction, created = FilterPrediction.objects.update_or_create(
            filter=filter,
            comment_id=comment['id'],
            defaults=update_defaults
        )
        if mode not in ['initialize', 'iteration']:
            # to avoid back and forth actions in the iteration mode
            youtube.execute_action_on_comment(prediction.comment)
        predictions.append(prediction.serialize())

    # remove all old mistake clusters for this filter
    # because we assume calculating the mistake clusters itself is less expensive
    # compared to updating the reflection for each prediction
    MistakeCluster.objects.filter(filter=filter).delete()
    return predictions


def update_actions(filter, start_date=None):
    youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
    # retrieve comments that have been affected by this particular filter
    predictions = FilterPrediction.objects.filter(
        filter=filter, prediction=True
    ).filter(
        Q(groundtruth=True) | Q(groundtruth__isnull=True)
    ).all()
    if start_date is not None:
        predictions = predictions.filter(comment__posted_at__gt=start_date)
    
    logger.info(f'Filter {filter.name} has {len(predictions)} comments to update actions.')
    for prediction in predictions:
        youtube.execute_action_on_comment(prediction.comment)






