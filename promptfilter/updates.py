from datetime import datetime
from django.db.models import Q
import logging
from .models import Comment, FilterPrediction, MistakeCluster
from .backend_filter import BackendPromptFilter
from .youtube import YoutubeAPI

logger = logging.getLogger(__name__)

def update_predictions(filter, mode, now_synchronized=None):
    """When we call this function, we want to also execute the corresponding action for each True prediction.

        @mode: we have various modes to make sure we do not waste too much computational resources
            'new': only update comments that have been posted after the last run;
            'all': update all comments;
            'initialize': when the user first creates the filter, we will randomly sample 100 comments to preview the filter;
            'iteration': when the user wants to iterate on their filters.
    """

    comments = filter.retrieve_update_comments(mode)
    if len(comments) == 0:
        return None
    logger.info(f'Filter {filter.name} has {len(comments)} comments at the mode {mode}.')
    
    # TODO: wondering whether we should update the last_run at the mode of 'initialize' or 'iteration'
    filter.last_run = datetime.now() if now_synchronized is None else now_synchronized
    filter.save()

    comments = [comment.serialize() for comment in comments]
    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    comments_with_preds = backend_filter.predict_comments_consistently(comments)

    # update the predictions in the database and execute the corresponding action
    youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
    predictions = []
    for comment in comments_with_preds:
        # update the prediction in the database;
        # we will empty the reflection because it might not be applicable for the new filter
        # TODO: we might want to conditionally remove the reflection
        prediction, created = FilterPrediction.objects.update_or_create(
            filter=filter,
            comment_id=comment['id'],
            defaults={
                'prediction': comment['prediction'],
                'confidence': comment['confidence'],
                'reflection': ''
            }
        )
        if mode not in ['initialize', 'iteration']:
            # to avoid back and forth actions in the iteration mode
            youtube.execute_action_on_prediction(prediction)
        predictions.append(prediction.serialize())

    # remove all old mistake clusters for this filter
    MistakeCluster.objects.filter(filter=filter).delete()
    return predictions


def update_actions(filter, mode):
    youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
    predictions = FilterPrediction.objects.filter(
        filter=filter, prediction=True
    ).filter(
        Q(groundtruth=True) | Q(groundtruth__isnull=True)
    ).all()
    if mode == 'new' and filter.last_run:
        predictions = predictions.filter(comment__posted_at__gt=filter.last_run)
    for prediction in predictions:
        youtube.execute_action_on_prediction(prediction)






