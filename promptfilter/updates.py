from datetime import datetime
from django.db.models import Q
import logging
from .models import Comment, FilterPrediction
from .backend_filter import BackendPromptFilter
from .youtube import YoutubeAPI

logger = logging.getLogger(__name__)


def update_predictions(filter, mode, now_synchronized=None):
    """When we call this function, we want to also execute the corresponding action for each True prediction."""

    comments = filter.retrieve_update_comments(mode)
    if len(comments) == 0:
        return None
    logger.info(f'Filter {filter.name} has {len(comments)} comments at the mode {mode}.')
    
    filter.last_run = datetime.now() if now_synchronized is None else now_synchronized
    filter.save()

    comments = [comment.serialize() for comment in comments]
    backend_filter = BackendPromptFilter.create_backend_filter(filter)


    comments_with_preds = backend_filter.predict_comments_consistently(comments)

    # update the predictions in the database and execute the corresponding action
    youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
    predictions = []
    for comment in comments_with_preds:
        # update the prediction in the database
        prediction, created = FilterPrediction.objects.update_or_create(
            filter=filter,
            comment_id=comment['id'],
            defaults={'prediction': comment['prediction']}
        )
        youtube.execute_action_on_prediction(prediction)
        predictions.append(prediction.serialize())

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






