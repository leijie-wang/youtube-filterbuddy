from venv import logger
from celery import shared_task
from .models import PromptFilter, User
from .updates import update_predictions
from .youtube import YoutubeAPI
from .backend_filter import BackendPromptFilter

@shared_task
def update_predictions_task(filter_id, mode):
    logger.info(f"Updating predictions for filter {filter_id} with mode {mode}")
    filter = PromptFilter.objects.get(id=filter_id)
    predictions = update_predictions(filter, mode)
    logger.info(f"Predictions for filter {filter_id} have been updated.")
    return { 'predictions': predictions }

@shared_task
def predict_comments_task(filter, comments):
    logger.info(f"Predicting comments for filter {filter} with {len(comments)} comments.")
    backend_filter = BackendPromptFilter(**filter)
    predictions = backend_filter.predict_comments_consistently(comments)
    logger.info(f"Predictions for filter {filter['description']} on {len(comments)} comments have been completed.")
    return { 'predictions': predictions }

@shared_task
def synchronize_youtube_task(username):
    logger.info(f"Synchronizing youtube for user {username}")
    user = User.objects.get(username=username)
    youtube = YoutubeAPI(user.oauth_credentials)
    youtube.synchronize(user)
    logger.info(f"Synchronization for user {username} has been completed.")
    return { 'message': 'Synchronization has been completed.' }