from venv import logger
from celery import shared_task
from .models import PromptFilter
from .utils import update_predictions, predict_comments

@shared_task
def update_predictions_task(filter_id, mode):
    logger.info(f"Updating predictions for filter {filter_id} with mode {mode}")
    filter = PromptFilter.objects.get(id=filter_id)
    predictions = update_predictions(filter, mode)
    logger.info(f"Predictions for filter {filter_id} have been updated.")
    return predictions

@shared_task
def predict_comments_task(filter, comments):
    logger.info(f"Predicting comments for filter {filter} with {len(comments)} comments.")
    predictions = predict_comments(filter, comments)
    logger.info(f"Predictions for filter {filter['description']} on {len(comments)} comments have been completed.")
    return predictions