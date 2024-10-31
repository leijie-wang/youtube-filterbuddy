from venv import logger
from celery import shared_task
from .models import PromptFilter

@shared_task
def update_predictions_task(filter_id, mode):
    logger.info(f"Updating predictions for filter {filter_id} with mode {mode}")
    filter = PromptFilter.objects.get(id=filter_id)
    filter.update_predictions(mode)
    logger.info(f"Predictions for filter {filter_id} have been updated.")