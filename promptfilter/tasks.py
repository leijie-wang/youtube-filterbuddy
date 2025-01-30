from venv import logger
from celery import shared_task
from .models import PromptFilter, User, MistakeCluster, FilterPrediction
from .updates import update_predictions
from .youtube import YoutubeAPI
from .backend_filter import BackendPromptFilter
from .llm_buddy import LLMBuddy

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
def generate_clusters_task(filter_id, mistakes):
    logger.info(f"Generating clusters for filter {filter_id} with {len(mistakes)} mistakes.")
    
    filter = PromptFilter.objects.get(id=filter_id)
    backend_filter = BackendPromptFilter.create_backend_filter(filter)

    buddy = LLMBuddy()
    reflections = buddy.reflect_on_mistakes_parellel(backend_filter, mistakes)

    # Iterate and update each mistake instance
    for index, mistake in enumerate(mistakes):
        reflection = reflections[index]
        
        # Find the prediction instance using filter and comment ID
        prediction_instance = FilterPrediction.objects.filter(
            filter=filter,
            comment_id=mistake['id']
        ).first()

        # Update reflection if the prediction exists
        if prediction_instance and 'reflection' in reflection:
            prediction_instance.reflection = reflection['reflection']
            prediction_instance.save()  # Save changes immediately
        else:
            logger.warning(f"Prediction instance not found for comment {mistake['id']}.")
        
    clusters = buddy.generate_interesting_clusters(backend_filter, mistakes)
    if len(clusters) > 0:
        summary = buddy.interpret_refine_infos(backend_filter, clusters[0])
        clusters[0]['summary'] = summary
    
    for cluster in clusters:
        # cache these clusters
        MistakeCluster.create_cluster(filter, cluster)

    logger.info(f"Clusters for filter {filter.name} on {len(mistakes)} mistakes have been completed.")
    return { 'clusters': clusters }

@shared_task
def synchronize_youtube_task(username):
    logger.info(f"Synchronizing youtube for user {username}")
    user = User.objects.get(username=username)
    youtube = YoutubeAPI(user.oauth_credentials)
    youtube.synchronize(user)
    logger.info(f"Synchronization for user {username} has been completed.")
    return { 'message': 'Synchronization has been completed.' }