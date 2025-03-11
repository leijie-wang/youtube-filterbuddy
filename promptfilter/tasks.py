from venv import logger
from celery import shared_task
import math
import random
from django.db.models import F
from .models import PromptFilter, User, MistakeCluster, FilterPrediction
from .updates import update_predictions
from .youtube import YoutubeAPI
from .backend_filter import BackendPromptFilter
from .llm_buddy import LLMBuddy

@shared_task
def update_predictions_task(filter_id, mode, start_date):
    logger.info(f"Updating predictions for filter {filter_id} with mode {mode}")
    filter = PromptFilter.objects.get(id=filter_id)
    predictions = update_predictions(filter, mode, start_date=start_date)
    logger.info(f"Predictions for filter {filter.name} have been updated.")
    return { }

@shared_task
def predict_comments_task(filter, comments):
    logger.info(f"Predicting comments for filter {filter} with {len(comments)} comments.")
    backend_filter = BackendPromptFilter(**filter)
    predictions = backend_filter.predict_comments_consistently(comments)
    logger.info(f"Predictions for filter {filter['description']} on {len(comments)} comments have been completed.")
    return { 'predictions': predictions }

@shared_task
def generate_clusters_task(filter_id):
    
    
    filter = PromptFilter.objects.get(id=filter_id)
    backend_filter = BackendPromptFilter.create_backend_filter(filter)

    mistake_instances = FilterPrediction.objects.filter(
        groundtruth__isnull=False, filter=filter
    ).exclude(
        groundtruth=F('prediction')
    )
    mistakes = [mistake.serialize() for mistake in mistake_instances]
    logger.info(f"Generating clusters for filter {filter.name} with {len(mistakes)} mistakes.")
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
            prediction_instance.save()
        else:
            logger.warning(f"Prediction instance not found for comment {mistake['id']}.")
        
    clusters = buddy.generate_interesting_clusters(backend_filter, mistakes)
    if len(clusters) > 0:
        summary = buddy.interpret_refine_infos(backend_filter, clusters[0])
        clusters[0]['summary'] = summary
    
    # we want to return the version with ids.
    cluster_instances = []
    for cluster in clusters:
        # cache these clusters
        inst = MistakeCluster.create_cluster(filter, cluster)
        cluster_instances.append(inst.serialize())
    
    logger.info(f"Clusters for filter {filter.name} on {len(mistakes)} mistakes have been completed with {len(clusters)} clusters.")
    return { 'clusters': cluster_instances }

@shared_task
def calibrate_prompt_task(filter_id):
    filter = PromptFilter.objects.get(id=filter_id)
    logger.info(f"Calibrating prompt for filter {filter.name}")

    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    annotations = FilterPrediction.objects.filter(
        filter=filter,
        groundtruth__isnull=False
    ).all()

    annotations = [annotation.serialize() for annotation in annotations]

    buddy = LLMBuddy()
    calibrated_filter = buddy.calibrate_prompt(backend_filter, annotations)
    logger.info(f"Calibrated prompt for filter {filter.name} has been completed.")

    calibrated_filter = calibrated_filter.serialize()
    filter = filter.update_filter(calibrated_filter)
    # update predictions
    update_predictions(filter, 'refresh')
    return { 'calibratedFilter': filter.serialize() }


@shared_task
def select_interesting_comments_task(filter_id, needed_num):
    buddy = LLMBuddy()
    # sample a balanced set of positive and negative comments
    filter = PromptFilter.objects.filter(id=filter_id).first()
    logger.info(f"Selecting interesting comments for filter {filter.name}")
    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    interesting_comments = buddy.select_interesting_comments(backend_filter, N = math.ceil(needed_num / 2))
    # randomize the order of the comments
    interesting_comments = random.sample(interesting_comments, len(interesting_comments))
    logger.info(f"Selected {len(interesting_comments)} interesting comments for filter {filter.name}")
    return { 'interestingComments': interesting_comments }

@shared_task
def synchronize_youtube_task(username):
    logger.info(f"Synchronizing youtube for user {username}")
    user = User.objects.get(username=username)
    youtube = YoutubeAPI(user.oauth_credentials)
    youtube.synchronize(user)
    logger.info(f"Synchronization for user {username} has been completed.")
    return { 'message': 'Synchronization has been completed.' }

@shared_task
def sync_all_youtube_accounts():
    logger.info(f"Synchronizing all youtube accounts")
    users = User.objects.all()
    for user in users:
        youtube = YoutubeAPI(user.oauth_credentials)
        youtube.synchronize(user)
    logger.info(f"Synchronization for all {len(users)} users has been completed.")
    return { 'message': 'Synchronization for all users has been completed.' }

@shared_task
def refine_prompt_task(filter_id, cluster):
    
    filter = PromptFilter.objects.filter(id=filter_id).first()
    logger.info(f"Refining prompt for filter {filter.name} with {len(cluster)} comments.")
    backend_filter = BackendPromptFilter.create_backend_filter(filter)

    buddy = LLMBuddy()
    refined_filter = buddy.refine_prompt(backend_filter, cluster)
    logger.info(f"Refined prompt for filter {filter.name} has been completed.")
    return { 'refinedFilter': refined_filter.serialize() }