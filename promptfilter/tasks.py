from venv import logger
from celery import shared_task
import math
import random
from django.db.models import F
from torch import square_
from .models import PromptFilter, User, MistakeCluster, FilterPrediction
from .updates import update_predictions
from .youtube import YoutubeAPI
from .backend_filter import BackendPromptFilter
from .llm_buddy import LLMBuddy
from .utils import add_log, copy_filter
import time

@shared_task
def update_predictions_task(filter_id, mode, start_date, cached_predictions):
    logger.info(f"Updating predictions for filter {filter_id} with mode {mode}")
    filter = PromptFilter.objects.get(id=filter_id)
    predictions = update_predictions(filter, mode, start_date=start_date, cached_predictions=cached_predictions)
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
        # this is because wthe id of the mistake refers to the id of the comment.
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
    logger.info('#' * 100)
    logger.info(f"Calibrating prompt for filter {filter.name}")
    logger.info('#' * 100)

    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    annotations = FilterPrediction.objects.filter(
        filter=filter,
        groundtruth__isnull=False
    ).all()

    annotations = [annotation.serialize() for annotation in annotations]

    buddy = LLMBuddy()
    calibrated_filter = buddy.calibrate_prompt(backend_filter, annotations)
    logger.info('#' * 100)
    logger.info(f"Calibrated prompt for filter {filter.name} has been completed.")
    logger.info('#' * 100)
    calibrated_filter = calibrated_filter.serialize()
    logger.info(f"Calibrated filter: {calibrated_filter}")
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
    statistics = youtube.synchronize(user)
    logger.info(f"Synchronization for user {username} has been completed.")
    add_log(
        user, 'synchronize_youtube', 
        f"Synchronization for user {username} has been completed with {statistics['newCommentsCount'] } new comments and {statistics['newVideosCount']} new videos.",
    )
    return { 
        'message': 'Synchronization has been completed.',  
        'statistics': statistics
    }

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
    logger.info(f"Refining prompt for filter {filter.name} with {len(cluster['cluster'])} comments.")
    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    logger.info(f'This filter has {len(backend_filter.training_examples)} training examples.')

    buddy = LLMBuddy()
    refined_filters = buddy.refine_prompt(backend_filter, cluster, topN=3)
    logger.info(f"Refined prompt for filter {filter.name} has been completed.")
    return { 'refinedFilters': [filter.serialize() for filter in refined_filters] }

@shared_task
def automatic_iterations_baseline_task(filter_id):
    start_time = time.time()
    now_filter = PromptFilter.objects.filter(id=filter_id).first()
    if now_filter is None:
        logger.error(f"Filter with id {filter_id} does not exist.")
        return { 'error': f"Filter with id {filter_id} does not exist." }

    backend_filter = BackendPromptFilter.create_backend_filter(now_filter)
    annotations = FilterPrediction.objects.filter(
        filter=now_filter,
        groundtruth__isnull=False,
        experiment_type__in=['train', 'audit']
    ).all()

    annotations = [annotation.serialize() for annotation in annotations]
    logger.info(f"Automatic iterations baseline for filter {now_filter.name} with {len(annotations)} annotations.")

    from .optimize_prompt import PromptOptimizer
    optimizer = PromptOptimizer()
    calibrated_filter = optimizer.calibrate_prompt(backend_filter, annotations, rounds=3, beam_size=2)

    updated_filter = now_filter.update_filter(calibrated_filter.serialize())
    # apply the updated filter to all comments
    update_predictions(updated_filter, 'all')
    updated_filter.calibrated = True
    updated_filter.save()
    end_time = time.time()
    logger.info(f"Automatic iterations baseline for filter {now_filter.name} has been completed in {end_time - start_time} seconds.")
    return {
        'iteratedFilter': updated_filter.serialize()
    }



@shared_task
def experiment_calibrate_prompt_task(source_filter_id, whether_initialize=False):
    logger.info(f"Starting experiment calibration for filter ID {source_filter_id} with whether_initialize={whether_initialize}")
    from concurrent.futures import ThreadPoolExecutor
    filter = PromptFilter.objects.filter(id=source_filter_id).first()
    # check the calibrated filters already exist
    created_filters = PromptFilter.objects.filter(
        channel=filter.channel,
        approach__in=['circle', 'square']
    )
    if created_filters.count() == 2:
        if whether_initialize:
            logger.info(f"Re-initializing experiment filters for {filter.name}.")
            created_filters.delete()
        else:
            logger.info(f"Returning existing experiment filters for {filter.name}.")
            return {
                'createdFilters': [f.serialize() for f in created_filters]
            }

    def run_calibration(approach):
        logger.info(f"Checking/Creating experiment filter: {filter.name} [{approach}]")
        new_filter = copy_filter(filter, f"{filter.name} [{approach}]", restart=True)
        new_filter.approach = approach
        new_filter.save()

        backend_filter = BackendPromptFilter.create_backend_filter(new_filter)
        annotations = FilterPrediction.objects.filter(
            filter=new_filter,
            groundtruth__isnull=False,
            experiment_type='train'
        ).all()
        annotations = [annotation.serialize() for annotation in annotations]
        logger.info(f"Calibrating prompt in {approach} for experiment filters based on filter {new_filter.name} with {len(annotations)} annotations.")

        if approach == 'circle':
            from .optimize_prompt import PromptOptimizer
            optimizer = PromptOptimizer()
            calibrated_filter = optimizer.calibrate_prompt(backend_filter, annotations, rounds=3, beam_size=2)
        else: # square
            buddy = LLMBuddy()
            calibrated_filter = buddy.calibrate_prompt(backend_filter, annotations, rounds=3, beam_size=2)

        updated_filter = new_filter.update_filter(calibrated_filter.serialize())
        # apply the updated filter to all comments
        update_predictions(updated_filter, 'all')
        updated_filter.calibrated = True
        updated_filter.save()
        return updated_filter

    new_filters = []
    if whether_initialize:
        new_filters.append(run_calibration('circle'))
        square_filter = run_calibration('square')
        new_filters.append(square_filter)
    else:
        circle_filter = copy_filter(filter, f"{filter.name} [Circle]", restart=True)
        circle_filter.approach = 'circle'
        circle_filter.save()

        square_filter = copy_filter(filter, f"{filter.name} [Square]", restart=True)
        square_filter.approach = 'square'
        square_filter.save()

        new_filters = [circle_filter, square_filter]
        time.sleep(20)

    return {
        'createdFilters': [f.serialize() for f in new_filters]
    }