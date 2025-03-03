
import json
import logging
from math import log
import os
import random
from django.db.models import F
from django.db.utils import OperationalError
from django.http import JsonResponse, HttpResponseRedirect
from django.middleware.csrf import get_token
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from functools import wraps
import google_auth_oauthlib.flow

from . import tasks
from . import updates
from . import utils
from .backend_filter import BackendPromptFilter
from .youtube import YoutubeAPI
from .llm_buddy import LLMBuddy
from .models import Channel, PromptFilter, FilterPrediction, Comment, User, MistakeCluster
from .backend_filter import BackendPromptFilter




logger  = logging.getLogger(__name__)
buddy = LLMBuddy()
FRONTEND_URL = os.getenv("FRONTEND_URL", "localhost:3001")
IS_LOCAL = os.getenv("IS_LOCAL", "True") == "True"

@csrf_exempt
def csrf_token_view(request):
    # Generate and return the CSRF token
    token = get_token(request)
    return JsonResponse({'csrfToken': token})

def verify_user(request):
    if 'credentials' in request.session and 'myChannelId' in request.session['credentials']:
        channel_id = request.session['credentials']['myChannelId']
        channel = Channel.objects.filter(id=channel_id).first()
        if channel:
            return True
    return False

def user_verification_required(view_func):
    """Decorator that checks user verification before calling the view function."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not verify_user(request):
            return JsonResponse({
                'error': 'User verification failed',
                'message': 'User credentials or channel information is missing or invalid.'
            }, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def __authorize_oauth():
    # for authentication with oauth
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        'client_secret.json',
        scopes=['https://www.googleapis.com/auth/youtube.force-ssl']
    )
    flow.redirect_uri = 'https://api.youtube.filterbuddypro.com/oauth2callback/'
    authorization_url, state = flow.authorization_url(
        # Recommended, enable offline access so that you can refresh an access token without
        # re-prompting the user for permission. Recommended for web server apps.
        access_type='offline',
        # Optional, enable incremental authorization. Recommended as a best practice.
        include_granted_scopes='true',
        # Optional, set prompt to 'consent' will prompt the user for consent
        prompt='consent'
    )
    # redirect users to the given url
    return JsonResponse({'redirectUrl': authorization_url, 'isAuthorized': False})

@user_verification_required
def switch_mode(request):
    request_data = json.loads(request.body)
    current_mode = request_data.get('current_mode', 'auto')
    channel_id = request.session['credentials']['myChannelId']

    channel = Channel.objects.filter(id=channel_id).first()
    user = channel.owner
    
    logger.info(f'The user {user.username} wants to switch from {current_mode} mode.')
    if current_mode == 'auto':
        user.moderation_access = False
        # reset the oauth credentials to a fake one
        user.oauth_credentials = utils.populate_fake_credentials(channel_id)
        user.save()
        return JsonResponse(
            {
                'message': f"User {user.username} has been switched to manual mode."
            }, safe=False
        )
    elif current_mode == 'manual':
        if IS_LOCAL:
            user.moderation_access = True
            user.save()
            return JsonResponse(
                {
                    'message': f"User {user.username} has been switched to auto mode, but we cannot authorize the user in a local environment."
                }, safe=False
            )
        
        # in this case, the moderation access will be updated when creating the account.
        return __authorize_oauth()

def whether_account_exists(request):
    request_data = json.loads(request.body)
    handle = request_data.get('handle', None)
    if handle is not None:
        user = User.objects.filter(username=handle).first()
        return JsonResponse(
            {
                'exists': user is not None,
            }, safe=False
        )
    else:
        return JsonResponse(
            {
                'exists': False,
            }, safe=False
        )
            
def authorize_user(request):
    # for the test user
    request_data = json.loads(request.body)

    # if the user has already authorized
    if verify_user(request):
        channel_id = request.session['credentials']['myChannelId']
        channel = Channel.objects.filter(id=channel_id).first()
        user = channel.owner

        return JsonResponse(
            {
                'username': user.username,
                'channel': channel.name,
                'isAuthorized': True,
                'redirectUrl': f'{FRONTEND_URL}/overview',
            }, safe=False
        )

    """
        whether test:
            - if true, then users want to authenticate via YouTube API
            - if false, then we will fake the authentication
            - if null, then we will use the default behavior from previous settings as the user has already logged in before.
    """
    whether_test = request_data.get('whether_test', False)
    handle = request_data.get('handle', None)
    
    if whether_test is not True and handle is None:
        logger.warning("We cannot authorize the user locally without a handle.")
        return JsonResponse(
            {
                'message': "We cannot authorize the user locally without a handle.",
            }, safe=False
        )
    
    if whether_test is None:
        user = User.objects.filter(username=handle).first()
        if user is not None:
            # if the user has already authorized, then we will use the previous settings
            whether_test = not user.moderation_access
    logger.info(f'We try to authorize the user {handle} with the test mode {whether_test}.')
    # for authentication without oauth
    if whether_test is True:
        user = User.objects.filter(username=handle).first()
        if not user:
            # create a test user
            youtube = YoutubeAPI({})
            user, channel = youtube.create_account(oauth=False, handle=handle)
        else:
            channel = Channel.objects.filter(owner=user).first()
        request.session['credentials'] = user.oauth_credentials
        logger.info(f"user: {user}; channel: {channel} authorized")
        return JsonResponse(
            {
                'username': user.username,
                'channel': channel.name,
                'isAuthorized': True,
                'redirectUrl': f'{FRONTEND_URL}/overview'
            }, safe=False
        )
    
    if IS_LOCAL and whether_test is False:
        # if it is local, then we only support logging in without oauth
        logger.warning("We cannot authorize the user in a local environment.")
        return JsonResponse({'redirectUrl': '', 'isAuthorized': False})

    return __authorize_oauth()
    
def oauth2_callback(request):
    state = request.session.get('state')
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        'client_secret.json',  # Path to your OAuth client secret JSON file
        scopes=['https://www.googleapis.com/auth/youtube.force-ssl'],
        state=state
    )
    flow.redirect_uri = 'https://api.youtube.filterbuddypro.com/oauth2callback/'

    # Fetch the authorization response
    authorization_response = request.build_absolute_uri()
    logger.info(f"authorization_response: {authorization_response}")
    flow.fetch_token(authorization_response=authorization_response)

    # Store credentials in the session
    credentials = flow.credentials
    request.session['credentials'] = utils.credentials_to_dict(credentials)

    # we still need to store credentials in the backend
    user, channel = YoutubeAPI(credentials).create_account()
    request.session['credentials']['myChannelId'] = channel.id
    return HttpResponseRedirect(
        f'https://youtube.filterbuddypro.com/overview?owner={user.username}&channel={channel.name}&isNewAccount'
    )

@user_verification_required
def logout_user(request):
    if 'credentials' in request.session:
        del request.session['credentials']
    page_message = 'Credentials have been cleared.'
    return JsonResponse({'message': page_message})

@user_verification_required
def synchronize_youtube(request):
    request_data = json.loads(request.body)
    owner = request_data.get('owner')
    forced = request_data.get('forced', False)
    user = User.objects.filter(username=owner).first()
    
    now_time = timezone.now()
    if ((not forced)
        and (user.last_sync is not None)
        and (now_time - user.last_sync < timezone.timedelta(minutes=60))
    ):
        return JsonResponse(
            {
                'message': 'Synchronization has been initiated recently. Please wait for a few minutes.',
                'taskId': None,
            }, safe=False
        )
    else:
        task = tasks.synchronize_youtube_task.delay(user.username)
        return JsonResponse(
            {
                'message': 'Synchronization has been initiated.',
                'taskId': task.id
            }, safe=False
        )
    
@user_verification_required
def request_user(request):
    request_data = json.loads(request.body)
    owner = request_data.get('owner')
    owner = User.objects.filter(username=owner).first()
    if not owner:
        return JsonResponse({'error': 'Owner of the channel parameter is required'}, status=400)
    else:
        return JsonResponse(
            {
                'user': owner.username,
                'permissionMode': 'auto' if owner.moderation_access else 'manual',
                'channel': owner.channel.name,
                'avatar': owner.avatar
            }, safe=False
        )

@user_verification_required
def request_filters(request):
    # Retrieve the 'owner' GET parameter
    request_data = json.loads(request.body)
    owner = request_data.get('owner')
    owner = User.objects.filter(username=owner).first()
    if not owner:
        return JsonResponse({'error': 'Owner of the channel parameter is required'}, status=400)

    channel = Channel.objects.filter(owner=owner).first()
    if not channel:
        return JsonResponse({'error': 'Channel not found'}, status=404)
    
    filters = PromptFilter.objects.filter(channel=channel)
    numbers_of_new_comments = [utils.number_of_new_comments(filter) for filter in filters]
    filters_data = [ filter.serialize() for filter in filters]
    for index, filter in enumerate(filters_data):
        filter['numberOfNewCaughtComments'] = numbers_of_new_comments[index]
    return JsonResponse(
            {'filters': filters_data}, 
            safe=False
        )

@user_verification_required
def request_comments(request):
    request_data = json.loads(request.body)
    filter_id = request_data.get('filter')
    whether_iterate = request_data.get('iterate', False)
    
    try:
        filter_id = int(filter_id)
    except:
        return JsonResponse({'error': 'The id of the filter is required.'}, status=400)
    
    filter = PromptFilter.objects.filter(id=filter_id).first()
    comments = utils.retrieve_predictions(filter, whether_iterate)
    return JsonResponse(
            {'comments': comments}, 
            safe=False
        )

@user_verification_required
def request_comment_info(request):
    """
        Retrieve the replies and the video information of a comment
        If the comment itself is a reply, we will retrieve the information of the parent comment
    
    """
    request_data = json.loads(request.body)
    comment_id = request_data.get('comment')
    
    filter_id = request_data.get('filter')
    # the requested comment must have the prediction
    prediction = FilterPrediction.objects.filter(filter=filter_id, comment=comment_id).first()
    if not prediction:
        return JsonResponse({'error': 'The id of the comment is required.'}, status=400)
    
    comment_info = None
    if prediction.comment.parent is not None:
        # the parent of a prediction is not necessarily a prediction
        parent_comment = prediction.comment.parent
        # check whether the parent comment has an associated prediction
        parent_prediction = FilterPrediction.objects.filter(filter=filter_id, comment=parent_comment.id).first()
        if parent_prediction is None:
            # the parent comment does not have an associated prediction
            now_comment = parent_comment
            comment_info = now_comment.serialize(as_prediction=True)
        else:
            now_comment = parent_prediction.comment
            comment_info = parent_prediction.serialize()
    else:
        now_comment = prediction.comment
        comment_info = prediction.serialize()

    # because we want more info about a video in addition to the title
    comment_info['video'] = prediction.comment.video.serialize()
    comment_info['replies'] = []
    for reply in now_comment.replies.all():
        # check whether the reply has an associated prediction
        reply_prediction = FilterPrediction.objects.filter(filter=filter_id, comment=reply.id).first()
        if reply_prediction:
            reply_info = reply_prediction.serialize()
        else:
            # this is to make sure the reply has the same structure as the comment so that the frontend can render it
            reply_info = reply.serialize(as_prediction=True)
        comment_info['replies'].append(reply_info)
        
    return JsonResponse(
            {'commentInfo': comment_info}, 
            safe=False
        )

@user_verification_required
def poll_tasks(request):
    from celery.result import AsyncResult
    request_data = json.loads(request.body)
    task_id = request_data.get('task_id')
    task_result = AsyncResult(task_id)
    if task_result.ready():
        try:
            results = task_result.get()
            return JsonResponse({
                "status": 'completed',
                "message": f"Task {task_id} is completed",
                "result": results
            })
        except OperationalError as e:
            logger.error(f"OperationalError: {e}")
            return JsonResponse({
                "status": 'pending',
                "message": f"Task {task_id} is still pending but an error occurred: {e}",
            }) 
    else:
        return JsonResponse({
            "status": 'pending',
            "message": f"Task {task_id} is still pending"
        })

@user_verification_required
def initialize_prompt(request):
    request_data = json.loads(request.body)
    name = request_data.get('name')
    description = request_data.get('description')
    example = request_data.get('example')

    proposed_prompt = buddy.initialize_prompt(
        name, description, example
    )
    
    return JsonResponse(
        {
            'prompt': proposed_prompt
        }, safe=False
    )

@user_verification_required
def explore_prompt(request):
    request_data = json.loads(request.body)
    filter = request_data.get('filter')
    needed_num = request_data.get('needed_num', 10)
    logger.info('filter: {filter}')
    filter = PromptFilter.objects.filter(id=filter['id']).first()
    if filter is None:
        return JsonResponse(
            {
                'message': f"The filter does not exist.",
                'taskId': None
            }, safe=False
        )

    task = tasks.select_interesting_comments_task.delay(filter.id, needed_num)
    task_id = task.id
    return JsonResponse(
        {   
            'taskId': task_id
        }, safe=False
    )

@user_verification_required
def save_groundtruth(request):
    """
        Save the groundtruth of a list of comments.
    """
    request_data = json.loads(request.body)
    filter = request_data.get('filter')
    comments = request_data.get('comments')

    for comment in comments:
        prediction = FilterPrediction.objects.filter(filter=filter['id'], comment=comment['id']).first()
        if prediction is not None:
            prediction.groundtruth = comment['groundtruth']
            prediction.save()
    return JsonResponse(
        {
            'message': f"The groundtruths of the filter {filter['name']} for {len(comments)} has been successfully updated."
        }, safe=False
    )
    
@user_verification_required
def improve_prompt(request):
    request_data = json.loads(request.body)
    filter = request_data.get('filter')

    clusters = None
    task_id = None

    filter = PromptFilter.objects.filter(id=filter['id']).first()
    if filter is None:
        return JsonResponse(
            {
                'message': f"The filter {filter['name']} does not exist.",
                'clusters': None,
                'taskId': None

            }, safe=False
        )

    # retrieve clusters for this filter and serialize them
    cluster_instances = MistakeCluster.objects.filter(filter=filter)
    if cluster_instances.count() > 0:
        clusters = [cluster.serialize() for cluster in cluster_instances]
        # ensure that the first cluster has an non-empty field
        if not clusters[0]['summary']:
            backend_filter = BackendPromptFilter.create_backend_filter(filter)
            summary = buddy.interpret_refine_infos(backend_filter, clusters[0])
            clusters[0]['summary'] = summary
            cluster_instances[0].summary = summary
            cluster_instances[0].save()
    else:
        minimum_mistake_count = 3
        mistake_instances = FilterPrediction.objects.filter(
                groundtruth__isnull=False, filter=filter
            ).exclude(
                groundtruth=F('prediction')
            )

        if mistake_instances.count() < minimum_mistake_count:
            return JsonResponse(
                {   
                    'message': f"There are less than {minimum_mistake_count} mistakes to improve for the filter {filter.name}.",
                    'clusters': [],
                    'taskId': None
                }, safe=False
            )
        
        task = tasks.generate_clusters_task.delay(filter.id)
        task_id = task.id
        
    
    return JsonResponse(
        {   
            'message': f"We have started to generate clusters for the filter {filter.name}.",
            'taskId': task_id,
            'clusters': clusters
        }, safe=False
    )

@user_verification_required
def summarize_cluster(request):
    request_data = json.loads(request.body)
    cluster = request_data.get('cluster')
    filter = request_data.get('filter')

    old_suggestion_id = request_data.get('old_suggestion')
    if old_suggestion_id:
        # TODO: do we want to simply mark it as inactive or delete it?
        # the only benefit of marking it as inactive is that we can make sure that future suggestions are not repeated
        MistakeCluster.objects.filter(id=old_suggestion_id).delete()

    filter = PromptFilter.objects.filter(id=filter['id']).first()
    backend_filter = BackendPromptFilter.create_backend_filter(filter)
    summary = buddy.interpret_refine_infos(backend_filter, cluster)
    
    return JsonResponse(
        {
            'summary': summary
        }, safe=False
    )

@user_verification_required
def clarify_prompt(request):
    request_data = json.loads(request.body)
    filter = request_data.get('filter')

    filter = PromptFilter.objects.filter(id=filter['id']).first()
    comment = request_data.get('comment')

    prediction = FilterPrediction.objects.filter(filter=filter, comment=comment['id']).first()
    if prediction is not None:
        backend_filter = BackendPromptFilter.create_backend_filter(filter)
        reflection = buddy.reflect_on_mistake(backend_filter, prediction.serialize())
        # update the reflection field of this prediction
        prediction.reflection = reflection
        prediction.save()
        return JsonResponse(
            {
                'followup': reflection
            }, safe=False
        )
    else:
        return JsonResponse(
            {
                'message': f"The prediction for the comment {comment['id']} does not exist."
            }, safe=False
        )

@user_verification_required
def refine_prompt(request):
    request_data = json.loads(request.body)
    cluster = request_data.get('cluster')
    

    # confirm that the prediction exists
    if 'cluster' not in cluster or len(cluster['cluster']) == 0:
        return JsonResponse(
            {
                'message': f"There are no mistakes to refine."
            }, safe=False
        )
    
    filter = request_data.get('filter')
    task = tasks.refine_prompt_task.delay(filter['id'], cluster)
    task_id = task.id
    return JsonResponse(
        {
            'message': f"We have started to refine the prompt for the filter {filter['name']}.",
            'taskId': task_id
        }, safe=False
    )
    
@user_verification_required
def save_prompt(request):

    request_data = json.loads(request.body)
    new_filter = request_data.get('filter')
    mode = request_data.get('mode')
    start_date = request_data.get('start_date')

    if 'id' in new_filter:
        filter = PromptFilter.objects.filter(id=new_filter['id']).first()
    else:
        # create a new filter
        channel = Channel.objects.filter(owner__username=new_filter['owner']).first()
        filter = PromptFilter(name=new_filter['name'], description=new_filter['description'], channel=channel)
        filter.save()
    logger.info(f"Saving a filter: {filter.serialize(view=True)}")

    if filter.last_run is None or filter.whether_changed(new_filter) or mode == 'all':
        # if the filter has not been run before or the description has been updated
        logger.info(f"Detected changes in the filter {filter.name}")

        filter.update_filter(new_filter)
        filter.save()
        
        task = tasks.update_predictions_task.delay(filter.id, mode, start_date)
        task_id = task.id
        return JsonResponse(
            {
                'message': f"The description and the predictions of the filter {filter.name} has been successfully updated.",
                'taskId': task_id,
                'filter': filter.serialize()
            }, safe=False
        )
    elif 'action' in new_filter and filter.action != new_filter['action']:
        logger.info(f"Updating the action of the filter {filter.name}")
        filter.action = new_filter['action']
        filter.save()

        # We do not revert the old actions but simply impose the new action
        updates.update_actions(filter, start_date)
        return JsonResponse(
            {
                'message': f"The action of the filter {filter.name} has been successfully updated.",
                'filter': filter.serialize(),
                'taskId': None
            }, safe=False
        )
    else:
        logger.info(f"Detected no changes in the filter {filter.name}")
        return JsonResponse(
            {
                'message': f"The filter {filter.name} has not been updated as no changes were detected.",
                'taskId': None,
                'filter': filter.serialize()
            }, safe=False
        )

@user_verification_required
def delete_prompt(request):
    request_data = json.loads(request.body)
    filter = request_data.get('filter')
    filter = PromptFilter.objects.filter(id=filter['id']).first()
    filter.delete_prompt()
    return JsonResponse(
        {
            'message': f"The filter {filter.name} has been successfully deleted."
        }, safe=False
    )

@user_verification_required
def explain_prediction(request):
    request_data = json.loads(request.body)
    
    filter = request_data.get('filter')
    filter = PromptFilter.objects.filter(id=filter['id']).first()
    comment = request_data.get('comment')
    
    # retrieve explanation for the classification decision if any
    prediction = FilterPrediction.objects.filter(filter_id=filter, comment_id=comment['id']).first()
    if prediction and prediction.explanation:
        explanation = prediction.explanation
    else:
        backend_filter = BackendPromptFilter.create_backend_filter(filter)
        explanation = buddy.explain_prediction(backend_filter, prediction.serialize())
        if prediction:
            prediction.explanation = explanation
            prediction.save()
    

    return JsonResponse(
        {
            'explanation': explanation
        }, safe=False
    )

@user_verification_required
def revert_prediction(request):
    request_data = json.loads(request.body)
    
    filter = request_data.get('filter')
    comment = request_data.get('comment')
    is_mistake = request_data.get('is_mistake')

    # retrieve explanation for the classification decision if any
    prediction = FilterPrediction.objects.filter(filter_id=filter['id'], comment_id=comment['id']).first()
    logger.info(f'prediction: {comment}')
    logger.info(f'filter id: {filter["id"]}')
    if prediction.prediction is not None:
        # reverting the prediction only makes sense if there is a not None prediction
        prediction.groundtruth = not prediction.prediction if is_mistake else prediction.prediction
        message = f"We have marked this comment's groundtruth as {prediction.groundtruth}."
        prediction.save()
    else:
        message = "This comment has not been classified yet."
    return JsonResponse(
        {
            'message': message,
            'groundtruth': prediction.groundtruth
        }, safe=False
    )

@user_verification_required
def refresh_predictions(request):

    request_data = json.loads(request.body)
    filter = request_data.get('filter')
    comments = request_data.get('comments')

    logger.info(f'We want to refresh their predictions of {len(comments)} comments.')
    task = tasks.predict_comments_task.delay(filter, comments)

    return JsonResponse(
        {
            'message': f"We start to calculate the predictions of the filter {filter['name']}.",
            'taskId': task.id
        }, safe=False
    )
