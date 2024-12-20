
import json
import logging
from math import log
import os
import random
import time
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseRedirect
from django.middleware.csrf import get_token
from django.db.models import Q, F, Case, When, Value, IntegerField
from functools import wraps
import google_auth_oauthlib.flow


from . import utils
from .models import Channel, PromptFilter, FilterPrediction, Comment, User
from .llm_filter import LLMFilter
from .llm_buddy import LLMBuddy
from . import tasks
from .youtube import YoutubeAPI

logger  = logging.getLogger(__name__)
buddy = LLMBuddy()
FRONTEND_URL = os.getenv("FRONTEND_URL", "localhost:3001")


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

def authorize_user(request):
    # for the test user
    request_data = json.loads(request.body)
    if request_data.get('whether_test', False):
        logger.info("Creating a test user")
        user = User.objects.filter(username='TheYoungTurks').first()
        if not user:
            # create a test user
            user = utils.populate_test_users()
            utils.populate_filters(user.channel)
        channel = Channel.objects.filter(owner=user).first()
        request.session['credentials'] = user.oauth_credentials
        return JsonResponse(
            {
                'user': user.username,
                'channel': channel.name,
                'redirectUrl': f'{FRONTEND_URL}/overview?owner={user.username}&channel={channel.name}'
            }, safe=False
        )
    
    if verify_user(request):
        logger.info("User has already been authorized.")
        channel_id = request.session['credentials']['myChannelId']
        channel = Channel.objects.filter(id=channel_id).first()
        user = channel.owner
        return JsonResponse(
            {
                'user': user.username,
                'channel': channel.name,
                'redirectUrl': f'{FRONTEND_URL}/overview?owner={user.username}&channel={channel.name}'
            }, safe=False
        )
    
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
    return JsonResponse({'redirectUrl': authorization_url})

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
        f'https://youtube.filterbuddypro.com/overview?owner={user.username}&channel={channel.name}'
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
    
    filters_data = [ filter.serialize() for filter in filters]

    return JsonResponse(
            {'filters': filters_data}, 
            safe=False
        )

@user_verification_required
def request_comments(request):
    request_data = json.loads(request.body)
    filter_id = request_data.get('filter')
    
    try:
        filter_id = int(filter_id)
    except:
        return JsonResponse({'error': 'The id of the filter is required.'}, status=400)
    
    filter = PromptFilter.objects.filter(id=filter_id).first()
    comments = utils.retrieve_predictions(filter)
    return JsonResponse(
            {'comments': comments}, 
            safe=False
        )

@user_verification_required
def request_comment_info(request):
    request_data = json.loads(request.body)
    comment_id = request_data.get('comment')
    
    filter_id = request_data.get('filter')
    prediction = FilterPrediction.objects.filter(filter=filter_id, comment=comment_id).first()
    if not prediction:
        return JsonResponse({'error': 'The id of the comment is required.'}, status=400)
    
    if prediction.comment.parent is not None:
        prediction = FilterPrediction.objects.filter(filter=filter_id, comment=prediction.comment.parent.id).first()
    
    comment_info = prediction.serialize()
    comment_info['video'] = prediction.comment.video.serialize()
    comment_info['replies'] = []
    for reply in prediction.comment.replies.all():
        # check whether the reply has an associated prediction
        reply_prediction = FilterPrediction.objects.filter(filter=filter_id, comment=reply.id).first()
        if reply_prediction:
            reply_info = reply_prediction.serialize()
        else:
            # this is to make sure the reply has the same structure as the comment so that the frontend can render it
            reply_info = reply.serialize()
            reply_info['prediction'] = None
            reply_info['explanation'] = ''
            reply_info['groundtruth'] = None
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
        results = task_result.get()
        return JsonResponse({
            "status": 'completed',
            "message": f"Task {task_id} is completed",
            "result": results
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

    proposed_prompt = buddy.initialize_prompt(example)
    
    return JsonResponse(
        {
            'prompt': proposed_prompt
        }, safe=False
    )

@user_verification_required
def explore_prompt(request):
    request_data = json.loads(request.body)
    owner = request_data.get('owner')
    print(f"owner: {owner}")
    
    if not owner:
        return JsonResponse({'error': 'Owner of the channel parameter is required'}, status=400)
    
    channel = Channel.objects.filter(owner__username=owner).first()
    if not channel:
        return JsonResponse({'error': 'Channel not found'}, status=404)

    

    name = request_data.get('name')
    description = request_data.get('description')
    prompt = {'name': name, 'description': description}

    # randomly sample comments from the database to begin with
    comments = Comment.objects.filter(video__channel=channel).order_by('posted_at')
    if not comments.exists():
        return JsonResponse({'error': 'No comments found for this channel'}, status=404)

    comments = random.sample(list(comments), 50)
    datasets = [comment.content for comment in comments]
    
    llm_filter = LLMFilter(prompt, debug=False)
    predictions = llm_filter.predict(datasets)

    # summarize the predictions
    positive_num = sum(predictions)
    negative_num = len(predictions) - positive_num
    print(f"There are {positive_num} positive predictions and {negative_num} negative predictions.")

    comments = [comment.serialize() for comment in comments]
    for index, comment in enumerate(comments):
        comment["prediction"] = predictions[index]
        # update the prediction in the database
        FilterPrediction.objects.create(
            filter=prompt['id'],
            comment=comment['id'],
            prediction=comment['prediction']
        ).save()
    comments = sorted(comments, key=lambda x: x["prediction"], reverse=True)
    # sample a balanced set of positive and negative comemnts
    return JsonResponse(
        {
            'comments': comments
        }, safe=False
    )

@user_verification_required
def improve_prompt(request):
    request_data = json.loads(request.body)
    filter = request_data.get('filter')

    filter = PromptFilter.objects.filter(id=filter['id']).first()
    if filter is None:
        return JsonResponse(
            {
                'message': f"The filter {filter['name']} does not exist."
            }, safe=False
        )

    mistakes = FilterPrediction.objects.filter(
            groundtruth__isnull=False, filter=filter
        ).exclude(
            groundtruth=F('prediction')
        )

    if mistakes.count() == 0:
        return JsonResponse(
            {
                'summary': '',
                'cluster': []
            }, safe=False
        )
    summary, cluster = buddy.improve_suggestion(filter, mistakes)
    return JsonResponse(
        {
            'summary': summary,
            'cluster': [prediction.serialize() for prediction in cluster]
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
        followup = buddy.clarify_prompt(filter, prediction)
        return JsonResponse(
            {
                'followup': followup
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
    filter = request_data.get('filter')

    filter = PromptFilter.objects.filter(id=filter['id']).first()
    comment = request_data.get('comment')
    prediction = FilterPrediction.objects.filter(filter=filter, comment=comment['id']).first()

    followup = request_data.get('followup')
    if prediction is not None:
        refined_prompt = buddy.refine_prompt(filter, prediction, followup)
        return JsonResponse(
            {
                'refinedDescription': refined_prompt
            }, safe=False
        )
    else:
        return JsonResponse(
            {
                'message': f"The prediction for the comment {comment['id']} does not exist."
            }, safe=False
        )

@user_verification_required
def save_prompt(request):

    request_data = json.loads(request.body)
    new_filter = request_data.get('filter')
    mode = request_data.get('mode')
    # either 'all' or 'new': 'all' means updating all comments, 'new' means updating only new comments

    if 'id' in new_filter:
        filter = PromptFilter.objects.filter(id=new_filter['id']).first()
    else:
        channel = Channel.objects.filter(owner__username=new_filter['owner']).first()
        filter = PromptFilter(name=new_filter['name'], description=new_filter['description'], channel=channel)
        filter.save()

    logger.info(f"filter that is saved: {filter}")
    if filter.last_run is None or filter.description != new_filter['description']:
        logger.info(f"Updating the description of the filter {filter.name}")
        filter.description = new_filter['description']
        if 'action' in new_filter:
            filter.action = new_filter.get('action')
        filter.save()
        task = tasks.update_predictions_task.delay(filter.id, mode)
        task_id = task.id
        # it is possible that the user updates both the description and the action.
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
        youtube = YoutubeAPI(filter.channel.owner.oauth_credentials)
        youtube.execute_action(filter)
        comments = utils.retrieve_predictions(filter)
        return JsonResponse(
            {
                'message': f"The action of the filter {filter.name} has been successfully updated.",
                'filter': filter.serialize(),
                'predictions': comments,
                'taskId': None
            }, safe=False
        )
    else:
        return JsonResponse(
            {
                'message': f"The filter {filter.name} has not been updated as no changes were detected.",
                'taskId': None
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
    # TODO: for newly initialized filters, the filter id is not available
    comment = request_data.get('comment')

    # retrieve explanation for the classification decision if any
    prediction = FilterPrediction.objects.filter(filter_id=filter['id'], comment_id=comment['id']).first()
    if prediction and prediction.explanation:
        explanation = prediction.explanation
    else:
        explanation = buddy.explain_prediction(prediction.filter, prediction)
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
    prediction = FilterPrediction.objects.filter(filter=filter['id'], comment=comment['id']).first()
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
    
    affected_comments = FilterPrediction.objects.filter(filter=filter['id']).annotate(
        # Assign a ranking for groundtruth (more important)
        groundtruth_rank=Case(
            When(groundtruth=True, then=Value(1)),   # True (positive) ranked highest
            When(groundtruth=False, then=Value(1)),  # False (negative) has the same rank as True
            When(groundtruth=None, then=Value(2)),   # None ranked lowest
            output_field=IntegerField(),
        ),
        # Assign a ranking for prediction (less important)
        prediction_rank=Case(
            When(prediction=True, then=Value(1)),    # True (positive) ranked highest
            When(prediction=False, then=Value(2)),   # False (negative) ranked second
            When(prediction=None, then=Value(3)),    # None ranked lowest
            output_field=IntegerField(),
        )
    ).order_by('groundtruth_rank', 'prediction_rank', '-comment__posted_at')

    prioritized_number = FilterPrediction.objects.filter(Q(prediction=True) | Q(groundtruth__isnull=False)).count()
    if prioritized_number > 200:
        comments = affected_comments[:prioritized_number]
    else:
        comments = affected_comments[:200]

    logger.info(f'There are {len(comments)} comments to be updated.')
    comments = [comment.serialize() for comment in comments]

    task = tasks.predict_comments_task.delay(filter, comments)

    return JsonResponse(
        {
            'message': f"The predictions of the filter {filter['name']} has been successfully updated.",
            'taskId': task.id
        }, safe=False
    )
