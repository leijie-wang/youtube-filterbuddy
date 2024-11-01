from doctest import Example
import json
import logging
from math import exp
from numpy import negative
import random
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseRedirect
from django.middleware.csrf import get_token
from django.db.models import Q, F
from django.shortcuts import redirect
import google.oauth2.credentials
import google_auth_oauthlib.flow

from . import utils
from .models import Channel, PromptFilter, FilterPrediction, Comment
from .llm_filter import LLMFilter
from .llm_buddy import LLMBuddy
from .tasks import update_predictions_task
from .youtube import YoutubeAPI

logger  = logging.getLogger(__name__)
buddy = LLMBuddy()

@csrf_exempt
def csrf_token_view(request):
    # Generate and return the CSRF token
    token = get_token(request)
    return JsonResponse({'csrfToken': token})


def verify_user(request):
    if 'credentials' in request.session and 'channel_id' in request.session['credentials']:
        channel_id = request.session['credentials']['channel_id']
        channel = Channel.objects.filter(id=channel_id).first()
        if channel:
            return True
    return False

def authorize_user(request):
    if verify_user(request):
        channel_id = request.session['credentials']['channel_id']
        channel = Channel.objects.filter(id=channel_id).first()
        user = channel.owner
        return JsonResponse({
            'user': user.username,
            'channel': channel.name
        }, safe=False)

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
    return JsonResponse({'authorizeUrl': authorization_url})


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
    request.session['credentials']['channel_id'] = channel.id
    return HttpResponseRedirect(
        f'https://youtube.filterbuddypro.com/?user={user.username}&channel={channel.name}'
    )


def request_filters(request):
    # Retrieve the 'owner' GET parameter
    request_data = json.loads(request.body)
    owner = request_data.get('owner')
    
    if not owner:
        return JsonResponse({'error': 'Owner of the channel parameter is required'}, status=400)
    
    channel = Channel.objects.filter(owner__username=owner).first()
    if not channel:
        return JsonResponse({'error': 'Channel not found'}, status=404)
    
    filters = PromptFilter.objects.filter(channel=channel)
    
    filters_data = [ filter.serialize() for filter in filters]
    
    return JsonResponse(
            {'filters': filters_data}, 
            safe=False
        )


def request_predictions(request):
    filter_id = request.GET.get('filter')
    
    if filter_id is None or not filter_id.isnumeric():
        return JsonResponse({'error': 'The id of the filter is required.'}, status=400)
    
    predictions = FilterPrediction.objects.filter(filter=filter_id)
    logger.info(f"predictions: {len(predictions)}")
    comments = [prediction.serialize() for prediction in predictions]
    
    return JsonResponse(
            {'comments': comments}, 
            safe=False
        )


def initialize_prompt(request):
    request_data = json.loads(request.body)
    name = request_data.get('name')
    description = request_data.get('description')
    example = request_data.get('example')

    chatbot = utils.ChatCompletion()
    system_prompt = """
        A user is writing down their content moderation preferences in prompts but has difficulties clearly communicating their preferences. 
        However, they could intuitively tell the groundtruth of a text (1 represents that the text should be removed, and 0 represents that the text should be kept).
        Your task is to write a prompt for the user based on their labeled examples.

        ### TASK DESCRIPTION
        STEP 1: Reason about overarching theme of the examples.
        STEP 2: Write a prompt that would label the examples correctly.

        ### RULES FOR PROMPT WRITING
        RULE 1: Begin the prompt with "Remove texts that"


        ### EXAMPLE 1
        #### INPUT
        <Text>I want to commit suicide</Text>
        <Groundtruth>1</Groundtruth>
        <Text>Sometimes I just to hurt him to the point that he won't recover</Text>
        <Groundtruth>1</Groundtruth>
        <Text>Your jokes are killing me</Text>
        <Groundtruth>0</Groundtruth>
        <Text>That semester just killed my motivation to try</Text>
        <Groundtruth>0</Groundtruth>

        #### OUTPUT
        <Reasoning>
            The examples indicate that texts mentioning killing oneself or others should be removed.
            However, discussions of killing in a purely metaphorical sense should not be removed.
        </Reasoning>
        <Prompt>
            Remove texts that talk about a person killing another person or a person killing themselves, but not if discussed in a metaphorical sense.
        </Prompt>

        ### EXAMPLE 2
        #### INPUT
        <Text>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Text>
        <Groundtruth>0</Groundtruth>
        <Text>Republicans are full of bullshit. Never trust them</Text>
        <Groundtruth>0</Groundtruth>
        <Text>She is autistic I mean look at her</Text>
        <Groundtruth>1</Groundtruth>

        #### OUTPUT
        <Reasoning>
           The examples indicate that texts insulting individuals by using derogatory terms should be removed.
           However, insults of groups should not be removed.
        </Reasoning>
        <Prompt>
            Remove comments that use derogatory terms to insult individuals. However, do NOT remove:
                - texts that use derogatory terms to insult groups
        </Prompt>

        ### EXAMPLE 3
        #### INPUT
        <Text>most people are too stupid to own a gun</Text>
        <Groundtruth>1</Groundtruth>
        <Text>She is uneducated because she didn't have the right resources offered to her growing up. That's a failing of our country</Text>
        <Groundtruth>0</Groundtruth>
        <Text>He's uneducated. Plain and simple</Text>
        <Groundtruth>1</Groundtruth>
        <Text>This girl is a joke</Text>
        <Groundtruth>0</Groundtruth>

        #### OUTPUT
        <Reasoning>
            The examples indicate that texts demeaning an individual's or group's intelligence should be removed.
            However, mere explanations of low intelligence that are not demeaning should not be removed.
        </Reasoning>
        <Prompt>
            Remove texts that demean a persons intelligence or multiple people's intelligence. However, do NOT remove:
                - texts that explain an individual or group's situation regarding intelligence, rather than demean their intelligence
        </Prompt>

        ### EXAMPLE 4
        #### INPUT
        <Text>These conservatives are always trying to shoot up schools</Text>
        <Groundtruth>1</Groundtruth>
        <Text>Republican control states are more dangerous than Democrats control states.</Text>
        <Groundtruth>0</Groundtruth>
        <Text>The liberal agenda is one of censorship, crime, and hate</Text>
        <Groundtruth>1</Groundtruth>

        #### OUTPUT
        <Reasoning>
           The examples indicate that texts negatively stereotyping political parties and their related names like "conservatives" and "liberals" should be removed.
           However, texts that mention political parties only in relation to the states that they control should not be removed.
        </Reasoning>
        <Prompt>
            Remove texts that negatively stereotype political parties (and their related names, e.g., "conservatives" and "liberals"). However, do NOT remove:
                - texts that mention political parties only in relation to the states that they control
        </Prompt>
    """
    user_prompt = f"""
        <Text>{example}</Text>
        <Groundtruth>1</Groundtruth>
    """
    response = chatbot.chat_completion(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        type="text"
    )
    proposed_prompt = chatbot.extract_xml(response, "Prompt")
    print(proposed_prompt)
    return JsonResponse(
        {
            'prompt': proposed_prompt
        }, safe=False
    )


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

    logger.info(f"filter: {filter}")
    if filter.last_run is None or filter.description != new_filter['description']:
        filter.description = new_filter['description']
        if 'action' in new_filter:
            filter.action = new_filter.get('action')
        filter.save()
        task = update_predictions_task.delay(filter.id, mode)
        task_id = task.id
        return JsonResponse(
            {
                'message': f"The description and the predictions of the filter {filter.name} has been successfully updated.",
                'taskID': task_id,
                'filter': filter.serialize()
            }, safe=False
        )
    elif 'action' in new_filter and filter.action != new_filter['action']:
        filter.action = new_filter['action']
        filter.save()
        # TODO: update the actions on Youtube based on the new action
        return JsonResponse(
            {
                'message': f"The action of the filter {filter.name} has been successfully updated.",
                'filter': filter.serialize()
            }, safe=False
        )
    else:
        return JsonResponse(
            {
                'message': f"The filter {filter.name} has not been updated as no changes were detected."
            }, safe=False
        )
    
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


def revert_prediction(request):
    request_data = json.loads(request.body)
    
    filter = request_data.get('filter')
    comment = request_data.get('comment')
    reverted = request_data.get('reverted')
    # retrieve explanation for the classification decision if any
    prediction = FilterPrediction.objects.filter(filter=filter['id'], comment=comment['id']).first()
    if prediction and prediction.prediction is not None:
        # reverting the prediction only makes sense if there is a not None prediction
        prediction.groundtruth = not prediction.prediction if reverted else prediction.prediction
        message = f"We have marked this comment as {prediction.groundtruth}."
        prediction.save()
    else:
        message = "This comment has not been classified yet."
    return JsonResponse(
        {
            'message': message
        }, safe=False
    )
