from calendar import c
import copy
import comm
import ipywidgets as widgets
from IPython.display import display
import threading
import logging
import math
import numpy as np
import random
import time


from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from .chat_completion import ChatCompletion
from .models import FilterPrediction, Comment
from . import updates
from . import utils

logger = logging.getLogger(__name__)

class LLMBuddy:

    def __init__(self):
        self.llm_client = ChatCompletion()
        self.failure_scenarios = [
            'This comment should have been caught by some positive rubric (including the case where there lacks such a positive rubric) but was not.',
            'This comment should not have been caught by a negative rubric but was.',
            'This comment should not have been caught by a positive rubric but was.',
            'This comment should have been caught by some negative rubric (including the case where there lacks such a negative rubric) but was not.',
        ]
        self.debug = True

    def initialize_prompt(self, name, description, example):
        system_prompt = """
            <tasks>
                A content creator is writing down their content moderation/curation preferences in a prompt but has difficulties clearly communicating their preferences.
                To begin with, content creators might provide two kinds of inputs to help you understand their preferences:
                    - they could intuitively tell the groundtruth of a text (1 represents that the text should be caught by the prompt, and 0 represents that the text should not be caught).
                    - the content creator might also provide a draft prompt.
                If the content creator has already provided a draft prompt, then you should simply refine this draft based on the examples.
                Otherwise, you should write a prompt for them based on their labeled examples.

                Note that content creators might want to catch negative comments, or they might want to catch positive comments.
            </tasks>

            <steps>
                <step1>
                    With the given name, labeled examples, and the optional draft prompt,
                    reason what content moderation preferences the content creator wants to communicate.
                </step1>
                <step2>
                    Based on your reasoning in step 1, write down the prompt for the content creator.
                    Your prompt should begin with 'Comments that ...', a sentence that simply describes a category of comments but should not indicate the action (e.g., 'be removed', 'be caught', etc.)
                    Your prompt should be clear, self-explanatory, and concise.
                </step2>
            </steps>

            Write your output in the following xml format.
            <Reasoning>Your reasoning at the step 1</Reasoning>
            <Prompt>Your prompt at the step 2</Prompt>

            <examples>
                <example>
                    <Name>Personal insults</Name>
                    <Text>most people are too stupid to own a gun</Text>
                    <Groundtruth>1</Groundtruth>
                    <Draft>remove comments that attack people as stupid.</Draft>

                    <Reasoning>
                        Based on the example provided, the content creator wants to remove comments that directly attack others by calling them "stupid" or otherwise insulting their intelligence. 
                        The original draft prompt is already clear but can be slightly refined to cover related terms and ensure clarity.
                    </Reasoning>
                    <Prompt>
                        Comments that insult or belittle others by attacking their intelligence.
                    </Prompt>
                    
                </example>
                <example>
                    <Name>Constructive Feedback</Name>
                    <Text></Text>
                    <Groundtruth></Groundtruth>
                    <Draft>Find comments that offer helpful suggestions for improvement while still being supportive.</Draft>

                    <Reasoning>
                        Without specific examples provided, I need to work directly from the draft prompt.
                        The content creator wants to identify comments that provide constructive feedback in a supportive manner.
                        This suggests they value comments that balance critique with encouragement - not just negative criticism, but helpful input delivered with positive intent.
                        The draft prompt captures this intent but could be refined to be more specific about the balanced nature of these comments and to follow the required format.
                    </Reasoning>
                    <Prompt>
                        Comments that balance specific suggestions for content improvement with genuine appreciation or encouragement.
                    </Prompt>
                </example>

                <example>
                    <Name>Political Stereotypes</Name>
                    <Text>The liberal agenda is one of censorship, crime, and hate</Text>
                    <Groundtruth>1</Groundtruth>
                    <Draft>Catch comments that include political stereotypes</Draft>

                    <Reasoning>
                        Based on this example, the content creator wants to target comments that stereotype or attack an entire political group,
                        suggesting negative traits or wrongdoing purely on the basis of political affiliation. 
                        The draft prompt already indicates catching comments that include political stereotypes, but we can refine it for clarity and consistency.
                    </Reasoning>
                    <Prompt>
                        Comments that stereotype or disparage an entire political group by attributing negative traits solely based on their affiliation.
                    </Prompt>
                </example>
                <example>
                    <Name>Violent Threats</Name>
                    <Text>
                        All you traitors had better watch your backs. I’m going to find every last one of you, and you won't be able to run from the consequences. 
                        It’s only a matter of time before you get what’s coming to you.
                    </Text>
                    <Groundtruth>1</Groundtruth>
                    <Draft></Draft>
                    
                    <Reasoning>
                        In this example, the comment explicitly threatens harm or violence against others by warning them to “watch their backs” and promising consequences. 
                        With no draft prompt provided, the content creator’s preference can be inferred as wanting to address direct threats of violence or harm.
                    </Reasoning>
                    <Prompt>
                        Comments that contain direct threats of violence or harm toward others.
                    </Prompt>
                </example>
                
            </examples>
        """
        user_prompt = f"""
            <Name>{name}</Name>
            <Text>{example}</Text>
            <Groundtruth>1</Groundtruth>
            <Draft>{description}</Draft>
        """
        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        proposed_prompt = self.llm_client.extract_xml(response, "Prompt")
        return proposed_prompt

    def select_interesting_comments(self, filter, N=5):
        """
            Given a new filter, we want to select some interesting commments for user annotations.
            Ideally, we want to a balanced dataset with N positive comments and N negative comments.
            But 2N comments that are interesting enough are also acceptable.
        """
        positive_comments = []
        negative_comments = []
        remaining_negative_comments = []

        predictions = FilterPrediction.objects.filter(filter_id=filter.id, groundtruth__isnull=True).all()
        predictions = [pred.serialize() for pred in predictions]
        total_predictions = len(predictions)

        def sample_interesting_comments():
            nonlocal predictions
            for pred in predictions:
                if pred['prediction'] == 1:
                    positive_comments.append(pred)
                elif pred['prediction'] == 0 and pred['confidence'] < 1:
                    negative_comments.append(pred)
                else:
                    remaining_negative_comments.append(pred)
            predictions = []
        
        def produce_new_predictions(batch_size=50):
            nonlocal total_predictions
            # we will first sample comments that do not have predictions yet
            new_comments = Comment.objects.filter(video__channel_id=filter.channel_id).exclude(predictions__filter_id=filter.id).all()
            new_comments = new_comments[:batch_size]
            new_comments = [comment.serialize() for comment in new_comments]
            new_predictions = filter.predict_comments_consistently(new_comments)
            for pred in new_predictions:
                FilterPrediction.objects.update_or_create(
                    filter_id=filter.id,
                    comment_id=pred['id'],
                    defaults={
                        'prediction': pred['prediction'],
                        'confidence': pred['confidence'],
                    }
                )
            total_predictions += len(new_predictions)
            return new_predictions

        
        
        # we first sample comments in case we already have enough interesting comments
        sample_interesting_comments()
        # if there are not enough predictions, we will create new comments
        while ((len(positive_comments) + len(negative_comments)) < 2 * N) and (total_predictions < 20 * N):
            predictions = produce_new_predictions()
            # what if there are less than 200 comments in total?
            if len(predictions) == 0:
                logger.info('No more comments to sample')
                break
            sample_interesting_comments()
            
            
        # based on the number of positive and negative comments, we will sample interesting comments
        interesting_comments = []
        logger.info(f'We have {len(positive_comments)} positive comments and {len(negative_comments)} negative comments to begin with')
        if len(positive_comments) + len(negative_comments) >= 2 * N:
            if len(positive_comments) < N:
                interesting_comments.extend(positive_comments)
                interesting_comments.extend(random.sample(negative_comments, 2 * N - len(positive_comments)))
            elif len(negative_comments) < N:
                interesting_comments.extend(negative_comments)
                interesting_comments.extend(random.sample(positive_comments, 2 * N - len(negative_comments)))
            else:
                interesting_comments = positive_comments[:N] + negative_comments[:N]
        else:
            interesting_comments = positive_comments + negative_comments
            needed_num = 2 * N - len(interesting_comments)
            logger.info(f'We want to sample {needed_num} comments from the remaining negative comments')
            # sample comments from the remaining negative comments, in particular, we sample comments with closer distance to the filter
            sampled_remaining_negative_comments = random.sample(remaining_negative_comments, min(30 * N, len(remaining_negative_comments)))
            comment_embeddings = self.llm_client.list_text_embedding([comment['content'] for comment in sampled_remaining_negative_comments])

            filter_embedding = self.llm_client.text_embedding(filter.stringify_filter())
            distances = cdist(comment_embeddings, [filter_embedding], metric='euclidean').flatten()

            comments_with_distance = list(zip(sampled_remaining_negative_comments, distances))
            # Sort comments by ascending distance (closer first)
            comments_with_distance.sort(key=lambda x: x[1])

            # Select the top 'shortfall' comments with closest distance
            sampled_remaining = [comment for comment, distance in comments_with_distance[:needed_num]]

            # Add sampled comments to interesting_comments
            interesting_comments.extend(sampled_remaining)
        return interesting_comments

    def interpret_comment(self, comment):
        system_prompt = """
            I am not familiar with US politics and not a native English speaker and have difficulties understanding youtube comments.
            Your task is to identify some uncommon words or abbreviations related to politics and English slangs (e.g., AOC, legalize bestiality, inbred swines )in the comment and explain their meanings.
            You should list your explanations for different terms in bullet points, and then generate a concise sentence that summarizes the meaning of this comment.
            Your answer should remain concise and clear to read. Your answer will later be displayed in html so use <br> to separate lines.
        """

        user_prompt = f"""
            <Comment>{comment['content']}</Comment>
        """

        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        return response

    def explain_prediction(self, filter, prediction):
        """
            Explain the prediction of the model based on the given prompt and comment
        """
        system_prompt = """
            A user is writing down their content moderation preference as a prompt.
            This prompt is then used by crowdworkers to classify comments as either 1 (the comment matches the prompt) or 0 (the comment does not match the prompt). 
            However, the user might have trouble understanding why crowdworkers classified a comment in a certain way.
            Your task is to explain their classification decisions based on the given prompt, comment, and the assigned label.

            <Task>
                <step1>
                    Reason about why crowdworkers decided to assign this label to the comment given its content and the prompt.
                    You should try your best to understand the reasoning behind the classification decision.
                    You should not provide your own opinion on whether the comment should be classified as 1 or 0.
                    
                    In your reasoning, if the comment contains explicit phrases that are related to the prompt, you should mention them.
                    Specifically, if the comment is thus classified as 1, you should directly refer to the explicit phrases in the comment that match the prompt.
                    But if the comment is still classified as 0, you should reason what other aspects of the comment nullify the explicit phrases that match the prompt.
                </step1>
                <step2>
                    Summarize your reasoning in one sentence for the user. It should be clear and concise. 
                    In your summary, if the comemnt contains explicit phrases that are related to the prompt, you should mention them.
                </step2>
            <Task>

            <Examples>
                <Example>
                    <Prompt>
                        <Rubric>Comments that talk about a person killing another person</Rubric>
                    </Prompt>
                    <Comment>I want to commit suicide</Comment>
                    <Label>0</Label>

                    <Reasoning>
                        The comment explicitly mentions "suicide," which refers to self-harm. 
                        However, it does not involve the act of one person killing another, which is the specific focus of the prompt.
                    </Reasoning>
                    <Explanation>
                        The comment mentions "suicide", indicating self-harm rather than one person killing another.
                    </Explanation>
                </Example>
                <Example>
                    <Prompt>
                        <Rubric>Comments that promote spam or self-promotion unrelated to the video content</Rubric>
                    </Prompt>
                    <Comment>Subscribe to my channel for free giveaways and exclusive content!</Comment>
                    <Label>1</Label>
                    
                    <Reasoning>
                        The comment explicitly includes the phrase "subscribe to my channel," which directly promotes self-promotion.
                        Additionally, it mentions "free giveaways and exclusive content," further indicating spam or unrelated self-promotion.
                    </Reasoning>
                    <Explanation>
                        The comment explicitly includes "subscribe to my channel" and references "free giveaways," 
                        which are clear indicators of self-promotion unrelated to the video content.
                    </Explanation>
                </Example>
                <Example>
                    <Prompt>
                        <Rubric>Remove comments that directly use toxic language to insult someone</Rubric>
                    </Prompt>
                    <Comment>This video reminds me of how disgusting and pathetic some idiots on this platform can be.</Comment>
                    <Label>0</Label>
                    <Reasoning>
                    The comment explicitly uses toxic phrases such as "disgusting," "pathetic," and "idiots."
                    However, these phrases are used to describe a general group of people rather than directly insulting someone.
                    </Reasoning>
                    <Explanation>
                        Even though the comment uses words like 'disgusting' and 'idiots', they are used to describe a general group of people rather than directly insulting someone.
                    </Explanation>
                </Example>
            </Examples>

            Write your response in the following xml format.
            <Reasoning>Your reasoning at the step 1</Reasoning>
            <Explanation>Your summary of the reasoning at the step 2</Explanation>
        """

        user_prompt = f"""
            <Prompt>
                {filter.stringify_filter()}
            </Prompt>
            <Comment>{prediction['content']}</Comment>
            <Label>{prediction['prediction'] * 1}</Label>
        """

        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        explanation = self.llm_client.extract_xml(response, "Explanation")
        return explanation
    
    def reflect_on_mistake(self, filter, comment):
        why_failed_system_prompt = """
            <Task>
                A content creator is writing down their content moderation preference as a prompt.
                While the concept owner might have challenges in clearly communicating their preferences, they could intuitively tell the ground truth of a comment.
                Given a comment, your task is to reason about possible reasons that the user might want to classify the comment as such.
            </Task>
            
            <steps>
                <step1>
                    Examine the comment against the prompt, and reason why the content creator gave the comment its particular groundtruth label.
                    <Requirements>
                        We have a few requirements for your reasons.
                        1. Each reason should be formatted in this way: 'comments that ...'
                        2. Your reasons should specific, clear, and concise. 
                        3. Each reason should reference the specific phrases of the comment to make your point in quotes.
                        4. If you have multiple reasons, you should ensure that they are distinct and do not overlap.
                            Two reasons are distinct if they are based on different aspects / phrases of the comment.
                            You should not include a more general reason if you have a more specific one that covers the same aspect of the comment.
                            For instance, if one reason is that 'the user does not consider the term "stupid" as a derogatory term.',
                            then you should not include a more general reason like 'the user has a higher tolerance for derogatory terms.'
                        5. You should list your reasons in bullet points.
                    </Requirements>
                </step1>
            </steps>

            <Examples>
                <Example>
                    <Prompt>Comments that use derogatory terms to insult individuals.</Prompt>
                    <Comment>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Comment>
                    <Groundtruth>0</Groundtruth>
                    
                    <Reasoning>
                        This comment is not considered by the content creator as matching the prompt possibly because they do not want to catch
                            <reason>comments that insults a group ('the grand jury') rather than specific individuals.</reason>
                            <reason>comments that only use the term "stupid" as the only derogatory term.</reason>
                    </Reasoning>
                </Example>
                <Example>
                    <Prompt>Comments that talk about a person killing another person.</Prompt>
                    <Comment>I want to commit suicide.</Comment>
                    <Groundtruth>1</Groundtruth>
                    
                    <Reasoning>
                        This comment is considered by the content creator as matching the prompt possibly because they want to catch
                            <reason>comments that mention 'suicide' in addition to killing others.</reason>
                    </Reasoning>
                </Example>
            </Examples>

            Write your output strictly in the following xml format:
            <Reasoning>
                Write down your reasoning in step 1 here.
                You should start with 'This comment is [not] considered by the content creator as matching the prompt because they [do not] want to catch',
                after which, lists potential reasons and wrap each reason in <reason> </reason> tags.
            </Reasoning>
        """
        user_prompt = f"""
            <Prompt>
                {filter.stringify_filter(structured=False)}
            </Prompt>
            <Comment>{comment['content']}</Comment>
            <Groundtruth>{comment['groundtruth']}</Groundtruth>
        """

        response = self.llm_client.chat_completion(
            system_prompt = why_failed_system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        logger.debug(f'Reflect preference response: {response}\n')
        reasons = self.llm_client.extract_xml(response, "Reasoning")
        return reasons
    
    def reflect_on_mistakes_parellel(self, filter, mistakes):
        reflections = [{}] * len(mistakes)  # Placeholder for reflections
        threads = []

        def process_reflection(index, mistake):
            """Compute reflection for a single mistake and store it in the reflections list."""
            try:
                if mistake['reflection']:
                    reflection = mistake['reflection']
                else:
                    reflection = self.reflect_on_mistake(filter, mistake)
                
                embedding = self.llm_client.text_embedding(reflection)
                reflections[index] = {
                    'reflection': reflection,
                    'embedding': embedding
                }
            except Exception as e:
                print(f"Error computing reflection for mistake at index {index}: {e}")

        # Start threads
        for i, mistake in enumerate(mistakes):
            thread = threading.Thread(target=process_reflection, args=(i, mistake))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return reflections

    def cluster_mistakes_unsupervised(self, mistakes, min_samples=2):
        """Cluster mistakes based on their similarity which could help propose a new rubric."""
        if len(mistakes) == 1:
            # special cases: when the user only wants to refine on one mistake.
            return [mistakes]

        def generate_clusters(now_mistakes, eps):
            now_embeddings = [item['embedding'] for item in now_mistakes]
            now_embeddings = np.array(now_embeddings)
            # this ensures that the generated clusters contains at least 2 items
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(now_embeddings)

            clusters = {}
            used_mistakes = []
            for label, item in zip(labels, now_mistakes):
                # DBSCAN uses label = -1 for noise/outliers
                if label == -1:
                    continue
                clusters.setdefault(label, []).append(item)
                used_mistakes.append(item)
            return clusters, now_mistakes

        
        cluster_candidates = []
        remaining_mistakes = [mistake for mistake in mistakes]
        
        eps = 0.5
        # we will stop when we have found at least one cluster
        while eps < 0.8 and not cluster_candidates:
            clusters, remaining_mistakes = generate_clusters(remaining_mistakes, eps)
            new_clusters = list(clusters.values())
            cluster_sizes = [len(cluster) for cluster in new_clusters]
            cluster_sizes_str = ', '.join([str(size) for size in cluster_sizes])
            logger.debug(f'We have {len(new_clusters)} clusters with the size of {cluster_sizes_str} with eps={eps}')
            cluster_candidates.extend(new_clusters)
            eps += 0.1

        # order the cluser candidates by their length and return the longest one
        cluster_candidates = sorted(cluster_candidates, key=lambda x: len(x), reverse=True)
        return cluster_candidates[:1]

    def cluster_mistakes_for_rubrics(self, rubrics, mistakes, threshold=1, min_samples=3):
        """Identify mistakes that might help improve a particular rubric"""
        rubric_embeddings = [self.llm_client.text_embedding(rubric['rubric']) for rubric in rubrics]
        mistake_embeddings = [mistake['embedding'] for mistake in mistakes]

        distance_matrix = cdist(mistake_embeddings, rubric_embeddings, metric='euclidean')
        clustered_mistakes = {rubric['rubric']: [] for rubric in rubrics}

        for i, row in enumerate(distance_matrix):
            # determine the closest rubric for each mistake
            min_idx = np.argmin(row)
            dist = row[min_idx]
            if dist < threshold:
                clustered_mistakes[rubrics[min_idx]['rubric']].append(mistakes[i])
        
        if len(mistakes) >= 2 * min_samples:
            # we only filter out the clusters that are too small when we have enough mistakes
            # otherwise, we will keep all the clusters, for instance, when the user only wants to refine on one mistake.
            clustered_mistakes = { rubric: mistakes for rubric, mistakes in clustered_mistakes.items() if len(mistakes) >= min_samples }
        else:
            # we want to remove empty clusters
            clustered_mistakes = { rubric: mistakes for rubric, mistakes in clustered_mistakes.items() if len(mistakes) >= 1 }
            
        return clustered_mistakes

    def add_new_rubric(self, filter, mistakes, rubric_kind, candidate_num=2, round=2, prefilter=False):
        logger.info('*' * 100)
        logger.info(f'We are adding a new {rubric_kind} rubric with {len(mistakes)} mistakes')
        # for mistake in mistakes:
        #     logger.info(f"\t{mistake['groundtruth']}\t{mistake['content']}\n\t{mistake['reflection']}\n")
        #     logger.info('-' * 50)
        system_prompt = f"""
            <Task>
                A content creator is writing down their content moderation preference as a prompt.
                This prompt is then used by crowdworkers to classify comments as either 1 (the comment matches the prompt) or 0 (the comment does not match the prompt). 
                However, the content creator might not clearly communicate their preferences, which could lead to misclassification by crowdworkers.
                An expert linguist has examined these mistakes, suggested what is missing in the original prompt for each mistake.
                Your task is to add a new {rubric_kind} rubric to incorporate the expert's suggestion.
            </Task>

            <steps>
                <step1>
                    Examine these misclassified comments and the expert's suggestion for each mistake.
                    Reason how you might add a new {rubric_kind} rubric to make sure that crowdworkers can correctly classify such comments in the future.
                </step1>
                <step2>
                    Write down the new {rubric_kind} rubric to incorporate the expert's suggestion.
                    As crowdworkers' performances are very sensitive to the rubric's wording, you should generate {candidate_num} candidate rubrics and we will later select the best one in a pilot study.

                    Here are a few more requirements for your new rubric:
                    - Each of your rubric should start with "Comments that ..." c
                    - To help crowdworkers understand the new rubric, you can provide at least two representative examples of SPECIFIC PHRASES/WORDS.
                    - Keep the language of the rubric concise and specific; avoid being verbose, ambiguous, or overly general.


                    Here are a few examples of good rubrics for your reference:
                    - Comments that use derogatory terms to insult individuals, such as "stupid" or "idiot."
                    - Comments that use severe personal insults, such as calling someone a “douchebag,” “scumbag,” or “bastard,” while excluding milder terms like “fool” or “idiot.”
                    - Comments that publish private data such as “home address,” “phone number,” or “email,”
                </step2>
            </steps>

            Write your response in the following xml format.
            <Reasoning>Your reasoning of how to add a new {rubric_kind} rubric at step 1.</Reasoning>
            <NewRubrics>
                <Rubric>Write your new rubric here.</Rubric>
                <Rubric>Write your new rubric here.</Rubric>
            </NewRubrics>
        """
        
        new_filters = [[] for _ in range(round)]
        def run(now_mistakes, round_index):
            nonlocal new_filters
            problem_comments_str = ""
            for comment in now_mistakes:
                problem_comments_str += f"""
                    <Comment>{comment['reflection']}</Comment>
                """
            user_prompt = f"""
                <Prompt>{filter.stringify_filter(structured=True)}</Prompt>      
                <ProblemComments>
                    {problem_comments_str}
                </ProblemComments>
            """
            response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            ) 
            logger.debug(f'Add new rubric response\n: {response}\n\n')
            new_rubrics = self.llm_client.extract_xml(response, "Rubric")

            for new_rubric in new_rubrics:
                new_filter = copy.deepcopy(filter)
                logger.info(f'[Prompt Candidate] Add a new {rubric_kind} rubric: {new_rubric}')
                new_filter.update_rubric(new_rubric, rubric_kind, comments=now_mistakes)
                new_filter.attributes['action'] = 'Add a new rubric'
                new_filters[round_index].append(new_filter)
        
        
        if len(mistakes) <= 20:
            # if there are not enough mistakes, we will simply run the task in a single thread
            run(mistakes, 0)
        else:
            threads = []
            for i in range(round):
                # Sample a batch of mistakes
                batch_mistakes = random.sample(mistakes, min(20, len(mistakes)))
                
                # Create and start a thread for this batch
                thread = threading.Thread(target=run, args=(batch_mistakes, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        
        new_filters = [filter for sublist in new_filters for filter in sublist]
        logger.info('*' * 100)
        if prefilter:
            # evaluate the performances of the new filters on the mistakes dataset and select the top 3.
            batch_size = min(20, len(mistakes)//3)
            epochs = int(len(mistakes) // batch_size * len(new_filters) * 0.4)
            best_filters = updates.select_best_filters(new_filters, mistakes, topN=candidate_num, epochs=epochs, batch_size=batch_size, exploration=5)
            return best_filters
        else:
            return new_filters

    def edit_rubric(self, filter, rubric, mistakes, rubric_kind, candidate_num=2, round=2, prefilter=False):
        logger.info('*' * 100)
        logger.info(f'We are editing the {rubric_kind} rubric with {len(mistakes)} mistakes: {rubric}')
        if len(mistakes) < 3:
            for mistake in mistakes:
                logger.info(f"\t{mistake['groundtruth']}\t{mistake['content']}\n\t{mistake['reflection']}\n")
                logger.info('-' * 50)
        system_prompt = f"""
            <Task>
                A content creator is writing down their content moderation preference as a prompt.
                Their prompts consist of three parts: 
                    - overall description of their preferences
                    - positive rubrics that address particular categories of comments they want to catch
                    - negative rubrics that address particular categories of comments they do not want to catch.
                This prompt is then used by crowdworkers to classify comments as either 1 (the comment matches the prompt) or 0 (the comment does not match the prompt). 
                However, the content creator might not clearly communicate their preferences, which could lead to misclassification by crowdworkers.
                
                For a given {rubric_kind} rubric, expert linguists has identified a set of misclassified comments that demonstrate which aspects of user preferences this rubric might fail to consider.
                Your task is to edit this {rubric_kind} rubric to incorporate these nuances in misclassified comments.
            </Task>

            <steps>
                <step1>
                    Examine these misclassified comments and the expert's suggestion for each mistake, and summarize what is missing in the original rubric.
                    Make sure you carefully read from the expert's suggestion that whether the content creator wants to catch or not catch a specific category of comments.
                    Do not hallucinate.
                </step1>
                <step2>
                    Examine these misclassified comments and the expert's suggestion for each mistake.
                    Reason how you might edit this rubric to make sure that crowdworkers can correctly classify such comments in the future.
                </step2>
                <step3>
                    Edit the identified rubric to make sure it can correctly classify such comments in the future.
                    As crowdworkers' performances are very sensitive to the prompt's wording, you should generate {candidate_num} candidate rubrics and we will later select the best one in a pilot study.

                    Here are a few more requirements for your new rubric:
                    - Each of your rubric should start with "Comments that ...".
                    - To help crowdworkers understand the new rubric, you are encouraged to provide at least two representative examples of SPECIFIC PHRASES/WORDS.
                        However, you should not remove any original examples in the problem rubric, as they are added in previous iterations of the prompt.
                    - Keep the language of the rubric concise and specific; avoid being verbose, ambiguous, or overly general.
                    - We are doing this for content moderation to protect users' online experiences, so please do not refrain from using sensitive words or phrases in your rubric, which are necessary to accurately capture the user's preferences.
                </step3>
                
            </steps>

            <Examples>
                <Example>
                    <ProblemRubric>Comments that talk about a person killing another person</ProblemRubric>
                    <ProblemComments>
                        <Comments>The content creators wants to catch comments that mention a person committing suicide.</Comments>
                    </ProblemComments>

                    <NewRubrics>
                        <Rubric>Comments that talk about a person killing another person or themselves.</Rubric>
                        <Rubric>Comments that focus on lethal harm, including a person killing someone else or taking their own life.</Rubric>
                        <Rubric>Comments that mention homocide or suicide.</Rubric>
                    </NewRubrics>
                </Example>
                <Example>
                    <ProblemRubric>Comments that use derogatory comments to insult other people, such as 'he is scumbag', 'douchebag'.</ProblemRubric>
                    <ProblemComments>
                        <Comments>The content creator does not want to catch comments that simply use 'fool' in the sentence.</Comments>
                        <Comments>The content creator wants to catch comments that use 'bastard' to insult others.</Comments>
                    </ProblemComments>

                    <NewRubrics>
                        <Rubric>Comments that use strong derogatory insults, including but not limited to “scumbag,” “douchebag,” or “bastard,” while excluding milder terms such as “fool” or “idiot.”</Rubric>
                        <Rubric>Comments containing highly offensive language (e.g., “scumbag,” “bastard,” “douchebag”) intended to demean others, noting that less severe insults like “fool” or “idiot” are not flagged.</Rubric>
                        <Rubric>Comments that convey extreme contempt through strong insults, such as “douchebag,” “bastard,” or “scumbag,” though milder words (e.g., “idiot,” “fool”) fall outside this rubric.</Rubric>
                    </NewRubrics>
                <Example>
            </Examples>

            Write your response in the following xml format.
            <Summary>Your summary of what is missing in the old rubric at the step 1</Summary>
            <Reasoning>Your reasoning of how to edit the rubric at step 2.</Reasoning>
            <NewRubrics>
                <Rubric>Write your new rubric here.</Rubric>
                <Rubric>Write your new rubric here.</Rubric>
                <Rubric>Write your new rubric here.</Rubric>
            </NewRubrics>
        """
        
        new_filters = [[] for _ in range(round)]
        def run(now_mistakes, round_index):
            nonlocal new_filters
            problem_comments_str = ""
            for comment in now_mistakes:
                problem_comments_str += f"""
                    <Comment>{comment['reflection']}</Comment>
                """
            user_prompt = f"""
                <Prompt>{filter.stringify_filter(structured=True)}</Prompt>
                <ProblemRubric>{rubric}</ProblemRubric>      
                <ProblemComments>
                    {problem_comments_str}
                </ProblemComments>
            """
            response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            ) 
            # logger.info(f'Edit rubric response\n: {response}\n\n')
            new_rubrics = self.llm_client.extract_xml(response, "Rubric")
            logger.info(f'[Prompt Candidates] Edit the {rubric_kind} rubric from: {rubric}')
            for new_rubric in new_rubrics:
                new_filter = copy.deepcopy(filter)
                logger.info(f'\t--{new_rubric}')
                new_filter.update_rubric(new_rubric, rubric_kind, comments=now_mistakes, old_rubric=rubric)
                new_filter.attributes['action'] = 'Edit an existing rubric'
                new_filters[round_index].append(new_filter)

        if len(mistakes) <= 10:
            # if there are not enough mistakes, we will simply run the task in a single thread
            run(mistakes, 0)
        else:
            threads = []
            for i in range(round):
                # Sample a batch of mistakes
                batch_mistakes = random.sample(mistakes, min(10, len(mistakes)))
                
                # Create and start a thread for this batch
                thread = threading.Thread(target=run, args=(batch_mistakes, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        
        new_filters = [filter for sublist in new_filters for filter in sublist]

        if prefilter:
            # evaluate the performances of the new filters on the mistakes dataset and select the top 3.
            batch_size = min(20, len(mistakes) // 3)
            epochs = int(len(mistakes) // batch_size * len(new_filters) * 0.4)
            best_filters = updates.select_best_filters(new_filters, mistakes, topN=candidate_num, epochs=epochs, batch_size=batch_size, exploration=5)
            return best_filters
        else:
            return new_filters

    def generate_few_shots(self, filter, comments, k=4, rounds=3, sample_preferences='short-comments'):
        """
        Add a few shots to the filter by asking the user to label the groundtruth of the comments.
        """
        def select_examples(sample_from):
            if sample_preferences == 'short-comments':
                sample_from = [comment for comment in sample_from if len(comment['content']) < 200]
            elif sample_preferences == 'high-confidence':
                # sort by confidence from high to low
                sample_from = sorted(sample_from, key=lambda x: x['confidence'])
                sample_from = sample_from[:3*k]
            elif sample_preferences == 'content-diversity':
                embeddings = np.array([comment['embedding'] for comment in sample_from])
                selected_indices = utils.sample_diverse_embeddings(embeddings, k)
                sample_from = [sample_from[i] for i in selected_indices]
            elif sample_preferences == 'failure-diversity':
                embeddings = np.array([comment['reflection-embedding'] for comment in sample_from])
                selected_indices = utils.sample_diverse_embeddings(embeddings, k)
                sample_from = [sample_from[i] for i in selected_indices]

            print(f'There are {len(sample_from)} comments to sample from.')
            if len(sample_from) <= k:
                logger.warning(f'Not enough comments to sample from. Returning all {len(sample_from)} comments.')
                return sample_from
            else:
                few_shots_examples = random.sample(sample_from, k)
                return few_shots_examples

        if sample_preferences == 'failure-diversity':
            for comment in comments:
                comment['reflection-embedding'] = self.llm_client.text_embedding(comment['reflection'])
        elif sample_preferences == 'content-diversity':
            for comment in comments:
                comment['embedding'] = self.llm_client.text_embedding(comment['content'])
        
        # we tentatively only select misclassified comments
        mistake_comments = [comment for comment in comments if comment['groundtruth'] != comment['prediction']]
        new_filters = []
        for _ in range(rounds):
            few_shots_examples = select_examples(mistake_comments)
            new_filter = copy.deepcopy(filter)
            # can we add new examples to the filter?
            new_filter.few_shots = few_shots_examples
            new_filters.append(new_filter)
        
        # examine whether any filter has the exact same few shots
        new_filters = utils.deduplicate_filters(new_filters)
        return new_filters
    
    def add_few_shots(self, filter, comments):
        comments_copy = [comment.copy() for comment in comments]
        new_filters = self.generate_few_shots(filter, comments_copy)
        best_filters = self.select_best_filters(new_filters, comments, topN=1)
        return best_filters[0]

    def generate_interesting_clusters(self, filter, mistakes, min_samples=3):
        logger.info(f'There are in total {len(mistakes)} mistakes for the filter {filter.name}')
        reflections = self.reflect_on_mistakes_parellel(filter, mistakes)
        for index, mistake in enumerate(mistakes):
            mistake['reflection'] = reflections[index]['reflection']
            mistake['embedding'] = reflections[index]['embedding']
        
        refine_clusters = []
        
        false_positives = [mistake for mistake in mistakes if mistake['groundtruth'] == 0]
        false_negativse = [mistake for mistake in mistakes if mistake['groundtruth'] == 1]
        # if we want to add a new negative rubric
        if false_positives:
            now_clusters = self.cluster_mistakes_unsupervised(false_positives, min_samples=min_samples)
            for cluster in now_clusters:
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'negative',
                    'action': 'add',
                })
        # if we want to add a new positive rubric
        if false_negativse:
            now_clusters = self.cluster_mistakes_unsupervised(false_negativse, min_samples=min_samples)
            for cluster in now_clusters:
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'positive',
                    'action': 'add'
                })

        # if we want to edit an existing positive rubric
        if filter.positives:
            now_clusters = self.cluster_mistakes_for_rubrics(filter.positives, mistakes, min_samples=min_samples)
            logger.info(f'We have {len(now_clusters)} clusters for editing positive rubrics')
            for positive_rubric, cluster in now_clusters.items():
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'positive',
                    'action': 'edit',
                    'rubric': positive_rubric
                })
        # if we want to edit an existing negative rubric
        if filter.negatives:
            now_clusters = self.cluster_mistakes_for_rubrics(filter.negatives, mistakes, min_samples=min_samples)
            logger.info(f'We have {len(now_clusters)} clusters for editing negative rubrics')
            for negative_rubric, cluster in now_clusters.items():
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'negative',
                    'action': 'edit',
                    'rubric': negative_rubric
                })
        
        # we will rank the clusters based on their size
        refine_clusters = sorted(refine_clusters, key=lambda x: len(x['cluster']), reverse=True)
        # TODO: we need to further summarize each cluster.
        return refine_clusters[:3]
    
    def interpret_refine_infos(self, filter, refine_cluster):
        problem_comments_str = ""
        for comment in refine_cluster['cluster']:
            problem_comments_str += f"""
                <Comment>{comment['reflection']}</Comment>
            """
        if refine_cluster['action'] == 'edit':
            rubric_kind = refine_cluster['kind']
            system_prompt = f"""
                <Task>
                    A content creator is writing down their content moderation preference as a prompt.
                    Their prompts consist of three parts: 
                        - overall description of their preferences
                        - positive rubrics that address particular categories of comments they want to catch
                        - negative rubrics that address particular categories of comments they do not want to catch.
                    This prompt is then used by crowdworkers to classify comments as either 1 (the comment matches the prompt) or 0 (the comment does not match the prompt). 
                    However, the content creator might not clearly communicate their preferences, which could lead to misclassification by crowdworkers.
                    
                    For a given {rubric_kind} rubric, expert linguists has identified a set of misclassified comments that demonstrate which aspects of user preferences this rubric might fail to consider.
                    Your task is to inform the content creator about how this {rubric_kind} rubric will be further edited to incorporate these nuances in misclassified comments.
                </Task>

                <steps>
                    <step1>
                        Examine these misclassified comments and the expert's suggestion for each mistake, and summarize what is missing in the original rubric.
                        Make sure you carefully read from the expert's suggestion that whether the content creator wants to catch or not catch a specific category of comments.
                        Do not hallucinate.
                    </step1>
                    <step2>
                        Examine these misclassified comments and the expert's suggestion for each mistake.
                        Reason how you might edit this rubric to make sure that crowdworkers can correctly classify such comments in the future.
                    </step2>
                    <step3>
                        Write down a sentence with less than 40 words that explains to the content creator how this {rubric_kind} rubric will be further edited to incorporate these nuances in misclassified comments.
                        
                        - Your explanation should be in the following format:
                            'After examining your annotations, do you want to [summary of the edits we want to make]?'
                            Your should try to draw a clear contrast betweeen what the current prompt fails and what you want to change it to.
                        - Keep the language of your explanation concise and specific; avoid being verbose, ambiguous, or overly general.
                        - We are doing this for content moderation to protect users' online experiences, so please do not refrain from using sensitive words or phrases in your rubric, which are necessary to accurately capture the user's preferences.
                    </step3>
                    
                </steps>

                <Examples>
                    <Example>
                        <ProblemRubric>Comments that talk about a person killing another person</ProblemRubric>
                        <ProblemComments>
                            <Comments>The content creators wants to catch comments that mention a person committing suicide.</Comments>
                        </ProblemComments>

                        <Explanation>
                            Upon examining your annotations, do you also want to catch comments about committing suicide?
                        </Explanation>
                    </Example>
                    <Example>
                        <ProblemRubric>Comments that use derogatory comments to insult other people, such as 'he is scumbag', 'douchebag'.</ProblemRubric>
                        <ProblemComments>
                            <Comments>The content creator does not want to catch comments that simply use 'fool' in the sentence.</Comments>
                            <Comments>The content creator wants to catch comments that use 'bastard' to insult others.</Comments>
                        </ProblemComments>

                        <Explanation>
                            After examining your annotations, do you want to further catch "bastard" while excluding milder terms like "fool"?
                        </Explanation>
                    <Example>
                </Examples>

                Write your response in the following xml format.
                <Summary>Your summary of what is missing in the old rubric at the step 1</Summary>
                <Reasoning>Your reasoning of how to edit the rubric at step 2.</Reasoning>
                <Explanation>Your explanation of how this rubric will be further edited at the step 3.</Explanation>
            """
            user_prompt = f"""
                <Prompt>{filter.stringify_filter(structured=True)}</Prompt>
                <ProblemRubric>{refine_cluster['rubric']}</ProblemRubric>      
                <ProblemComments>
                    {problem_comments_str}
                </ProblemComments>
            """
            response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            ) 
            logger.info(f'Interpret refine infos for editing a rubric response\n: {response}\n\n')
            interpretation = self.llm_client.extract_xml(response, "Explanation")
            return interpretation
        elif refine_cluster['action'] == 'add':
            rubric_kind = refine_cluster['kind']
            system_prompt = f"""
                <Task>
                    A content creator is writing down their content moderation preference as a prompt.
                    This prompt is then used by crowdworkers to classify comments as either 1 (the comment matches the prompt) or 0 (the comment does not match the prompt). 
                    However, the content creator might not clearly communicate their preferences, which could lead to misclassification by crowdworkers.
                    An expert linguist has examined these mistakes, suggested what is missing in the original prompt for each mistake.
                    Your task is to inform the content creator about how a new {rubric_kind} rubric will be added to incorporate these nuances in misclassified comments.
                </Task>

                <steps>
                    <step1>
                        Examine these misclassified comments and the expert's suggestion for each mistake.
                        Reason how you might add a new {rubric_kind} rubric to make sure that crowdworkers can correctly classify such comments in the future.
                    </step1>
                    <step2>
                        Write down a sentence with less than 40 words that explains to the content creator how a new {rubric_kind} rubric will be further added to incorporate these nuances in misclassified comments.
                        
                        - Your explanation should be in the following format:
                            'After examining your annotations, do you want to [summary of the new rubric we want to add]?'
                            Your should try to draw a clear contrast betweeen what the current prompt fails and what you want to change it to.
                        - Keep the language of your explanation concise and specific; avoid being verbose, ambiguous, or overly general.
                        - We are doing this for content moderation to protect users' online experiences, so please do not refrain from using sensitive words or phrases in your rubric, which are necessary to accurately capture the user's preferences.

                        Here are a few examples of good rubrics for your reference:
                        - Your filter currently only lists derogatory terms like "scumbag" and "douchebag" as examples, 
                            After examining your annotations, do you want to further catch "bastard" while excluding milder terms like "fool"?
                        - Your filter currently only lists derogatory terms like "scumbag" and "douchebag" as examples, 
                            After examining your annotations, do you want to further catch "bastard" while excluding milder terms like "fool"?
                    </step2>
                </steps>

                Write your response in the following xml format.
                <Reasoning>Your reasoning of how to add a new {rubric_kind} rubric at step 1.</Reasoning>
                <Explanation>Your explanation of how a new rubric will be further added at the step 2.</Explanation>
            """
            user_prompt = f"""
                <Prompt>{filter.stringify_filter(structured=True)}</Prompt>      
                <ProblemComments>
                    {problem_comments_str}
                </ProblemComments>
            """
            response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            ) 
            logger.info(f'Add new rubric response\n: {response}\n\n')
            interpretation = self.llm_client.extract_xml(response, "Explanation")
            return interpretation

    def refine_prompt(self, filter, refine_clusters, weighted=True):
        if not isinstance(refine_clusters, list):
            # when we only have one cluster, as opposed to a list of clusters
            if 'action' not in refine_clusters:
                # we need to generate clusters that correspond to various actions to refine the prompt
                refine_clusters = self.generate_interesting_clusters(filter, refine_clusters['cluster'])
            else:
                refine_clusters = [refine_clusters]
        
        start_time = time.time()
        # we add filter to the new filters to ensure that the performance does not drop
        # we deep copy it to avoid modifying the original filter
        new_filters = [copy.deepcopy(filter)]
        for refine_info in refine_clusters:
            if refine_info['action'] == 'add':
                new_filters.extend(self.add_new_rubric(filter, refine_info['cluster'], refine_info['kind']))
            else:
                new_filters.extend(self.edit_rubric(filter, refine_info['rubric'], refine_info['cluster'], refine_info['kind']))
        end_time = time.time()
        logger.info(f'It takes {end_time - start_time} seconds to generate prompt candidates for the filter {filter.name}')


        start_time = time.time()
        # TODO: determine how should we build the training dataset and how to highlight the mistakes.

        # we increase the weight of the focused comments
        if weighted:
            focused_comment_ids = []
            for refine_cluster in refine_clusters:
                focused_comment_ids.extend([comment['id'] for comment in refine_cluster['cluster']])
            focused_comment_ids = list(set(focused_comment_ids))
            
            for training_example in filter.training_examples:
                weight = 1
                if training_example['id'] in focused_comment_ids:
                    weight = 2
                training_example['weight'] = weight
        else:
            for training_example in filter.training_examples:
                training_example['weight'] = 1

        # we want to avoid wasting resources on predictions we have already calculated.
        new_filters[0].attributes['predictions'] = {
            example['id']: {
                'prediction': example['prediction'],
                'groundtruth': example['groundtruth'],
                'confidence': example['confidence'],
            } for example in filter.training_examples
        }

        best_filters = self.select_best_filters(
            new_filters, filter.training_examples, topN=1
        )
        end_time = time.time()
        logger.info(f'It takes {end_time - start_time} seconds to select the best filter among {len(new_filters)} candidates')

        return best_filters

    def select_best_filters(self, filters, comments, strategy='bandit', topN=1, epochs=None, batch_size=20, exploration=5):
        """
        Select the best filter based on the performance on the comments.

        @param filters: a list of filters to select from, eahc should be an instance of PromptFilterBackend
        @param comments: a list of comments of the type dict, with the key content and groundtruth
        @param strategy: the strategy to select the best filters, can be 'bandit' or 'overall'
        @param topN: the number of best filters to return
        @param epochs: the number of epochs to run
        @param batch_size: the number of comments to evaluate in each epoch
        @param exploration: the exploration factor for the bandit strategy

        @return: a list of the best filters
        """
        logger.info('#' * 100)
        if(len(comments) < 2 * batch_size):
            strategy = 'overall'
            logger.info(f'Not enough comments to run bandit strategy. Switching to overall strategy.')

        high_weights_comments_ids = set([comment['id'] for comment in comments if comment['weight'] > 1])
        if strategy == 'bandit':
            logger.info(f"Running bandit strategy with {len(filters)} filters to select the top {topN} from {len(comments)} comments.")
            
            records = []
            counts = [0] * len(filters)
            values = [0] * len(filters)

            def select_best_arm(t):
                
                for arm in range(len(filters)):
                    # If any arm has not been pulled yet, choose it to explore
                    if counts[arm] == 0:
                        return arm
                
                ucb_scores = []
                for arm in range(len(filters)):
                    average_reward = values[arm]
                    bonus = exploration * ((np.log(t) / counts[arm])) ** 0.5
                    ucb_scores.append(average_reward + bonus)
                best_arm = np.argmax(ucb_scores)
                logger.info(f"Selecting the best arm for round {t} as {best_arm}.")
                return best_arm

            epochs = epochs if epochs else math.ceil(math.ceil(len(comments) / batch_size) * len(filters) * 0.4)
            logger.info(f'\tWith {epochs} rounds, batch size {batch_size}, and exploration factor {exploration}.')
            for t in range(epochs):
                samples = random.sample(comments, batch_size)
                best_arm = select_best_arm(t)
                counts[best_arm] += 1
                best_filter = filters[best_arm]

                comments_copy = [comment.copy() for comment in samples]
                if 'predictions' not in best_filter.attributes:
                    best_filter.attributes['predictions'] = {}
                comments_copy = best_filter.predict_comments_consistently(comments_copy, best_filter.attributes['predictions'])

                for comment in comments_copy:
                    best_filter.attributes['predictions'][comment['id']] = {
                        'prediction': comment['prediction'],
                        'groundtruth': comment['groundtruth'],
                        'confidence': comment['confidence'],
                    }
                performance = utils.eval_performance(comments_copy, print_comments=False, weighted=False)
                values[best_arm] += performance['f1'] / counts[best_arm]

                actual_best_arm = np.argmax(values)
                records.append({
                    'round': t,
                    'actual_best_arm': actual_best_arm,
                    'count': counts[actual_best_arm],
                })
            # return the top N best filters
            best_arms = np.argsort(values)[::-1][:topN]
            best_filters = [filters[arm] for arm in best_arms]
            logger.info(f"Best arm selected as {best_arms[0]} with the highest value of {values[best_arms[0]]}.")
            return best_filters
        elif strategy == 'overall':
            logger.info(f"Running overall strategy with {len(filters)} filters to select the top {topN}.")
            performances = []
            performances_high_weights = []
            for index, new_filter in enumerate(filters):
                comments_copy = [comment.copy() for comment in comments]
                comments_copy = new_filter.predict_comments_consistently(comments_copy)
                # we want to save the predictions in the filter attributes
                # so that we can use them when this filter is accepted.
                new_filter.attributes['predictions'] = {
                    comment['id']: {
                        'prediction': comment['prediction'],
                        'groundtruth': comment['groundtruth'],
                        'confidence': comment['confidence'],
                    } for comment in comments_copy
                }
                performance = utils.eval_performance(comments_copy, print_comments=False, weighted=False)
                performances.append(performance['f1'])
                logger.info(f'Filter {index}: Accuracy {performance["accuracy"]}, F1 {performance["f1"]}, Precision {performance["precision"]}, Recall {performance["recall"]}')
                
                higher_weight_comments = [comment for comment in comments_copy if comment['id'] in high_weights_comments_ids]
                if len(high_weights_comments_ids) > 0:
                    perf_on_higher_weights_comments = utils.eval_performance(higher_weight_comments, print_comments=False)
                    performances_high_weights.append(perf_on_higher_weights_comments['f1'])
                    logger.info(f'Accuracy on higher weight comments: {perf_on_higher_weights_comments["f1"]}')
            # return the top N best filters
            logger.info('#' * 100)
            if len(high_weights_comments_ids) == 0:
                best_indices = np.argsort(performances)[::-1][:topN]
                best_filters = [filters[i] for i in best_indices]
                return best_filters
            else:
                # we want to first make sure that the performance on the high weights comments is not too low
                base_perf = performances[0]
                baseline_perf_high_weights = performances_high_weights[0]
                best_indices = [0]
                filters[0].attributes['kind'] = 'original'
                for i in range(len(performances)):
                    # we do not consider the original filter as best filters
                    if performances_high_weights[i] > baseline_perf_high_weights:
                        best_indices.append(i)
                        changed_mistakes = abs(int((performances[i] - base_perf) * len(comments)))
                        filters[i].attributes['changedMistakes'] = changed_mistakes
                        if performances[i] > base_perf:
                            filters[i].attributes['kind'] = 'best'
                        else:
                            filters[i].attributes['kind'] = 'trade-off'
                # we further sort the best indices based on their overall performance from the highest to the lowest
                # if there are indices before the original filter, they represent an optimal refinement of the original filter
                # if there are only indices after the original filter, they represent a trade-off users need to make
                # otherwise, we do not find any filter that is better than the original filter in any sense.
                best_indices = sorted(best_indices, key=lambda x: performances[x], reverse=True)
                best_filters = [filters[i] for i in best_indices]
                return best_filters

    def calibrate_prompt(self, filter, annotations, rounds=3):
        """
        Calibrate the filter by running automatic prompt optimization algorithms.
        """
        for annotation in annotations:
            annotation['weight'] = 1

        start_time = time.time()
        def identify_mistakes(now_filter, first_round=False):
            annotations_copy = [annotation.copy() for annotation in annotations]
            if not first_round:
                # we do not need to predict the comments for the first round as we save the predictions in database.
                annotations_copy = now_filter.predict_comments_consistently(annotations_copy)
            mistakes = [annotation for annotation in annotations_copy if annotation['groundtruth'] != annotation['prediction']]
            return mistakes
        
        for round_index in range(rounds):
            logger.info('$' * 150)
            logger.info(f'Round {round_index + 1} of calibration for the filter {filter.name}')
            mistakes = identify_mistakes(filter, round_index == 0)
            refine_clusters = self.generate_interesting_clusters(filter, mistakes)
            refined_filters = self.refine_prompt(filter, refine_clusters, weighted=False)
            filter = refined_filters[0]
        
        logger.info('$' * 150)
        logger.info(f'We finally start to add few shots to the filter')
    
        # then add few shot examples
        filter = self.add_few_shots(filter, annotations)

        end_time = time.time()
        logger.info(f'It takes {end_time - start_time} seconds to calibrate the filter {filter.name}')
        return filter

    def label_groundtruth(self, now_comments, filter):
        # Current index tracker
        current_index = 0
        

        comment_html = widgets.HTML(layout=widgets.Layout(width='50%'))
        positive_button = widgets.Button(description="Positive", button_style="success")
        negative_button = widgets.Button(description="Negative", button_style="danger")
        show_explanation_button = widgets.Button(description="Show Explanation", button_style="info")
        progress_label = widgets.Label()

        def update_progress():
            progress_label.value = f'Progress: {current_index}/{len(now_comments)}'
        
        def display_current_comment():
            """
            Display the current comment without the explanation by default.
            """
            if current_index < len(now_comments):
                content = now_comments[current_index]["content"]
                prediction = now_comments[current_index]["prediction"]
                comment_html.value = (
                    f'<div style="white-space: pre-wrap; word-wrap: break-word;">'
                    f'<strong>Content:</strong> {content}<br>'
                    f'<strong>Prediction:</strong> {prediction}<br>'
                    f'<em>Explanation not shown yet</em>'
                    f'</div>'
                )
                show_explanation_button.disabled = False
            else:
                comment_html.value = '<div style="font-weight: bold;">All comments labeled!</div>'
                positive_button.disabled = True
                negative_button.disabled = True
                show_explanation_button.disabled = True

        def label_comment(label):
            nonlocal current_index
            if current_index < len(now_comments):
                now_comments[current_index]['groundtruth'] = label

                # Move to next unlabeled comment
                while current_index < len(now_comments) and 'groundtruth' in now_comments[current_index]:
                    current_index += 1

                update_progress()

                display_current_comment()

        def show_explanation(btn):
            # Only show explanation if we have a valid current_index
            explanation = self.interpret_comment(now_comments[current_index])
            content = now_comments[current_index]["content"]
            prediction = now_comments[current_index]["prediction"]
            comment_html.value = (
                f'<div style="white-space: pre-wrap; word-wrap: break-word;">'
                f'<strong>Content:</strong> {content}<br>'
                f'<strong>Prediction:</strong> {prediction}<br>'
                f'<strong>LLM Explanation:</strong> {explanation}'
                f'</div>'
            )

        # Link buttons to the labeling function
        positive_button.on_click(lambda btn: label_comment(1))
        negative_button.on_click(lambda btn: label_comment(0))
        show_explanation_button.on_click(show_explanation)

        update_progress()
        display(widgets.VBox([
            progress_label,
            comment_html,
            widgets.HBox([positive_button, negative_button, show_explanation_button])
        ]))
        display_current_comment()

    def helllo_world(self):
        print('hello world dddd')