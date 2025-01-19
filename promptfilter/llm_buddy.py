import copy
import comm
import ipywidgets as widgets
from IPython.display import display
import logging
import numpy as np
import random


from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from .chat_completion import ChatCompletion
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

    def initialize_prompt(self, example):
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
        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        proposed_prompt = self.llm_client.extract_xml(response, "Prompt")
        return proposed_prompt

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
                <Rubric>{filter['description']}</Rubric>
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
                            - comments that insults a group ('the grand jury') rather than specific individuals.
                            - comments that only use the term "stupid" as the only derogatory term.
                    </Reasoning>
                </Example>
                <Example>
                    <Prompt>Comments that talk about a person killing another person.</Prompt>
                    <Comment>I want to commit suicide.</Comment>
                    <Groundtruth>1</Groundtruth>
                    
                    <Reasoning>
                        This comment is considered by the content creator as matching the prompt possibly because they want to catch
                            -  comments that mention 'suicide' in addition to killing others. 
                    </Reasoning>
                </Example>
            </Examples>

            Write your output strictly in the following xml format:
            <Reasoning>Write down your reasoning in step 1 here.</Reasoning>
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
        logger.info(f'Reflect preference response: {response}\n')
        reasons = self.llm_client.extract_xml(response, "Reasoning")
        return reasons
    
    def cluster_mistakes_unsupervised(self, mistakes):
        """Cluster mistakes based on their similarity which could help propose a new rubric."""
        def generate_clusters(now_mistakes, eps, min_samples=2):
            now_embeddings = [item['embedding'] for item in now_mistakes]
            now_embeddings = np.array(now_embeddings)
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
            logger.info(f'We have {len(new_clusters)} clusters with the size of {cluster_sizes_str} with eps={eps}')
            cluster_candidates.extend(new_clusters)
            eps += 0.1

        # order the cluser candidates by their length and return the longest one
        cluster_candidates = sorted(cluster_candidates, key=lambda x: len(x), reverse=True)
        return cluster_candidates[:1]

    def cluster_mistakes_for_rubrics(self, rubrics, mistakes, threshold=0.9):
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
        
        for rubric, mistakes in clustered_mistakes.items():       
            logger.info(f'Clustered {len(mistakes)} mistakes for rubric: {rubric}')
            if len(mistakes) == 0:
                del clustered_mistakes[rubric]
            
        return clustered_mistakes

    def add_new_rubric(self, filter, mistakes, rubric_kind, candidate_num=2, round=2, prefilter=False):
        logger.info('*' * 100)
        logger.info(f'We are adding a new {rubric_kind} rubric with {len(mistakes)} mistakes')
        for mistake in mistakes:
            logger.info(f"\t{mistake['groundtruth']}\t{mistake['content']}\n\t{mistake['reflection']}\n")
            logger.info('-' * 50)
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
        
        def run(now_mistakes):
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
            logger.info(f'Add new rubric response\n: {response}\n\n')
            new_rubrics = self.llm_client.extract_xml(response, "Rubric")
            return new_rubrics
        
        new_filters = []
        for _ in range(round):
            # we will try different batches for iteration;
            # if the number of mistakes is less than 20, we will also try to randomize the mistakes
            now_mistakes = random.sample(mistakes, min(20, len(mistakes)))
            new_rubrics = run(now_mistakes)
            for new_rubric in new_rubrics:
                new_filter = copy.deepcopy(filter)
                new_filter.update_rubric(new_rubric, rubric_kind, comments=now_mistakes)
                new_filters.append(new_filter)
        
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
        
        def run(now_mistakes):
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
            logger.info(f'Edit rubric response\n: {response}\n\n')
            new_rubrics = self.llm_client.extract_xml(response, "Rubric")
            return new_rubrics
        
        new_filters = []
        for _ in range(round):
            # we will try different batches for iteration.
            now_mistakes = random.sample(mistakes, min(10, len(mistakes)))
            new_rubrics = run(now_mistakes)
            for new_rubric in new_rubrics:
                new_filter = copy.deepcopy(filter)
                new_filter.update_rubric(new_rubric, rubric_kind, comments=now_mistakes, old_rubric=rubric)
                new_filters.append(new_filter)
        
        if prefilter:
            # evaluate the performances of the new filters on the mistakes dataset and select the top 3.
            batch_size = min(20, len(mistakes) // 3)
            epochs = int(len(mistakes) // batch_size * len(new_filters) * 0.4)
            best_filters = updates.select_best_filters(new_filters, mistakes, topN=candidate_num, epochs=epochs, batch_size=batch_size, exploration=5)
            return best_filters
        else:
            return new_filters
    
    def add_few_shots(self, filter, comments, k=4, rounds=3, sample_preferences=None):
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
            new_filter.few_shots = few_shots_examples
            new_filters.append(new_filter)
        
        # examine whether any filter has the exact same few shots
        new_filters = utils.deduplicate_filters(new_filters)
        return new_filters
    
    def refine_prompt(self, filter, mistakes):
        new_filters = []
        for mistake in mistakes:
            mistake['embedding'] = self.llm_client.text_embedding(mistake['reflection'])
        
        refine_clusters = []
        # if we want to add a new rubric
        false_positives = [mistake for mistake in mistakes if mistake['groundtruth'] == 0]
        false_negativse = [mistake for mistake in mistakes if mistake['groundtruth'] == 1]
        if false_positives:
            now_clusters = self.cluster_mistakes_unsupervised(false_positives)
            for cluster in now_clusters:
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'negative',
                    'action': 'add'
                })
        
        if false_negativse:
            now_clusters = self.cluster_mistakes_unsupervised(false_negativse)
            for cluster in now_clusters:
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'positive',
                    'action': 'add'
                })

        # if we want to edit an existing rubric
        if filter.positives:
            now_clusters = self.cluster_mistakes_for_rubrics(filter.positives, mistakes)
            for positive_rubric, cluster in now_clusters.items():
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'positive',
                    'action': 'edit',
                    'rubric': positive_rubric
                })
        
        if filter.negatives:
            now_clusters = self.cluster_mistakes_for_rubrics(filter.negatives, mistakes)
            for negative_rubric, cluster in now_clusters.items():
                refine_clusters.append({
                    'cluster': cluster,
                    'kind': 'negative',
                    'action': 'edit',
                    'rubric': negative_rubric
                })
        
        # we will rank the clusters based on their size
        refine_clusters = sorted(refine_clusters, key=lambda x: len(x['cluster']), reverse=True)
        for refine_info in refine_clusters[:3]:
            if refine_info['action'] == 'add':
                new_filters.extend(self.add_new_rubric(filter, refine_info['cluster'], refine_info['kind']))
            else:
                new_filters.extend(self.edit_rubric(filter, refine_info['rubric'], refine_info['cluster'], refine_info['kind']))
        return new_filters

    def select_best_filters(self, filters, comments, strategy='bandit', topN=1, epochs=20, batch_size=20, exploration=5):
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

        if strategy == 'bandit':
            logger.info(f"Running bandit strategy with {len(filters)} filters to select the top {topN}.")
            logger.info(f'\tWith {epochs} rounds, batch size {batch_size}, and exploration factor {exploration}.')
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


            for t in range(epochs):
                samples = random.sample(comments, batch_size)
                best_arm = select_best_arm(t)
                counts[best_arm] += 1
                best_filter = filters[best_arm]

                mistakes_copy = [comment.copy() for comment in samples]

                mistakes_copy = best_filter.predict_comments_consistently(mistakes_copy)
                performance = utils.eval_performance(mistakes_copy, best_filter, print_comments=False)
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
            logger.info(f"Best arm selected as {best_arm} with the highest value of {values[best_arm]}.")
            return best_filters
        elif strategy == 'overall':
            logger.info(f"Running overall strategy with {len(filters)} filters to select the top {topN}.")
            performances = []
            for new_filter in filters:
                print(f'Evaluate the filter:\n {new_filter.stringify_filter(structured=False)}\n')
                mistakes_copy = [comment.copy() for comment in comments]
                mistakes_copy = new_filter.predict_comments_consistently(mistakes_copy)
                performance = utils.eval_performance(mistakes_copy, new_filter, print_comments=False)
                performances.append(performance['f1'])
                print('Performance:', performance)
            # return the top N best filters
            best_indices = np.argsort(performances)[::-1][:topN]
            best_filters = [filters[i] for i in best_indices]
            return best_filters

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