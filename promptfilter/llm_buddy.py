import enum
from django.conf import settings
import logging
import random
import json
import threading
import time

from .chat_completion import ChatCompletion

logger = logging.getLogger(__name__)

class LLMBuddy:

    def __init__(self):
        self.llm_client = ChatCompletion()

    def explain_prediction(self, filter, prediction):
        """
            Explain the prediction of the model based on the given prompt and comment
        """
        system_prompt = """
            A user is writing down their content moderation preferences in prompts to receive a label (1 represents that the text should be removed, and 0 represents that the text should be kept).
            However, they have trouble understanding why certain texts were assigned their particular labels.
            Your task is to explain classification decisions based on the given prompt, text, and assigned label.

            ### TASK DESCRIPTION
            STEP 1: Reason about why the text was assigned its label based on its contents and the prompt that classified it.
            STEP 2: Abbreviate this explanation for users.


            ### EXAMPLE 1
            #### INPUT
            <Prompt><Rubric>Remove texts that talk about a person killing another person</Rubric></Prompt>
            <Text>I want to commit suicide</Text>
            <Label>0</Label>

            #### OUTPUT
            <Reasoning>
                The prompt is supposed to remove texts that talk about a person killing someone else.
                Since the text mentions killing oneself and not someone else, the text was not removed.
            </Reasoning>
            <Explanation>
                Texts mentioning killing oneself are not covered by the prompt.
            </Explanation>
        """

        user_prompt = f"""
            <Prompt><Rubric>{filter.description}</Rubric></Prompt>
            <Text>{prediction.comment.content}</Text>
            <Label>{prediction.prediction}</Label>
        """

        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        explanation = self.llm_client.extract_xml(response, "Explanation")
        return explanation
    
    def cluster_mistakes(self, mistakes):
        system_prompt = """
            A user is writing down their content moderation preferences in prompts but has difficulties clearly communicating their preferences. 
            However, they could intuitively tell the ground truth of a text (1 represents that the text should be removed, and 0 represents that the text should be kept).
            A prompt has misclassified the given example comments.
            Your task is to cluster some of the comments based on similar contents and reasons for being misclassified.


            ### TASK DESCRIPTION
            STEP 1: Reason about similarities between the comments, grouping them into clusters. Note that all comments were misclassified.
            STEP 2: Group some of the comments based on their similarities.


            ### EXAMPLE 1
            #### INPUT
            <Comment>
                <Index>1</Index>
                <Text>I want to commit suicide</Text>
                <Explanation>References to killing oneself are not included in the prompt.</Explanation>
                <Groundtruth>1</Groundtruth>
            </Comment>
            <Comment>
                <Index>2</Index>
                <Text>Sometimes I just want to die</Text>
                <Explanation>Wishes for suicide are not included in the prompt about killing others.</Explanation>
                <Groundtruth>1</Groundtruth>
            </Comment>
            <Comment>
                <Index>3</Index>
                <Text>Your jokes are killing me</Text>
                <Explanation>Mention of killing triggers removal.</Explanation>
                <Groundtruth>0</Groundtruth>
            </Comment>
            <Comment>
                <Index>4</Index>
                <Text>That semester just killed my motivation to try</Text>
                <Explanation>Discussions of killing cause removal.</Explanation>
                <Groundtruth>0</Groundtruth>
            </Comment>
            <Comment>
                <Index>5</Index>
                <Text>She is a dumb bitch</Text>
                <Explanation>Insults to intelligence are not removed by this prompt.</Explanation>
                <Groundtruth>1</Groundtruth>
            </Comment>

            #### OUTPUT
            <Reasoning>
                The first two positive examples mention a person killing oneself, so they should be clustered together.
                The third and fourth negative examples mention killing metaphorically, so they should be clustered together.
                The fifth positive example is unrelated to killing.

                Let's output the first cluster about a person killing oneself.
            </Reasoning>
            <Clustering>
                1, 2
            </Clustering>
            
            
            ### EXAMPLE 2
            #### INPUT
            <Comment>
                <Index>1</Index>
                <Text>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Text>
                <Explanation>The prompt says to remove insults against groups, and a jury is a group.</Explanation>
                <Groundtruth>0</Groundtruth>
            </Comment>
            <Comment>
                <Index>2</Index>
                <Text>Republicans are full of bullshit. Never trust them</Text>
                <Explanation>The comment insults Republicans as a group, so it is removed.</Explanation>
                <Groundtruth>0</Groundtruth>
            </Comment>
            <Comment>
                <Index>3</Index>
                <Text>She is autistic I mean look at her</Text>
                <Explanation>Describing someone as autistic may not be seen as a direct attack</Explanation>
                <Groundtruth>1</Groundtruth>
            </Comment>
            <Comment>
                <Index>4</Index>
                <Text>Lots of my family are autistic</Text>
                <Explanation>The text mentions autism in an insulting way.</Explanation>
                <Groundtruth>0</Groundtruth>
            </Comment>
            <Comment>
                <Index>5</Index>
                <Text>Neurodivergence is a woke term. Liberals throw it around but it just means people are demented</Text>
                <Explanation>Insults to people with mental conditions should be removed by the prompt.</Explanation>
                <Groundtruth>1</Groundtruth>
            </Comment>
            #### OUTPUT
            <Reasoning>
                The positive examples (the third and fifth example) use "autistic" and "neurodivergence" as insults against individuals.
                The three negative examples (the first, second, and fourth example) mention groups, with two insulting groups and one just describing a group.

                Let's output the second cluster about groups.
            </Reasoning>
            <Clustering>
                1, 2, 4
            </Clustering>
        """

        user_prompt = ""
        for index, mistake in enumerate(mistakes):
            user_prompt += f"""
                <Comment>
                    <Index>{index + 1}</Index>
                    <Text>{mistake.comment.content}</Text>
                    <Explanation>{mistake.explanation}</Explanation>
                    <Groundtruth>{mistake.groundtruth}</Groundtruth>
                </Comment>
            """

        response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            )
        print('Improving suggestion response: %s', response)
        clustering = self.llm_client.extract_xml(response, "Clustering")

        clustered_indices = [int(index) - 1 for index in clustering.split(",")]
        return [mistakes[index] for index in clustered_indices]
    
    def explain_cluster(self, clusters):
        system_prompt = """
            A user is writing down their content moderation preferences in prompts but has difficulties clearly communicating their preferences. 
            However, they could intuitively tell the ground truth of a text (1 represents that the text should be removed, and 0 represents that the text should be kept).
            A prompt misclassified example comments, which were clustered together.
            Your task is to explain why the given comments were clustered together.


            ### TASK DESCRIPTION
            STEP 1: Reason about similarities between the comments. Note that all comments were misclassified.
            STEP 2: Describe the similarities in this form: "# comments that contain..." (e.g., "3 comments that contain general stereotypical language about African Americans")


            ### EXAMPLE 1
            #### INPUT
            <Prompt>
                <Rubric>Remove texts containing explicit or implied threats.</Rubric>
            </Prompt>
            <Comments>
                <Comment>
                    <Index>1</Index>
                    <Text>I want to commit suicide</Text>
                    <Explanation>References to killing oneself are not included in the prompt.</Explanation>
                    <Groundtruth>1</Groundtruth>
                </Comment>
                <Comment>
                    <Index>2</Index>
                    <Text>Sometimes I just want to die</Text>
                    <Explanation>Wishes for suicide are not included in the prompt about killing others.</Explanation>
                    <Groundtruth>1</Groundtruth>
                </Comment>
            <Comments>


            #### OUTPUT
            <Reasoning>
                Both comments express a personal desire for self-harm or suicide rather than an explicit or implied threat towards others. 
                The prompt specifies removing content containing threats, but these comments reflect a desire for self-harm, which does not clearly align with the provided rubric.
            </Reasoning>
            <Explanation>
                2 comments that contain references to suicide, as opposed to threating others.
            </Explanation>
            
            ### EXAMPLE 2
            #### INPUT
            <Prompt>
                <Rubric>Remove comments that use derogatory terms such as stupid, idiot, losers to insult individuals.</Rubric>
            </Prompt>
            <Comments>
                <Comment>
                    <Index>1</Index>
                    <Text>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Text>
                    <Explanation>The prompt says to remove insults against groups, and a jury is a group.</Explanation>
                    <Groundtruth>0</Groundtruth>
                </Comment>
                <Comment>
                    <Index>2</Index>
                    <Text>Republicans are full of bullshit. Never trust them</Text>
                    <Explanation>The comment insults Republicans as a group, so it is removed.</Explanation>
                    <Groundtruth>0</Groundtruth>
                </Comment>
                <Comment>
                    <Index>3</Index>
                    <Text>Lots of my family are autistic</Text>
                    <Explanation>The text mentions autism in an insulting way.</Explanation>
                    <Groundtruth>0</Groundtruth>
                </Comment>
            <Comments>

            #### OUTPUT
            <Reasoning>
                The comments vary in the targets of their criticism but share a theme of negatively addressing groups through derogatory language.
                In comparison, the prompt specifies removing insults against individuals, which may not align with the content of these comments.
            </Reasoning>
            <Explanation>
                3 comments that contain references to groups, as opposed to specific individuals.
            </Explanation>
        """
        user_prompt = ""
        for index, mistake in enumerate(clusters):
            user_prompt += f"""
                <Comment>
                    <Index>{index }</Index>
                    <Text>{mistake.comment.content}</Text>
                    <Explanation>{mistake.explanation}</Explanation>
                    <Groundtruth>{mistake.groundtruth}</Groundtruth>
                </Comment>
            """
        response = self.llm_client.chat_completion(
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                type="text"
            )
        print('Generate explanations for clusters: %s', response)
        summary = self.llm_client.extract_xml(response, "Explanation")
        return summary
        
    def improve_suggestion(self, filter, mistakes):
        """Generate a suggestion to improve the prompt based on the given mistakes"""
        # threads = []
        # def process_mistake(mistake):
        #     if not mistake.explanation:
        #         mistake.explanation = self.explain_prediction(filter, mistake)

        # for mistake in mistakes:
        #     thread = threading.Thread(target=process_mistake, args=(mistake,))
        #     threads.append(thread)
        #     thread.start()

        # for thread in threads:
        #     thread.join()
        
        # clusters = self.cluster_mistakes(mistakes)
        # explanation = self.explain_cluster(clusters)
        # return explanation, clusters
        return '7 comments that contain the vulgar term "fuck," which is often considered inappropriate language for public discourse.', mistakes
    
    def clarify_prompt(self, filter, prediction):
        system_prompt = """
            A user is writing down their content moderation preferences in prompts but has difficulties clearly communicating their preferences. 
            However, they could intuitively tell the groundtruth of a text (1 represents that the text should be removed, and 0 represents that the text should be kept).
            The current prompt has misclassified the given example comment.
            Your task is to ask the user a question to clarify why they gave the example its particular groundtruth label.


            ### TASK DESCRIPTION
            STEP 1: Reason why exactly the prompt misclassified the given example comment.
            STEP 2: Ask a specific, simple, yes-no question in the format "Do you want to remove/keep this comment because it ...?" with only one clause.

        
            ### EXAMPLE 1
            #### INPUT
            <Prompt><Rubric>Remove texts that talk about a person killing another person</Rubric></Prompt>
            <Text>I want to commit suicide</Text>
            <Groundtruth>1</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The positive example suggests that the user also wants to remove mentions of suicide rather than simply a person killing another person. 
                Therefore, the prompt might classify the example incorrectly. 
            </Reasoning>
            <Follow-up>
                Do you want to remove this comment because it mentions a person killing themselves?
            </Follow-up>


            ### EXAMPLE 2
            #### INPUT
            <Prompt><Rubric>Remove comments that use derogatory terms to insult individuals.</Rubric></Prompt>
            <Text>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Text>
            <Groundtruth>0</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The negative example suggests that the user might still want to see comments that use derogatory terms against a group.
                Therefore, the prompt might classify the example incorrectly. 
            </Reasoning>
            <Follow-up>
                Do you want to keep this comment because it insults a group of people (a jury)?
            </Follow-up>


            ### EXAMPLE 3
            #### INPUT
            <Prompt><Rubric>Remove texts that demean a persons intelligence</Rubric></Prompt>
            <Text>most people are too stupid to own a gun</Text>
            <Groundtruth>1</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The positive example suggests that the user also wants to remove comments demeaning multiple people's intelligence rather than simply the intelligence of a single person. 
                Therefore, the prompt might classify the example incorrectly.
            </Reasoning>
            <Follow-up>
                Do you want to remove this comment because it demeans multiple people's intelligence?
            </Follow-up>


            ### EXAMPLE 4
            #### INPUT
            <Prompt><Rubric>Remove texts that stereotypes political parties and puts them in bad lights.</Rubric></Prompt>
            <Text>Republican control states are more dangerous than Democrats control states.</Text>
            <Groundtruth>0</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The negative example suggests that the user might still want to see comments that put states in "bad lights" even if the states are associated with their political controllers.
                Therefore, the prompt might classify the example incorrectly.
            </Reasoning>
            <Follow-up>
                Do you want to keep this comment because it does not stereotype political parties directly, since it focuses on states instead of solely on political parties?
            </Follow-up>


            ### EXAMPLE 5
            #### INPUT
            <Prompt><Rubric>Remove texts containing explicit or implied threats towards individuals or groups.</Rubric></Prompt>
            <Text>a society that wants peace must execute swiftly and without bias those that commit murder genesis 9 verse 6. you cannot have real freedom if you allow murderers to live. it is impossible</Text>
            <Groundtruth>1</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The positive example suggests that the user also wants to remove comments implying the threat of death against murderers. 
                Therefore, the prompt might classify the example incorrectly. 
            </Reasoning>
            <Follow-up>
                Do you want to remove this comment because the idea that murderers should be killed is an implied threat?
            </Follow-up>
            

            ### EXAMPLE 6
            #### INPUT
            <Prompt><Rubric>Remove texts that promote Nazism.</Rubric></Prompt>
            <Text>The left needs to strap up because if the reich wing takes over, death squads will be a thing with those people.</Text>
            <Groundtruth>0</Groundtruth>

            #### OUTPUT
            <Reasoning>
                The negative example suggests that the user might still want to see comments that use Nazi-related terms like "reich" as long as they do not promote Nazism.
                Therefore, the prompt might classify the example incorrectly.
            </Reasoning>
            <Follow-up>
                Do you want to keep this comment because it uses Nazi-related terms ("reich") to caution against Nazi-like behaviors rather than to promote Nazism?
            </Follow-up>
        """
        user_prompt = f"""
            <Prompt><Rubric>{filter.description}</Rubric></Prompt>
            <Text>{prediction.comment.content}</Text>
            <Groundtruth>{prediction.groundtruth}</Groundtruth>
        """
        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        print('Propose follow-up question response: %s', response)
        followup = self.llm_client.extract_xml(response, "Follow-up")
        return followup
    
    def refine_prompt(self, filter, prediction, followup):
        system_prompt = """
            A user is writing down their content moderation preferences in prompts but has difficulties clearly communicating their preferences. 
            However, they could intuitively tell the groundtruth of a text (1 represents that the text should be removed, and 0 represents that the text should be kept).
            The current prompt has misclassified the given example comment.
            The user answered a question to help clarify their preferences.
            Your task is to edit the prompt according to the user's answer.


            ### TASK DESCRIPTION
            STEP 1: Reason about the question and the user's answer.
            STEP 2: Follow the below rules to edit the prompt based on your earlier reasoning.


            ### RULES FOR EDITING
            RULE 1: Only make edits under 10 words.
            RULE 2: MOVE positive examples to a bulleted list beginning with "For example, remove:"
            RULE 3: MOVE negative examples to a bulleted list beginning with "However, do NOT remove:"
            
            ### EXAMPLE 1
            #### INPUT
            <Prompt><Rubric>Remove texts that talk about a person killing another person</Rubric></Prompt>
            <Text>I want to commit suicide</Text>
            <Groundtruth>1</Groundtruth>
            <Question>
                Do you want to remove this comment because it mentions a person killing themselves?
            </Question>
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to remove comments that mention suicide.
                However, the original prompt only mentions removing texts that talk about a person kiling another person, not killing themselves.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove texts that talk about a person killing another person or a person killing themselves.
            </ImprovedPrompt>


            ### EXAMPLE 2
            #### INPUT
            <Prompt><Rubric>Remove comments that use derogatory terms to insult individuals.</Rubric></Prompt>
            <Text>That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is</Text>
            <Groundtruth>0</Groundtruth>
            <Question>
                Do you want to keep this comment because it insults a group of people (a jury)?
            </Question>  
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to keep comments that use derogatory terms to insult groups.
                However, the original prompt does not specify what to do in the case of groups.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove comments that use derogatory terms to insult individuals. However, do NOT remove:
                    - texts that use derogatory terms to insult groups
            </ImprovedPrompt>


            ### EXAMPLE 3
            #### INPUT
            <Prompt><Rubric>Remove texts that demean a persons intelligence</Rubric></Prompt>
            <Text>most people are too stupid to own a gun</Text>
            <Groundtruth>1</Groundtruth>
            <Question>
                Do you want to remove this comment because it demeans multiple people's intelligence?
            </Question>  
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to remove comments that demean multiple people's intelligence.
                However, the original prompt only mentions removing texts that demean one person's intelligence, not multiple people's intelligence.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove texts that demean a persons intelligence or multiple people's intelligence.
            </ImprovedPrompt>


            ### EXAMPLE 4
            #### INPUT
            <Prompt><Rubric>Remove texts that stereotypes political parties and puts them in bad lights.</Rubric></Prompt>
            <Text>Republican control states are more dangerous than Democrats control states.</Text>
            <Groundtruth>0</Groundtruth>
            <Question>
                Do you want to keep this comment because it does not stereotype political parties directly, since it focuses on states instead of solely on political parties?
            </Question>  
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to keep comments that focus on states being controlled by political parties instead of solely on political parties.
                However, the original prompt only mentions removing texts that stereotype political parties, failing to mention a negative example: focusing on states in relation to political parties.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove texts that stereotype political parties and put them in bad lights. However, do NOT remove:
                    - texts that mention political parties only in relation to the states that they control
            </ImprovedPrompt>


            ### EXAMPLE 5
            #### INPUT
            <Prompt><Rubric>Remove texts containing explicit or implied threats towards individuals or groups.</Rubric></Prompt>
            <Text>a society that wants peace must execute swiftly and without bias those that commit murder genesis 9 verse 6. you cannot have real freedom if you allow murderers to live. it is impossible</Text>
            <Groundtruth>1</Groundtruth>
            <Question>
                Do you want to remove this comment because the idea that murderers should be killed is an implied threat?
            </Question>  
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to remove comments implying that murderers should be killed.
                However, the original prompt only mentions implied threats in general, failing to mention a positive example: texts implying that murderers should be killed.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove texts containing explicit or implied threats towards individuals or groups. For example, remove:
                    - texts implying that murderers should be killed
            </ImprovedPrompt>
            

            ### EXAMPLE 6
            #### INPUT
            <Prompt><Rubric>Remove texts that promote Nazism.</Rubric></Prompt>
            <Text>The left needs to strap up because if the reich wing takes over, death squads will be a thing with those people.</Text>
            <Groundtruth>0</Groundtruth>
            <Question>
                Do you want to keep this comment because it uses Nazi-related terms ("reich") to caution against Nazi-like behaviors rather than to promote Nazism?
            </Question>  
            <Answer>
                Yes.
            </Answer>

            #### OUTPUT
            <Reasoning>
                The user answered "Yes," which means that they want to keep comments that use Nazi-related terms to caution against Nazi-like behaviors.
                However, the original prompt only mentions removing texts that promote Nazism, failing to mention a negative example: Nazi-related terms used to caution against Nazi-like behaviors.
                Therefore, the prompt will be edited to include this preference.
            </Reasoning>
            <ImprovedPrompt>
                Remove texts that promote Nazism. However, do NOT remove:
                    - texts using Nazi-related terms (e.g., "reich") to caution against Nazi-like behaviors
            </ImprovedPrompt>
        """
        user_prompt = f"""
            <Prompt><Rubric>{filter.description}</Rubric></Prompt>
            <Text>{prediction.comment.content}</Text>
            <Groundtruth>{prediction.groundtruth}</Groundtruth>
            <Question>
                {followup['question']}
            </Question>  
            <Answer>
                {followup['answer']}
            </Answer>
        """
        response = self.llm_client.chat_completion(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            type="text"
        )
        print('Refine prompt response: %s', response)
        refined_prompt = self.llm_client.extract_xml(response, "ImprovedPrompt")
        return refined_prompt


        
