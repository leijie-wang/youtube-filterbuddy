from django.conf import settings
from collections import defaultdict
import logging
import random
import json
import threading
import time
from .chat_completion import ChatCompletion

logger = logging.getLogger(__name__)

class BasicPromptFilter:

    def __init__(self, prompt, debug=False, batch_size=10, COT=False):
        """
            debug: if debug is True, we will not prompt the model to get predictions
            prompt: we expect a string prompt 
        """
        # print(f'LLM filter at debug mode: {debug}; batch size: {batch_size}; chain of thought: {COT}')
        self.debug = debug 
        self.prompt = prompt
        self.BATCH_SIZE = batch_size
        self.chain_of_thought = COT
        self.system_prompt = f"""
            For each text in the dataset, give a 1 (True) or 0 (False) prediction to represent whether the text satisfies the description in the rubrics. 
            Each text starts with “DATA” and a number. Both the number and the text are enclosed by “<” and “>”. 
            
            In the following, the user will provide one rubric to help you make your decision. 
            If the given rubric is completely satisfied, give a True prediction. Otherwise, give a False prediction. 
            RETURN YOUR ANSWER in the json format {{“results”: [(index, prediction), ...]}} where (index, prediction) is a tuple, index is the number of the text in the dataset, and prediction is either 1 or 0.
        """
        self.llm_client = ChatCompletion()
    
    def _generate_dataset_list(self, dataset):
        """
            Concatenate datasets in the format of 1. data1\n2.data2\n escape the double quotes for each text
        """
        dataset_list = []
        for index in range(len(dataset)):
            text = dataset[index] # TODO: ensure that we remove the double quotes in the comments.
            dataset_list.append(f'DATA<{index}>: <{text}>')    
        return dataset_list
    
    def _handle_batch_prompt(self, batch, predictions, user_prompt):
        """
            @param batch: a list of texts
            @param predictions: a list of predictions
            @param user_prompt: the prompt that will be used to prompt the model
        """
        batch_str = "\n".join(batch)
        now_prompt = user_prompt + f"""\n\n\t### DATASETS: "{batch_str}","""
        while True:
            try:
                response = self.llm_client.chat_completion(self.system_prompt, now_prompt)

                response = json.loads(response or "{}")
                results = response.get("results", [])
                if not (type(results) == list and len(results) > 0):
                    continue


                batch_preds = []
                if type(results[0]) == dict and "index" in results[0] and "prediction" in results[0]:
                    for item in results:
                        batch_preds.append((
                                int(item["index"]), 
                                int(item["prediction"])
                            ))
                elif type(results[0]) == dict and len(results[0]) == 1:
                    for item in results:
                        (index, prediction), = item.items()
                        batch_preds.append((
                            int(index), 
                            int(prediction)
                        ))
                elif type(results[0]) == list and len(results[0]) == 2:
                    for item in results:
                        batch_preds.append((
                            int(item[0]), 
                            int(item[1])
                        ))
                elif type(results[0]) == tuple and len(results[0]) == 2:
                    for item in results:
                        batch_preds.append(
                            int(item[0]),
                            int(item[1])
                        )
                else:
                    logger.warning(f"response {results} is not in any expected format")
            except Exception as e:
                logger.error(f"Error occurred when prompting LLMs: {e}")
                continue


            if len(batch_preds) != len(batch):
                logger.warning(f"response length {len(results)} does not match batch length {len(batch)}\nresults: {results}")
                continue
            else:
                predictions += batch_preds
                return predictions
    
    def _test_prompt(self, prompt, dataset_list):
        
        user_prompt = f"""
            ### RUBRIC
            <Rubric>
                {prompt}
            </Rubric>
        """

        logger.debug(f"from the filter prompt: {user_prompt}")

        predictions = []
        threads = []
        max_threads = 10
        if not self.debug:
            for index in range(0, len(dataset_list), self.BATCH_SIZE):
                batch = dataset_list[index: index + self.BATCH_SIZE]
                while len(threads) >= max_threads:
                    for thread in threads:
                        if not thread.is_alive():
                            threads.remove(thread)
                
                logger.debug(f"now predicting batch from index {index} to {len(batch) + index}")
                thread = threading.Thread(target=self._handle_batch_prompt, args=(batch, predictions, user_prompt))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()  
        else:
            # generate a random number either 0 or 1
            predictions = [(index, random.randint(0, 1)) for index in range(len(dataset_list))]

        # sort predictions by the first element in each tuple
        predictions = sorted(predictions, key=lambda x: x[0])
        for index, item in enumerate(predictions):
            if index != item[0]:
                logger.warning(f"item at the index {index} has the index {item[0]}")
            else:
                predictions[index] = item[1]
    
        return predictions

    def _test_prompt_COT(self, prompt, dataset):
        chain_of_thought_prompt = f"""
            You are an excellent content moderation. You are asked to reason and determine whether the given comment satisfies the given rubric in the following steps.
            In particular, the rubric could consist of two components: the description of what comments match the rubric [PositiveRubric] and what comments should not match the rubric [NegativeRubric].
            If the comment satisfies the rubric, give a 1 (True) prediction. Otherwise, give a 0 (False) prediction.
            <step1>
                For each bullet point in the positive rubric, examine whether the given comment satisfies the positive point or not.
                You should first repeat the bullet point and then provide your reasoning, followed by a Yes, No, or Somewhat answer.
                Your reasoning should solely be based on this particular point. 
            </step1>
            <step2> For each bullet point in the negative rubric, examine whether the given comment satisfies the negative point or not.</step2>
            <step3> 
                Based on your reasoning in step 1 and step 2, make a final decision on whether the comment satisfies the rubric or not.
                In cases where the comment satisfies both some positive and negative rubrics, prioritize rubrics that are more specific in your decision.
            </step3>

            <Examples>
                <Example>
                    <Rubric>
                        <PositiveRubrics>Comments that directly use toxic language to insult someone.</PositiveRubrics>
                        <NegativeRubrics>
                            However, do not catch the following categories of comments:
                                - Comments that do not insult a specific person but rather a generic group of people.
                        </NegativeRubrics>
                    </Rubric>
                    <Comment>This video reminds me of how disgusting and pathetic some idiots on this platform can be.</Comment>
                    <PositiveReasoning>
                        <point1>
                            Comments that directly use toxic language to insult someone.
                            Reasoning: The comment uses toxic language ("disgusting," "pathetic," "idiots") to insult users on the platform.
                            Answer: Yes
                        </point1>
                    </PositiveReasoning>
                    <NegativeReasoning>
                        <point1>
                            Comments that do not insult a specific person but rather a generic group of people.
                            Reasoning: The comment targets a generic group ("some idiots on this platform") rather than a specific person, which aligns with this negative rubric.
                            Answer: Yes
                        </point1>
                    </NegativeReasoning>
                    <Reasoning>
                        The comment satisfies the negative rubric but does not satisfy the positive rubric.
                        Since the negative rubric explicitly excludes such comments, the comment does not match the rubric. The prediction is 0 (False).
                    </Reasoning>
                    <Prediction>0</Prediction>
                </Example>
                <Example>
                    <Rubric>
                        <PositiveRubrics>
                            Comments that contain discriminatory or prejudiced language against genders.

                            This includes in particular these comments:
                                - Comments that completely attribute a female politician's success to their appearances or sex life
                                e.g., "She only got elected because she's hot."
                        </PositiveRubrics>
                        <NegativeRubrics>
                            However, do not catch the following categories of comments:
                                - Comments that describe facts or statistics related to gender;
                                    e.g., ""Men are more likely to pursue engineering degrees than women."
                                - Comments that discuss gender-related political issues, even though they might be against gender equality;
                                    e.g., "Affirmative action for women in STEM is unfair because it gives them advantages over men who might actually be better qualified."
                        </NegativeRubrics>
                    </Rubric>
                    <Comment>Women are biologically less suited for leadership because they’re more emotional and can’t handle stress as well as men.</Comment>

                    <PositiveReasoning>
                        <point1>
                            Comments that contain discriminatory or prejudiced language against genders.
                            Reasoning: The comment makes a discriminatory statement against women, suggesting they are inherently less suited for leadership due to being "more emotional" and unable to "handle stress."
                            Answer: Yes
                        </point1>
                        <point2>
                            Comments that completely attribute a female politician's success to their appearances or sex life
                            Reasoning: The comment is not relevant to this point.
                            Answer: No
                        </point2>
                    </PositiveReasoning>
                    <NegativeReasoning>
                        <point1>
                            Comments that describe facts or statistics related to gender.
                            Reasoning: This statement may appear to draw on facts; on the other hand, it is not framed constructively or neutrally, and instead directly demeans women as a group.
                            Answer: Somewhat
                        </point1>
                        <point2>
                            Comments that discuss gender-related political issues, even though they might be against gender equality.
                            Reasoning: The comment does not address any political issues or policies related to gender. It is solely an expression of prejudice against women in leadership.
                            Answer: No
                        </point2>
                    </NegativeReasoning>
                    <Reasoning>
                        While this comment might be seen as a statement of fact, it is not framed neutrally or constructively and instead directly discriminates against women in leadership.
                    </Reasoning>
                    <Prediction>1</Prediction>
                </Example>
            </Examples>

            Write your output strictly in the following XML format:
            <PositiveReasoning>Your reasoning here.</PositiveReasoning>
            <NegativeReasoning>Your reasoning here.</NegativeReasoning>
            <Reasoning>Your reasoning here.</Reasoning>
            <Prediction>Your prediction here, 1 if the comment matches the rubric; otherwise 0.</Prediction>
        """
        predictions = []
        for comment in dataset:
            if len(predictions) % self.BATCH_SIZE == 0:
                logger.error(f"now predicting the comment at index {len(predictions)}")
            while True:
                try:
                    user_prompt = f"""
                        {prompt}
                        <Comment>{comment}</Comment>
                    """
                    response = self.llm_client.chat_completion(
                        system_prompt = chain_of_thought_prompt,
                        user_prompt = user_prompt,
                        type="text"
                    )
                    prediction = self.llm_client.extract_xml(response, "Prediction")
                    logger.error(f"Prediction: {prediction}\nComment:{comment}\nReasoning: {response}\n")
                    prediction = int(prediction)
                    predictions.append(prediction)
                    break
                except Exception as e:
                    logger.error(f"Error occurred when prompting LLMs: {e}")
                    continue
        if len(predictions) != len(dataset):
            logger.error(f"Number of predictions {len(predictions)} does not match the number of comments {len(dataset)}")
        return predictions

    def predict(self, X):
        """
            There is no training stage for LLM. We only test the model against X, y
            As different prompts might lead to different actions, we test the model against each prompt separately.

            Besides, it might also be possible that feeding the model with all data might overwhelm the model and lead to some idiosyncratic behavior.
            Therefore, we also feed the model with a small batch of data each time.

            @param X: a list of texts

            @return: a list of predictions with 0/1 for each text
        """

        start_time = time.time()
        
        retries = 3
        predictions = None
        while True and retries > 0:
            try:
                if not self.chain_of_thought:
                    dataset_list = self._generate_dataset_list(X)
                    predictions = self._test_prompt(self.prompt, dataset_list)
                else:
                    predictions = self._test_prompt_COT(self.prompt, X)
                break
            except Exception as e:
                logger.error(f"Error occurred in classifying comments: {e}")
                retries -= 1
        end_time = time.time()
        if predictions is not None:
            logger.info(f"LLM model testing time for the prompt on {len(X)}: {end_time - start_time} seconds\n")
            return predictions
        else:
            logger.error(f'LLM model testing failed for the prompt on {len(X)}')
            return [0] * len(X)
        
    def predict_comments(self, comments, **kwargs):
        """Predict the comments using the filter.
        
        Args:
            filter (dict): The filter information.
            comments (list): The list of serialized comments to predict.

        Returns:
            dict: A dictionary with comment IDs as keys and predictions as values.
        """

        datasets = [comment['content'] for comment in comments]
        predictions = self.predict(datasets)    
        predictions = [pred == 1 for pred in predictions]

        # summarize the predictions
        positive_num = sum(predictions)
        negative_num = len(predictions) - positive_num
        logger.debug(f'There are {positive_num} positive predictions and {negative_num} negative predictions.')

        results = {}
        for index, comment in enumerate(comments):
            results[comment['id']] = {
                'prediction': predictions[index],
                'confidence': 1
            }
        return results

    def predict_with_majority_vote(self, comments, rounds=5, randomized=True, batch_size=5):
        """
            Return the majority vote prediction for each comment, and the set of inconsistent comments.

            In these cases, we simply calculate the predictions in batches without chains of thought.

            @param comments: a list of serialized comments to predict.
            @param rounds: the number of rounds to predict
            @param randomized: whether to randomize the order of comments in each round
            @param batch_size: the batch size to predict

            @return: a dictionary with comment IDs as keys and predictions as values.
        """
        
        predictions_across_rounds = defaultdict(list)
        logger.info(f'Predicting comments using majority votes: {rounds} rounds x batch size {batch_size}.')
        for _ in range(rounds):
            if randomized:
                now_comments = random.sample(comments, len(comments))
            else:
                now_comments = comments[:]
            round_predictions = self.predict_comments(now_comments, batch_size=batch_size)
            for comment_id, pred in round_predictions.items():
                predictions_across_rounds[comment_id].append(pred['prediction'])
        
        # calculate the majority prediction as the final prediction
        results = {}
        for comment_id, preds in predictions_across_rounds.items():
            majority_prediction = max(set(preds), key=preds.count)
            confidence = preds.count(majority_prediction) / len(preds)
            results[comment_id] = {
                'prediction': majority_prediction,
                'confidence': confidence
            }
        return results
        
    def predict_comments_consistently(self, comments, **kwargs):
        """
            Predict the comments using the filter.

            @param comments: a list of serialized comments to predict.

            @return: a list of serialized comments with predictions and confidence scores.
        """
        
        start_time = time.time()
        predictions = self.predict_with_majority_vote(comments, **kwargs)
        end_time = time.time()
        logger.info(f'Predictions for the first round completed in {end_time - start_time:.2f} seconds.\n')


        for comment in comments:
            comment_pred = predictions[comment['id']]
            comment['prediction'] = comment_pred['prediction']
            comment['confidence'] = comment_pred['confidence']

        inconsistent_comments = [comment for comment in comments if comment['confidence'] < 1]
        logger.debug(f'There are {len(inconsistent_comments)} inconsistent comments.')
        
        # report how many positive and negative predictions we have
        positive_num = sum(1 for comment in comments if comment['prediction'])
        positive_inconsistent = sum(1 for comment in comments if comment['prediction'] and comment['confidence'] < 1)
        negative_num = len(comments) - positive_num
        negative_inconsistent = sum(1 for comment in comments if not comment['prediction'] and comment['confidence'] < 1)
        logger.debug(f'There are {positive_num} positive predictions and among them {positive_inconsistent} are inconsistent.')
        logger.debug(f'There are {negative_num} negative predictions and among them {negative_inconsistent} are inconsistent.')

        return comments