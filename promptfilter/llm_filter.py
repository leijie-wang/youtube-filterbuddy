import enum
from django.conf import settings
import logging
import random
import json
import threading
import time
from . import utils

logger = logging.getLogger(__name__)

class LLMFilter:

    def __init__(self, prompt, batch_size=10, debug=False, retry=False):
        """
            debug: if debug is True, we will not prompt the model to get predictions
        """
        self.debug = debug 
        self.prompt = prompt
        self.BATCH_SIZE = batch_size
        self.retry = retry
        self.system_prompt = f"""
            For each text in the dataset, give a 1 (True) or 0 (False) prediction to represent whether the text satisfies the description in the rubrics. 
            Each text starts with “DATA” and a number. Both the number and the text are enclosed by “<” and “>”. 
            
            In the following, the user will provide one rubric to help you make your decision. 
            If the given rubric is completely satisfied, give a True prediction. Otherwise, give a False prediction. 
            RETURN YOUR ANSWER in the json format {{“results”: [(index, prediction), ...]}} where (index, prediction) is a tuple, index is the number of the text in the dataset, and prediction is either 1 or 0.
        """
        self.llm_client = utils.ChatCompletion()
    
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
            response = self.llm_client.chat_completion(self.system_prompt, now_prompt)
            response = json.loads(response or "{}")
            if "results" in response:
                results = response["results"]
                batch_preds = []
                try:
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
                    logger.warning(f"reading individual predictions from results {results} raise an error: {e}")
            else:
                logger.warning(f"batch_response does not have 'results' key: {response}")

            if len(batch_preds) != len(batch):
                logger.warning(f"response length {len(results)} does not match batch length {len(batch)}\nresults: {results}")
            
            if self.retry and len(batch_preds) != len(batch):
                logger.warning(f"retrying to prompt the model with the batch again")
                continue
            elif len(batch) > 0 and len(batch_preds) / len(batch) < 0.9:
                logger.warning(f"response length {len(batch_preds)} is less than half of the batch length {len(batch)}")
                continue
            else:
                predictions += batch_preds
                return predictions
    
    def _test_prompt(self, prompt, dataset_list):
        user_prompt = f"""
            ### RUBRIC
            Rubric: <{prompt['description']}>
        """
        logger.debug(f"from the filter prompt: {user_prompt}")

        predictions = []
        threads = []
        if not self.debug:
            for index in range(0, len(dataset_list), self.BATCH_SIZE):
                batch = dataset_list[index: index + self.BATCH_SIZE]
                logger.debug(f"now predicting batch from index {index} to {len(batch) + index}")
                thread = threading.Thread(target=self._handle_batch_prompt, args=(batch, predictions, user_prompt))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()  
        else:
            # generate a random number either 0 or 1
            predictions = [(index, random.randint(0, 1)) for index in range(len(dataset_list))]

            
                
        if len(predictions) != len(dataset_list):
            logger.warning(f"response length {len(predictions)} does not match dataset length {len(dataset_list)}")
        
        # sort predictions by the first element in each tuple
        predictions = sorted(predictions, key=lambda x: x[0])
        for index, item in enumerate(predictions):
            if index != item[0]:
                logger.warning(f"item at the index {index} has the index {item[0]}")
            else:
                predictions[index] = item[1]
    
        return predictions

    def predict(self, X):
        """
            There is no training stage for LLM. We only test the model against X, y
            As different prompts might lead to different actions, we test the model against each prompt separately.

            Besides, it might also be possible that feeding the model with all data might overwhelm the model and lead to some idiosyncratic behavior.
            Therefore, we also feed the model with a small batch of data each time.

            @param X: a list of texts
        """        
        dataset_list = self._generate_dataset_list(X)

        start_time = time.time()
        predictions = self._test_prompt(self.prompt, dataset_list)
        
        end_time = time.time()
        logger.info(f"LLM model testing time for the prompt {self.prompt['name']}: {end_time - start_time} seconds")
        
        return predictions