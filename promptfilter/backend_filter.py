
import copy
import logging
from .basic_filter import BasicPromptFilter
from . import utils as utils
import os

logger = logging.getLogger(__name__)
RANDOM_PREDICTION = os.getenv('RANDOM_PREDICTION', 'False') == 'True'

class BackendPromptFilter:

    def __init__(self, name, description, **kwargs):
        """
            Initialize a BackendPromptFilter from a PromptFilter object.

            :param description: The description of the filter.
            :param positives: A list of positive rubrics with two keys: "rubric" and "examples".
            :param negatives: A list of negative rubrics with two keys: "rubric" and "examples".
            :param fewShotExamples: A list of examples with two keys: "content" and "groundtruth".
        """
        self.id = kwargs.get('id', None)
        self.name = name
        self.description = description

        positives = kwargs.get('positives', [])
        negatives = kwargs.get('negatives', [])
        self.positives = [self.parse_rubrics(**rubric) for rubric in positives]
        self.negatives = [self.parse_rubrics(**rubric) for rubric in negatives]

        fewShotExamples = kwargs.get('fewShotExamples', [])
        self.few_shots = fewShotExamples

        self.training_examples = kwargs.get('examples', [])
        self.histories = []

        self.channel_id = kwargs.get('channelId', None)

    @classmethod
    def create_backend_filter(cls, promptfilter):
        """
            Create a BackendPromptFilter from a PromptFilter object.
        """
        serialized_prompt_filter = promptfilter.serialize()
        return cls(
            name=serialized_prompt_filter['name'],
            description=serialized_prompt_filter['description'],
            id=serialized_prompt_filter['id'],
            positives=serialized_prompt_filter['positives'],
            negatives=serialized_prompt_filter['negatives'],
            fewShotExamples=serialized_prompt_filter['fewShotExamples'],
            examples=serialized_prompt_filter['examples'],
            channelId=serialized_prompt_filter['channelId'],
        )

    def parse_rubrics(self, rubric, id=None, examples=None):
        return {
            'id': id,
            'rubric': rubric,
            'examples': examples
        }
    
    def search_rubric(self, rubric, kind):
        if kind == 'positive':
            for positive_rubric in self.positives:
                if positive_rubric['rubric'] == rubric:
                    return positive_rubric
        elif kind == 'negative':
            for negative_rubric in self.negatives:
                if negative_rubric['rubric'] == rubric:
                    return negative_rubric
        return None

    def update_rubric(self, new_rubric, kind, comments=None, old_rubric=None):
        # we should keep a history of the filter
        self.histories.append(copy.deepcopy(self))

        comments = utils.clean_comments(comments)
        if old_rubric is not None:
            # we should update an existing rubric
            old_rubric_dict = self.search_rubric(old_rubric, kind)
            if old_rubric_dict is not None:
                old_rubric_dict['rubric'] = new_rubric
                if comments is not None:
                    old_rubric_dict['examples'].extend(comments)
        else:
            if kind == 'positive':
                self.positives.append(self.parse_rubrics(new_rubric, examples=comments))
            elif kind == 'negative':
                self.negatives.append(self.parse_rubrics(new_rubric, examples=comments))

    def stringify_filter(self, structured=False):
        positive_points = ''
        negative_points = ''
        few_shot_examples = ''
        if self.positives:
            if not structured:
                positive_points = 'This includes in particular these comments:\n\n'
                for rubric in self.positives:
                    # this is to ensure that points with newlines are formatted clearly
                    point = '\n\t\t'.join(rubric['rubric'].strip().split('\n'))
                    positive_points += f'\t\t\t- {point}\n'
            else:
                positive_points = ''
                for index, rubric in enumerate(self.positives):
                    positive_points += f"<rubric>{index}. {rubric['rubric']}</rubric>\n"
                positive_points = f"<positives>This includes in particular these comments:\n{positive_points}</positives>\n"
        
        if self.negatives:
            if not structured:
                negative_points = 'However, do not catch the following categories of comments:\n\n'
                for rubric in self.negatives:
                    # this is to ensure that points with newlines are formatted clearly
                    point = '\n\t\t'.join(rubric['rubric'].strip().split('\n'))
                    negative_points += f"\t\t\t- {point}\n"
            else:
                number_of_positives = len(self.positives)
                negative_points = ''
                for index, rubric in enumerate(self.negatives):
                    negative_points += f"<rubric>{number_of_positives + index}. {rubric['rubric']}</rubric>\n"
                negative_points = f"<negatives>However, do not catch the following categories of comments:\n{negative_points}</negatives>\n"

        if self.few_shots:
            few_shot_examples = f'Here are a few examples to illustrate what comments should be caught or not:\n'
            for example in self.few_shots:
                few_shot_examples += f"\t\t- <data>{example['content']}</data><prediction>{'True' if example['groundtruth'] == 1 else 'False'}</prediction>\n"

        if not structured:
            filter_string = f"""
                {self.description}

                {positive_points}

                {negative_points}
                {few_shot_examples}
            """
        else:
            filter_string = f"""
                <description>{self.description}</description>
                {positive_points}

                {negative_points}
            """
        return filter_string
    
    def predict_comments_consistently(self, comments, **kwargs):
        prompt_str = self.stringify_filter(structured=False)
        # logger.info(f'We are running LLMs with debug mode: {RANDOM_PREDICTION}.')
        prompt_filter = BasicPromptFilter(prompt_str, debug=RANDOM_PREDICTION)
        return prompt_filter.predict_comments_consistently(comments, rounds=5)

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'positives': [positive for positive in self.positives],
            'negatives': [negative for negative in self.negatives],
            'fewShotExamples': self.few_shots,
            'examples': self.training_examples,
        }