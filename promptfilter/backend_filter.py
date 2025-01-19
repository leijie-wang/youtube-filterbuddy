
import copy
import logging
from .basic_filter import BasicPromptFilter
from . import utils as utils


logger = logging.getLogger(__name__)

class BackendPromptFilter:
    def __init__(self, description, positives=None, negatives=None, fewShotExamples=None, **kwargs):
        """
            Initialize a BackendPromptFilter from a PromptFilter object.

            :param description: The description of the filter.
            :param positives: A list of positive rubrics with two keys: "rubric" and "examples".
            :param negatives: A list of negative rubrics with two keys: "rubric" and "examples".
            :param fewShotExamples: A list of examples with two keys: "content" and "groundtruth".
        """

        self.description = description
        self.positives = [self.parse_rubrics(**rubric) for rubric in positives or []]
        self.negatives = [self.parse_rubrics(**rubric) for rubric in negatives or []]
        self.few_shots = fewShotExamples
        self.histories = []

    @classmethod
    def create_backend_filter(cls, promptfilter):
        """
            Create a BackendPromptFilter from a PromptFilter object.
        """
        serialized_prompt_filter = promptfilter.serialize()
        return cls(
            description=serialized_prompt_filter['description'],
            positives=serialized_prompt_filter['positives'],
            negatives=serialized_prompt_filter['negatives'],
            fewShotExamples=serialized_prompt_filter['fewShotExamples']
        )

    def parse_rubrics(self, rubric, examples=None):
        return {
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
                self.positives.append(self.parse_rubrics(new_rubric, comments))
            elif kind == 'negative':
                self.negatives.append(self.parse_rubrics(new_rubric, comments))

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
                number_of_positives = len(filter.get('positives', []))
                negative_points = ''
                for index, rubric in enumerate(self.negatives):
                    negative_points += f"<rubric>{number_of_positives + index}. {rubric['rubric']}</rubric>\n"
                negative_points = f"<negatives>However, do not catch the following categories of comments:\n{negative_points}</negatives>\n"

        if self.few_shots:
            few_shot_examples = f'Here are a few examples to illustrate what comments should be caught or not:\n{self.few_shots_str}'
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
        prompt_filter = BasicPromptFilter(prompt_str)
        return prompt_filter.predict_comments_consistently(comments, **kwargs)
