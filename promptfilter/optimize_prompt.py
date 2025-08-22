from .chat_completion import ChatCompletion
from .models import FilterPrediction, Comment
import time
import logging
import random
from tqdm import tqdm
import copy
from .llm_buddy import LLMBuddy
logger = logging.getLogger(__name__)
class PromptOptimizer:
    
    def __init__(self):
        self.llm_client = ChatCompletion()
        self.buddy = LLMBuddy()
        self.opt = {
            'minibatch_size': 20,
            'errors_per_gradient': 4,
            'n_gradients': 4,
            'gradients_per_error': 1,
            'steps_per_gradient': 1,
            'mc_samples_per_step': 1,
            'max_expansion_factor': 4,
            'reject_on_errors': False,
            'beam_size': 2,
        }

    def _sample_error_str(self, texts, labels, preds, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ''

        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Text: \"{t.strip()}\"\nLabel: {l}\nPrediction: {p}\n\n'
            error_idx += 1
        return error_string.strip()
    
    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts
    
    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
            I'm trying to write a zero-shot classifier prompt.
        
            My current prompt is:
            <prompt>"{prompt.stringify_filter(structured=False)}"</prompt>

            But this prompt gets the following examples wrong:
            <error_string>{error_string}</error_string>

            give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
            Wrap each reason with <START> and <END> tags.
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        res = self.llm_client.chat_completion(
            system_prompt = gradient_prompt,
            user_prompt="",
            type="text"
        )
        logger.debug(f'response from LLM: {res}')
        feedbacks = []
        feedbacks += self.parse_tagged_text(res, "<START>", "<END>")
        return feedbacks
    
    def get_gradients(self, prompt, texts, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(
                prompt, error_string, self.opt['gradients_per_error'], n=1)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks
    
    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
            I'm trying to write a zero-shot classifier.
            
            My current prompt is:
            <prompt>"{prompt.stringify_filter(structured=False)}"</prompt>

            But it gets the following examples wrong:
            <error_string>{error_str}</error_string>

            Based on these examples the problem with this prompt is that <feedback_str>{feedback_str}</feedback_str>

            Based on the above information, I wrote {steps_per_gradient} different improved prompts.
            Each prompt is wrapped with <START> and <END>.

            The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = self.llm_client.chat_completion(
            system_prompt = transformation_prompt,
            user_prompt="",
            type="text"
        )
        logger.debug(f'response from LLM: {res}')
        new_prompts = []
        new_prompts += self.parse_tagged_text(res, "<START>", "<END>")
        return new_prompts
    
    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = self.llm_client.chat_completion(
            system_prompt=rewriter_prompt,
            user_prompt="",
            type="text",
            n=n
        )

        new_instructions = [x for x in new_instructions if x]
        return new_instructions
    
    def expand_candidates(self, prompts, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):

            # evaluate prompt on minibatch
            # _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)
            train_exs_copy = [ex.copy() for ex in train_exs]
            train_exs_copy = prompt.predict_comments_consistently(minibatch, prompt.attributes.get('predictions', {}))
            prompt.cache_predictions(train_exs_copy)
            texts = [ex['content'] for ex in train_exs_copy]
            labels = [ex['groundtruth'] for ex in train_exs_copy]
            preds = [ex['prediction'] for ex in train_exs_copy]
            ids = [ex['id'] for ex in train_exs_copy]

            # get gradients
            task_section = prompt.description
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, texts, labels, preds)
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        prompt, error_string, feedback, self.opt['steps_per_gradient'])
                    new_task_sections += tmp

             # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            
            tmp_new_prompts = []
            for sect in new_sections:
                new_prompt = copy.deepcopy(prompt)
                new_prompt.clear_caches()
                new_prompt.description = sect
                tmp_new_prompts.append(new_prompt)
            
            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p, id) in enumerate(zip(texts, labels, preds, ids)):
                        if l != p:
                            error_exs.append({'content': t, 'groundtruth': l, 'id': id})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))
                    tmp_new_prompts = []
                    tmp_new_prompts = self.buddy.__overall_select_best_filters(tmp_new_prompts, error_exs, topN=self.opt['max_expansion_factor'])
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, 
                        k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        # dedup prompts
        deduped = list({
            (getattr(o, "description", None) or "").strip().lower(): o
            for o in reversed(new_prompts)  # reverse to keep the last one
        }.values())[::-1]

        return deduped

    def calibrate_prompt(self, filter, train_exs, rounds=3, beam_size=2):
        self.opt['beam_size'] = beam_size
        for train_ex in train_exs:
            train_ex['weight'] = 1
        
        candidates = [filter]
        for round in tqdm(range(rounds + 1)):
            # expand candidates
            start_time = time.time()
            candidates = self.expand_candidates(candidates, train_exs)

            # choose the top candidates
            candidates = self.buddy.select_best_filters(candidates, train_exs, topN=self.opt['beam_size'])
            logger.info(f'Round {round + 1} of calibration: {len(candidates)} candidates after expansion')
            logger.info(f'It takes {time.time() - start_time} seconds to expand candidates')
            logger.info('=' * 150)
        return candidates[0]
