"""
Load Anthropic Helpful-Harmless (HH) dataset and convert to preference learning format.
"""
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import datasets
import random
import tqdm

from .prompt import (
    extract_anthropic_prompt,
    split_prompt_and_responses,
)

def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt1': {
            'responses': List[str],
            'pairs': List[Tuple[int, int]],
            'sft_target': str
        },
        ...
    }

    Prompts are structured as follows:
      \\n\\nHuman: <prompt>\\n\\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \\n\\nHuman: and end with \\n\\nAssistant:.

    For this dataset, the sft_target is just the chosen response.

    Args:
        split (str): 'train' or 'test'
        silent (bool): Whether to silence the progress bar.
        cache_dir (str): Directory to cache the dataset in.

    Returns:
        Dict[str, Dict]: The dataset keyed by prompt string.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_hh_pairwise_noise(split: str, silent: bool = False, cache_dir: str = None, noise_rate=0., partial_ratio=1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic HH dataset with random pairwise label noise.

    Args:
        split (str): 'train' or 'test'
        silent (bool): Whether to silence the progress bar.
        cache_dir (str): Directory to cache the dataset in.
        noise_rate (float): Fraction of training pairs to randomly flip (chosen <-> rejected).
        partial_ratio (float): Fraction of the dataset to use (1.0 = full dataset).

    Returns:
        Dict[str, Dict]: The dataset keyed by prompt string.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    if partial_ratio < 1.0:
        total_size = len(dataset)
        partial_size = int(total_size * partial_ratio)
        dataset = dataset.shuffle(seed=42).select(range(partial_size))
        print(f'Using partial dataset: {partial_size}/{total_size} ({partial_ratio:.2%})')

    data = defaultdict(lambda: defaultdict(list))
    cnt_noise = 0
    for row in tqdm.tqdm(dataset, desc='Processing HH noise={} partial={:.2%}'.format(noise_rate, partial_ratio), disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)

        if split == 'train' and random.random() < noise_rate:
            responses = [rejected, chosen]
            sft_target = rejected
            cnt_noise += 1
        else:
            responses = [chosen, rejected]
            sft_target = chosen

        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = sft_target

    print("{} noise_rate={:.2%}, partial_ratio={:.2%}, total={}, noisy={}".format(
        split, cnt_noise / len(dataset), partial_ratio, len(dataset), cnt_noise))
    return data
