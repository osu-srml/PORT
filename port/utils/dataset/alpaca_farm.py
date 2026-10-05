"""
Date: 2026-02-03
Overview: Load AlpacaFarm dataset and convert to preference learning format.
"""
from typing import Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
import tqdm

def get_alpaca_farm(split: str, silent: bool = False, cache_dir: str = None, partial_ratio=0.1, data_path: str = None) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Alpaca Farm human preference dataset and convert to preference learning format.

    This function loads the Alpaca Farm dataset from a local JSON file and processes
    it into a standardized format suitable for preference learning algorithms like DPO/TDPO.
    Each example contains a chosen and rejected response pair for the same prompt.

    Args:
        split (str): Dataset split to load ('train', 'test'). For Alpaca Farm, we'll use
            90% for train and 10% for test by default.
        silent (bool, optional): If True, suppresses progress bars and output.
            Defaults to False.
        cache_dir (str, optional): Directory to cache downloaded dataset (not used for local files).
            Defaults to None.
        partial_ratio (float, optional): Fraction of the dataset to use (0.0 to 1.0).
            Defaults to 0.1 (10%).
        data_path (str, optional): Path to the Alpaca Farm JSON file.
            Defaults to None (uses default path).

    Returns:
        Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
            Nested dictionary with the following structure:
            {
                'prompt1': {
                    'responses': List[str] - [chosen_response, rejected_response],
                    'pairs': List[Tuple[int, int]] - [(0, 1)] indicating preference,
                    'sft_target': str - The chosen response for SFT training
                },
                'prompt2': { ... },
                ...
            }

            Prompts are formatted as: "\n\nHuman: <instruction> <input>\n\nAssistant:"
            The instruction and input fields are combined to form the prompt.

    Note:
        - The Alpaca Farm dataset contains instruction, input, output_1, output_2, and preference fields
        - Preference field: 1 means output_1 is preferred, 2 means output_2 is preferred
        - The dataset is split 90/10 for train/test using a fixed seed for reproducibility
    """
    import json

    # Default data path if not provided
    if data_path is None:
        raise ValueError("data_path must be provided. Please specify the path to the alpaca_human_preference.json file.")

    # (f'Loading Alpaca Farm dataset ({split} split) from {data_path}...')

    # Load the JSON file
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    # Split the dataset 90/10 for train/test
    total_size = len(dataset)
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(total_size)

    test_size = int(total_size * 0.1)
    if split == 'test':
        split_indices = indices[:test_size]
    else:  # train
        split_indices = indices[test_size:]

    dataset = [dataset[i] for i in split_indices]

    # Apply partial ratio selection
    if partial_ratio < 1.0:
        split_total_size = len(dataset)
        partial_size = int(split_total_size * partial_ratio)
        dataset = dataset[:partial_size]
        print(f'Using partial Alpaca Farm dataset: {partial_size}/{split_total_size} ({partial_ratio:.2%})')

    print(f'Loaded {len(dataset)} examples')

    # Initialize nested dictionary to group responses by prompt
    data = defaultdict(lambda: defaultdict(list))

    # Process each preference pair in the dataset
    for row in tqdm.tqdm(dataset, desc='Processing Alpaca Farm', disable=silent):
        # Construct prompt from instruction and input
        # Combine instruction and input (if input is not empty)
        instruction = row['instruction']
        input_text = row['input']

        if input_text.strip():
            prompt_text = f"{instruction}\n{input_text}"
        else:
            prompt_text = instruction

        # Format prompt in Anthropic style
        prompt = f"\n\nHuman: {prompt_text}\n\nAssistant:"

        # Get the two outputs
        output_1 = ' ' + row['output_1']  # Add leading space like in HH dataset
        output_2 = ' ' + row['output_2']

        # Determine which output is preferred
        # preference field: 1 means output_1 is preferred, 2 means output_2 is preferred
        preference = row['preference']

        if preference == 1:
            chosen = output_1
            rejected = output_2
        elif preference == 2:
            chosen = output_2
            rejected = output_1
        else:
            # Skip examples with invalid preference values
            continue

        # Create response list with chosen first (index 0), rejected second (index 1)
        responses = [chosen, rejected]

        # Track current number of responses for this prompt (for indexing)
        n_responses = len(data[prompt]['responses'])

        # Add preference pair: (0, 1) means response 0 preferred over response 1
        data[prompt]['pairs'].append((n_responses, n_responses + 1))

        # Add both responses to the prompt's response list
        data[prompt]['responses'].extend(responses)

        # Set the chosen response as the target for supervised fine-tuning
        data[prompt]['sft_target'] = chosen

    return data
