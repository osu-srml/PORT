"""
Load various preference datasets and convert to preference learning format.
"""
import logging

from .alpaca_farm import get_alpaca_farm
from .anthropic_hh import (
    get_hh,
    get_hh_pairwise_noise,
)
from .stackexchange import get_stackexchange
from .stanford_human_preference import get_shp


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, noise_rate=0., partial_ratio=1.0):
    """Load the given dataset by name. Supported are 'shp', 'hh', 'hh_pairwise_noise', 'se', and 'alpaca_farm'.

    Args:
        name: Name of the dataset to load.
        split: Data split to load ('train', 'validation', 'test').
        silent: If True, suppresses loading messages.
        cache_dir: Directory to cache datasets.
        noise_rate: Noise rate for datasets that support noise injection.
        partial_ratio: Fraction of the dataset to use (1.0 = full dataset).
    """
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh_pairwise_noise':
        data = get_hh_pairwise_noise(split, silent=silent, cache_dir=cache_dir, noise_rate=noise_rate, partial_ratio=partial_ratio)
    elif name == 'se' or name == 'stackexchange':
        data = get_stackexchange(split, silent=silent, cache_dir=cache_dir)
    elif name == 'alpaca_farm':
        data = get_alpaca_farm(split, silent=silent, cache_dir=cache_dir, partial_ratio=partial_ratio)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data
