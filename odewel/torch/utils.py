import gc
from typing import List

import torch


def free_memory() -> None:
    """
    Frees up memory in PyTorch and calls the garbage collector.

    Garbage Collection Joke
    -----------------------
    Knock, Knock!
    Who's there?
    Garbage Collection.
    Garbage Collection who?
    Who has time for jokes with unfreed memory?
    """
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    gc.collect()


def prepend_string(
    string: str, list_of_strings: List[str], separator: str = "."
) -> List[str]:
    """
    Prepend a string to each item in a list of strings using a specified separator.

    Parameters
    ----------
    string : str
        The string to be prepended to each item.
    list_of_strings : List[str]
        The list of strings to be modified.
    separator : str, optional
        The separator to be used between `string` and each item in `list_of_strings`, by default "."

    Returns
    -------
    List[str]
        The list of modified strings.
    """
    return [string + separator + item for item in list_of_strings]
