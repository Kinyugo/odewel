import gc
import json
import os
from typing import Dict, List, Union

import torch
from torch import Tensor


class ShardedWeightsLoader:
    """
    Class for loading sharded weights.

    Parameters
    ----------
    index_file_path : Union[str, os.PathLike]
        The path to the index file that maps each weight to its weight file.
    weights_dir : Union[str, os.PathLike]
        The directory where the weight files are stored.
    weights_mapping_key : str
        Key in the index file that maps each weight to its weight file.
    device : torch.device
        The torch device on which to place the loaded weights.
    dtype : torch.dtype
        The dtype of the tensors to be loaded.


    Raises
    ------
    ValueError
        If the path for the index file or weight directory is invalid.

    """

    def __init__(
        self,
        index_file_path: Union[str, os.PathLike],
        weights_dir: Union[str, os.PathLike],
        weights_mapping_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not os.path.isfile(index_file_path):
            raise ValueError(f"Invalid path for index file: {index_file_path}")
        if not os.path.isdir(weights_dir):
            raise ValueError(f"Invalid path for weight directory: {weights_dir}")

        self.weights_dir = weights_dir
        self.device = device
        self.dtype = dtype

        with open(index_file_path, "r", encoding="utf-8") as f:
            self.weights_mapping = json.load(f)[weights_mapping_key]

    def __call__(self, module_name: str, weight_names: List[str]) -> Dict[str, Tensor]:
        """
        Loads weights for a specific module.

        Parameters
        ----------
        module_name : str
            The name of the module to be loaded.
        weight_names : List[str]
            The names of the weights for the specified module.

        Returns
        -------
        Dict[str, Tensor]
            A dictionary mapping weight names to their corresponding tensors.

        """
        # Find all the weights that contain the weights for the current module
        ckpt_names = list(
            set(
                [
                    self.weights_mapping[weight_name]
                    for weight_name in weight_names
                    if weight_name in self.weights_mapping.keys()
                ]
            )
        )

        # Keep track of the weights dict in one place
        weights_dict = {}
        for ckpt_name in ckpt_names:
            ckpt_path = os.path.join(self.weights_dir, ckpt_name)
            # We will move the weights to the correct device later
            curr_weights_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
            weights_dict.update(curr_weights_dict)

            del curr_weights_dict
            gc.collect()

        # Move the weights dict to the correct device
        if weights_dict:
            for weight_name, weight in weights_dict.items():
                weights_dict[weight_name] = weight.to(
                    device=self.device, dtype=self.dtype
                )

        return weights_dict
