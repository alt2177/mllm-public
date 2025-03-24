
import torch
import torch.nn as nn
from typing import Type, TypeVar, List
import re


NAME: str = "merger"
VALID_MERGE_METHODS: List[str] = [
    "linear",
    "ties",
    "dare-ties"
]

def check_for_additional_models(**kwargs) -> List[nn.Module]:
    """
    Function to detect additional models being passed into constructor
    :param **kwargs: any additional arguments
    :return additional_models: list of 
    """

    additional_models: List[nn.Module] = []
    pattern: re.Pattern[str] = re.compile(r"^model_\d+$")
    for key in kwargs:
        if pattern.match(key):
            additional_models.append(kwargs[key])

    return additional_models


class Merger():


    def __init__(self, model_1: nn.Module, model_2: nn.Module, merge_method: str, *args, **kwargs) -> None:
        self.merge_method: str = merge_method
        self.name: str = NAME
        self.models: List[nn.Module] = [model_1, model_2]

        if kwargs:
            additional_models: List[nn.Module] = check_for_additional_models(**kwargs) 
            self.models += additional_models


    def __str__(self) -> str:
        return f"Merger('name={self.name}')"





class LinearMerger(Merger):

    def __init__(self, merge_method: str) -> None:
        super().__init__(merge_method)


    def __str__(self) -> str:
        return f"LinearMerge(name={self.merge_method})"


    def linear_merge(self):
        raise NotImplementedError


    def merge(self, merge_method: str):
        return self.linear_merge()


class MergeMethod(Merger):

    def __init__(self, merge_method: str) -> None:
        super().__init__(merge_method)

    def __str__(self) -> str:
        return f"MergeMethod(name={self.merge_method})"


class TiesMerger(Merger):

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError 
