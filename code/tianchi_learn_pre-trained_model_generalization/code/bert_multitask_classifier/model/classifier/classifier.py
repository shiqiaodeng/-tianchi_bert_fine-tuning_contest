import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from enum import Enum

from ...common import logger
from dataclasses import dataclass

class ClassifierType(Enum):
    ATTENTION = 1

@dataclass
class ClassifierTaskArgs:
    task_name: str
    classifier_type: ClassifierType
    hidden_size: int
    num_classes: int

class MultiTaskClassifier(nn.Module):
    def __init__(self, args_list: list):
        super().__init__()
        self.__classifiers = nn.ModuleDict({args.task_name:
                                          BaseClassifier.create_object(args.classifier_type, args.hidden_size, args.num_classes)
                                          for args in args_list})
    
    def forward(self, task_name: str, hidden_states: torch.Tensor, mask: torch.Tensor):
        if task_name in self.__classifiers:
            return self.__classifiers[task_name](hidden_states, mask)
        else:
            raise ValueError(f"Task ID {task_name} not found in __classifiers.")

    def get_num_classes(self, task_name: str):
        if task_name in self.__classifiers:
            return self.__classifiers[task_name].get_num_classes()
        else:
            raise ValueError(f"Task ID {task_name} not found in __classifiers.")

class BaseClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def get_num_classes(self):
        return self.num_classes

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        raise ValueError("BaseClassifier should not be used directly. Use a subclass that implements the forward method.")

    @classmethod
    def create_object(cls, type: ClassifierType, hidden_size: int, num_classes: int):
        if type == ClassifierType.ATTENTION:
            return AttentionClassifier(hidden_size, num_classes)
        else:
            raise ValueError("Not supported classifier type: {}".format(type))

class AttentionClassifier(BaseClassifier):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__(hidden_size, num_classes)
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        h = self.attn(hidden_states, mask)
        out = self.fc(h)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor):
        q = self.fc(hidden_state).squeeze(dim=-1)
        q = q.masked_fill(mask, -np.inf)
        w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        h = w @ hidden_state
        return h.squeeze(dim=1)