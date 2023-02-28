import importlib
from abc import ABC, abstractmethod
from typing import List

import torch.nn


class IExperiment(ABC):
    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        raise NotImplemented

    @model.setter
    @abstractmethod
    def model(self, model: torch.nn.Module):
        raise NotImplemented

    @property
    @abstractmethod
    def train_loss(self) -> torch.nn.Module:
        raise NotImplemented

    @train_loss.setter
    @abstractmethod
    def train_loss(self, loss: torch.nn.Module):
        raise NotImplemented

    @property
    @abstractmethod
    def val_loss(self) -> torch.nn.Module:
        raise NotImplemented

    @val_loss.setter
    @abstractmethod
    def val_loss(self, loss: torch.nn.Module):
        raise NotImplemented

    @property
    @abstractmethod
    def train_loader(self) -> torch.utils.data.DataLoader:
        raise NotImplemented

    @train_loader.setter
    @abstractmethod
    def train_loader(self, loader: torch.utils.data.DataLoader):
        raise NotImplemented

    @property
    @abstractmethod
    def val_loader(self) -> torch.utils.data.DataLoader:
        raise NotImplemented

    @val_loader.setter
    @abstractmethod
    def val_loader(self, loader: torch.utils.data.DataLoader):
        raise NotImplemented


class Experiment(IExperiment):
    def __init__(self):
        self._model = None
        self._train_loss = None
        self._val_loss = None
        self._train_loader = None
        self._val_loader = None

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model

    @property
    def train_loss(self) -> torch.nn.Module:
        return self._train_loss

    @train_loss.setter
    def train_loss(self, loss: torch.nn.Module):
        self._train_loss = loss

    @property
    def val_loss(self) -> torch.nn.Module:
        return self._val_loss

    @val_loss.setter
    def val_loss(self, loss: torch.nn.Module):
        self._val_loss = loss

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader: torch.utils.data.DataLoader):
        self._train_loader = loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader: torch.utils.data.DataLoader):
        self._val_loader = loader


def get_experiment(experiment_patches: List[str]) -> IExperiment:
    exp = Experiment()

    for patch in experiment_patches:
        mod = importlib.import_module("exps." + patch)
        mod.patch(exp)

    return exp
