from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
import numpy as np
from preprocessamento import Preprocessing
import itertools

class Trainer:
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        ...
    
    @abstractmethod
    def TrainerFit(self, X_train:np.ndarray, Y_train:Optional[np.ndarray]) -> None:
        """Treina
        Args:
            Y_train: provavelmente você nao deve usar esse parametro"""
        ...

    @abstractmethod
    def TrainerPred(self, X:np.ndarray) -> np.ndarray:
        ...

class Evaluator:
    def __init__(self, **kwargs) -> None:
        ...
    def evaluate(self, Y:np.ndarray, Y_pred:np.ndarray, threshold=None, *args, **kwargs) -> tuple[float,dict]:
        assert threshold is None
        #Threshold nao implementado ainda 
        """Avalia as previsões do modelo, retornando uma métrica principal e outras auxiliares"""

        return np.count_nonzero(Y == Y_pred) / len(Y), {"accuracy": np.count_nonzero(Y == Y_pred) / len(Y)}

from abc import ABC, abstractmethod

class BaseHyperParamTuner(ABC):
    def __init__(self,
                 modelTrainerParams: dict[str, list],
                 X_train: np.ndarray,
                 X_val: np.ndarray,
                 X_test: np.ndarray,
                 anom_val: np.ndarray,
                 anom_test: np.ndarray):
        
        self.param_grid = modelTrainerParams
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.anom_val = anom_val
        self.anom_test = anom_test

        self.best_params: dict | None = None
        self.best_score: float = -np.inf
        self.best_model: Trainer | None = None
        self.results: list[dict] = []
        
        self._expandedModelGrid = self._expand_grid(self.param_grid)

    @staticmethod
    def _expand_grid(grid: dict[str, list]) -> list[dict]:
        if not grid:
            return [{}]
        keys = list(grid.keys())
        values = [v if isinstance(v, (list, tuple, np.ndarray)) else [v] for v in grid.values()]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @abstractmethod
    def _get_window_grid(self) -> list[dict]:
        """Define iteration over window parameters (if any)."""
        pass

    @abstractmethod
    def _prepare_data(self, window_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform data based on window params (or return as is)."""
        pass

    def tune(self, Trainer_factory: Callable[..., Trainer], evaluator: Evaluator) -> None:
        # Loop 1: Window Configurations (delegated to subclass)
        window_grid = self._get_window_grid()
        
        for window_params in window_grid:
            # 1. Prepare Data Strategy (Polymorphic step)
            try:
                Xtr, validationX, validationLabels = self._prepare_data(window_params)
            except ValueError as e:
                print(f"Skipping configuration due to error: {e}")
                continue

            # Loop 2: Model Hyperparameters
            for modelParams in self._expandedModelGrid:
                trainer = Trainer_factory(**modelParams)
                
                # Fit
                trainer.TrainerFit(Xtr, None)
                
                # Predict
                anom_val_pred = trainer.TrainerPred(validationX)

                # Evaluate
                score, metrics = evaluator.evaluate(validationLabels, anom_val_pred)

                record = {
                    "score": score,
                    "metrics": metrics,
                    "window_params": window_params,
                    "model_params": modelParams,
                    "tuner_type": self.__class__.__name__
                }
                self.results.append(record)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = record
                    self.best_model = trainer


class DeepLearningTuner(BaseHyperParamTuner):
    def __init__(self, window_params: dict[str, list], **kwargs):
        super().__init__(**kwargs)
        self.window_params = window_params
        self._expandedWindowGrid = self._expand_grid(self.window_params)

    def _get_window_grid(self) -> list[dict]:
        return self._expandedWindowGrid

    def _prepare_data(self, window_params: dict):
        # Dynamic Resizing Logic
        Xtr = Preprocessing.resizeFlattenedWindow(self.X_train, **window_params)
        Xval = Preprocessing.resizeFlattenedWindow(self.X_val, **window_params)
        anomVal = Preprocessing.resizeFlattenedWindow(self.anom_val, **window_params)
        
        validationX = np.concatenate((Xval, anomVal), axis=0)
        validationLabels = np.concatenate((np.ones(Xval.shape[0]), np.zeros(anomVal.shape[0])))
        
        return Xtr, validationX, validationLabels


class StaticFeaturesTuner(BaseHyperParamTuner):
    def __init__(self, **kwargs):
        # We do NOT accept window_params here. 
        # If the user tries to pass them, it goes into **kwargs and we ignore/warn or strict check.
        if 'window_params' in kwargs:
            raise TypeError("StaticFeaturesTuner does not support 'window_params'. Feature Engineering is fixed.")
            
        super().__init__(**kwargs)

    def _get_window_grid(self) -> list[dict]:
        # Return a single empty dict to run the loop exactly once
        return [{}]

    def _prepare_data(self, window_params: dict):
        # No resizing allowed. Return data as is.
        # Logic for validation set construction
        validationX = np.concatenate((self.X_val, self.anom_val), axis=0)
        validationLabels = np.concatenate((np.ones(self.X_val.shape[0]), np.zeros(self.anom_val.shape[0])))
        
        return self.X_train, validationX, validationLabels