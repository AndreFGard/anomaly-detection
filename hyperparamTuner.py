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

    def evaluate(self, Y: np.ndarray, Y_pred: np.ndarray, threshold=None, *args, **kwargs) -> tuple[float, dict]:
        assert threshold is None
        # Threshold not implemented yet
        """Evaluates the model's predictions, returning a main metric and other auxiliary metrics"""

        accuracy = accuracy_score(Y, Y_pred)
        precision = precision_score(Y, Y_pred, zero_division=0)
        recall = recall_score(Y, Y_pred, zero_division=0)
        f1 = f1_score(Y, Y_pred, zero_division=0)
        auc = roc_auc_score(Y, Y_pred)

        return auc, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BaseHyperParamTuner(ABC):
    def __init__(self,
                 modelTrainerParams: dict[str, list],
                 X_train: np.ndarray,
                 X_val: np.ndarray,
                 X_test: np.ndarray,
                 anom_val: np.ndarray,
                 anom_test: np.ndarray):
        
        self.param_grid = modelTrainerParams
        
        # NOTE: For StaticFeaturesTuner, these should be RAW data (or point-features).
        # For DeepLearningTuner, these are likely already flattened windows.
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.anom_val = anom_val
        self.anom_test = anom_test

        self.best_params: dict | None = None
        self.best_score: float = -np.inf
        self.best_model: Trainer | None = None
        self.results: list[dict] = []
        self._bestResult = {"EMPTY":0}
        
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
        """Define iteration over window parameters."""
        pass

    @abstractmethod
    def _prepare_data(self, window_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform data based on window params. 
        This is called inside the OUTER loop, ensuring expensive operations run only once per config.
        """
        pass

    def tune(self, Trainer_factory: Callable[..., Trainer], evaluator: Evaluator) -> None:
        idx = 0
        
        window_grid = self._get_window_grid()
        
        #caro pra carmaba
        for window_params in window_grid:
            print(f"Processing Window Config: {window_params}")
            

            Xtr, validationX, validationLabels = self._prepare_data(window_params)


            for modelParams in self._expandedModelGrid:
                idx+=1
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
                    self._bestResult = record


class DeepLearningTuner(BaseHyperParamTuner):
    def __init__(self, window_params: dict[str, list], **kwargs):
        super().__init__(**kwargs)
        self.window_params = window_params
        self._expandedWindowGrid = self._expand_grid(self.window_params)

    def _get_window_grid(self) -> list[dict]:
        return self._expandedWindowGrid

    def _prepare_data(self, window_params: dict):
        #APENAS REDIMENSIONA SEM LIDAR COM FEATURES
        Xtr = Preprocessing.resizeFlattenedWindow(self.X_train, **window_params)
        Xval = Preprocessing.resizeFlattenedWindow(self.X_val, **window_params)
        anomVal = Preprocessing.resizeFlattenedWindow(self.anom_val, **window_params)
        
        validationX = np.concatenate((Xval, anomVal), axis=0)
        validationLabels = np.concatenate((np.ones(Xval.shape[0]), np.zeros(anomVal.shape[0])))
        
        return Xtr, validationX, validationLabels

def recompute_preprocessing(pp):
    # Adicione *args para capturar (e ignorar) X_train, X_val, anom_val
    # Ou defina explicitamente: def a(X_tr, X_val, anom, new_window_size, ...)
    def a(*args, new_window_size, window_overlap, dimensionsPerSample=None):
        pp.preprocessar_todos_deepLearning(window_size=new_window_size,window_overlap=window_overlap)
        return *(pp.normal_splits)[:2],pp.anomalo_splits[0]
    return a



class StaticFeaturesTuner(BaseHyperParamTuner):
    """
    Tuner for Traditional ML models where windowing involves expensive Feature Engineering.
    """
    def __init__(self, 
                window_params: dict[str, list], 
                 feature_engineering_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]=recompute_preprocessing,  #type:ignore
                 **kwargs):
        """
        Args:
            feature_engineering_fn: Function that accepts (X_train, X_val, anom_val, **window_params)
                                    and returns (X_train_feats, X_val_feats, anom_val_feats).
                                    The inputs X_train etc. will be the RAW data stored in this class.
            window_params: Grid of parameters to pass to the function (e.g. {'window_size': [10, 20]}).
        """
        super().__init__(**kwargs)
        self.feature_engineering_fn = feature_engineering_fn
        self.window_params = window_params
        self._expandedWindowGrid = self._expand_grid(self.window_params)

    def _get_window_grid(self) -> list[dict]:
        return self._expandedWindowGrid

    def _prepare_data(self, window_params: dict):
        
        # 1. Call the user-provided function to generate features from RAW data
        # This is the "expensive" step, executed only once per window config via the Base class loop.
        X_train_feats, X_val_feats, anom_val_feats = self.feature_engineering_fn(
            self.X_train, 
            self.X_val, 
            self.anom_val, 
            **window_params
        )
        
        # 2. Prepare validation concatenation (Standard logic)
        validationX = np.concatenate((X_val_feats, anom_val_feats), axis=0)
        validationLabels = np.concatenate((
            np.ones(X_val_feats.shape[0]), 
            np.zeros(anom_val_feats.shape[0])
        ))
        
        return X_train_feats, validationX, validationLabels


