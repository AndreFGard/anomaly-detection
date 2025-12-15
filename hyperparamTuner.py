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

class HyperParamTuner:
    def __init__(self,
                    windowParams:dict[str,list],
                    modelTrainerParams:dict[str,list],
                    X_train:np.ndarray,
                    X_val:np.ndarray,
                    X_test:np.ndarray,
                    anom_val:np.ndarray,
                    anom_test:np.ndarray
                    ):
        
        """Tunar hiperparametros
    
        Args:
            windowParams: Dicionário com parâmetros do resizeWindows do Preprocessing (formato {new_window_size: [1], dimensionsPerSample: [9]})
            modelTrainerParams: Dicionário com parâmetros que serão passados ao trainer
            X_train: Dados de treino
            X_val: Dados de validação
            X_test: Dados de teste
            anom_val: Rótulos de validação
        """
        self.windowParams = windowParams
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

        self._expandedWindowGrid = self._expand_grid(self.windowParams)
        self._expandedModelGrid = self._expand_grid(self.param_grid)


    @staticmethod
    def _expand_grid(grid: dict[str, list]) -> list[dict]:
        if not grid:
            return [{}]

        keys = list(grid.keys())
        values = [
            v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for v in grid.values()
        ]

        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]


    
    def tune(
        self,
        Trainer_factory: Callable[..., Trainer],
        evaluator: Evaluator
    ) -> None:

        for windowParams in self._expandedWindowGrid:
            Xtr = Preprocessing.resizeFlattenedWindow(self.X_train, **windowParams)
            Xval = Preprocessing.resizeFlattenedWindow(self.X_val, **windowParams)
            anomVal = Preprocessing.resizeFlattenedWindow(self.anom_val, **windowParams)
            
            validationX = np.concatenate((Xval, anomVal), axis=0)
            validationLabels = np.concatenate((np.ones(Xval.shape[0]),
                                                np.zeros(anomVal.shape[0])))


            for modelParams in self._expandedModelGrid:
                trainer = Trainer_factory(**modelParams)

                trainer.TrainerFit(Xtr, None)
                anom_val_pred = trainer.TrainerPred(validationX)

                score, metrics = evaluator.evaluate(
                     validationLabels,anom_val_pred
                )

                record = {
                    "score": score,
                    "metrics": metrics,
                    "window_params": windowParams,
                    "model_params": modelParams,
                }

                self.results.append(record)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = record
                    self.best_model = trainer

        