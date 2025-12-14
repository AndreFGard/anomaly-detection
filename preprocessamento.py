from typing import no_type_check
import pandas as pd
import numpy as np
import sklearn
from analise_exploratoria_de_dados import EDA

df=None
class Preprocessing:
    def __init__(self,aplicar_filtro=True):
        """Classe que carrega e prepara os dados.
        \nMetodos:
            \n- run(): faz o preprocessamento"""

        dataset= EDA()
        dataset.carregar_e_preparar_dados()
        dataset.aplicar_resampling()
        dataset.aplicar_normalizacao()
        if aplicar_filtro:
            dataset.aplicar_filtro_savgol(dataset.df_normal)

        self.df_normal: pd.DataFrame  = dataset.df_normal #type:ignore
        self.df_faulty: pd.DataFrame = dataset.df_faulty  #type: ignore

        cols = list(set(self.df_normal.columns) ^set(["time", "label"]))
        self.df_normal = self.df_normal[cols]  #type:ignore
        self.df_faulty = self.df_faulty[cols]  #type:ignore
    
    @staticmethod
    def _train_test_split(df:pd.DataFrame, trainPer=0.6,valPer=0.05, testPer=0.35):
        assert(trainPer + valPer + testPer <= 1.0)

        size = df.shape[0]

        traindf = df.iloc[:int(size*trainPer)]
        valdf = df.iloc[int(size*trainPer)  : int(size*(valPer + trainPer))  ] 
        testdf = df.iloc[int(size*(valPer+ trainPer)): ] 
        return traindf,valdf,testdf
    
    @staticmethod
    def _getFixedWindows(df:pd.DataFrame, length, overlap, drop_incomplete=True):
        arr = df.values
        step = length - overlap
        n = arr.shape[0]

        starts = range(0, n - length + 1, step)
        windows = np.stack([arr[s:s+length] for s in starts], axis=0) #semelhante a np.array
        return windows

    def run(self, WINDOW_SIZE=60, WINDOW_OVERLAP=10) -> list[np.ndarray]:
        """Executa o preprocessamento completo e retorna os conjuntos preprocessados, sem exibir nada
        Args:
            WINDOW_SIZE: tamanho da janela em samples continuas
            WINDOW_OVERLAP: interseção entre uma janela e a seguinte ou à antecessora

        Returns:
            X_train, X_val, X_test, Y_val, Y_test: Windows já achatadas, com dados já normalizados, normalizados e limpos
        """

        X_train, X_val,X_test = self._train_test_split(self.df_normal)

        _, Y_val,Y_test = self._train_test_split(self.df_faulty,trainPer=0,valPer=0.1,testPer=0.9)


        WINDOW_SIZE, WINDOW_OVERLAP = 60, 10

        X_train_w = self._getFixedWindows(X_train, WINDOW_SIZE, WINDOW_OVERLAP)
        X_val_w = self._getFixedWindows(X_val,  WINDOW_SIZE, WINDOW_OVERLAP)
        X_test_w  = self._getFixedWindows(X_test,  WINDOW_SIZE, WINDOW_OVERLAP)

        Y_val_w  = self._getFixedWindows(Y_val, WINDOW_SIZE, WINDOW_OVERLAP)
        Y_test_w  = self._getFixedWindows(Y_test, WINDOW_SIZE, WINDOW_OVERLAP)


        X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1)
        X_val_flat = X_val_w.reshape(X_val_w.shape[0], -1)
        X_test_flat  = X_test_w.reshape(X_test_w.shape[0], -1)

        Y_val_flat = Y_val_w.reshape(Y_val_w.shape[0], -1)
        Y_test_flat  = Y_test_w.reshape(Y_test_w.shape[0], -1)

        return [X_train_flat,X_test_flat,X_val_flat, Y_val_flat, Y_test_flat]
        

# Exemplo de uso:
#pp = Preprocessing()
#pp.run()
