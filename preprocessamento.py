from typing import no_type_check
from matplotlib.pylab import normal
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler
from analise_exploratoria_de_dados import EDA
import time

df=None
class Preprocessing:
    eda= EDA()

    def __init__(self,arquivo_normal= "IMU_10Hz.csv", arquivos_anomalos=["IMU_hitting_platform.csv", "IMU_hitting_arm.csv"], filtro_savgol=True):
        """Classe que carrega e prepara os dados. Ao ser instanciada, apenas fará
        preparações que não serão customizadas (isto é, sem hiperparâmetros).
        
        Methods:
            preprocessar_df: Preprocessa completamnete um df, incluindo hiperparâmetros
            
            preprocessar_todos: Preprocessa todos os datasets carregados por esta classe"""
        

        self.normal: pd.DataFrame = None #type: ignore
        self.anomalos: list[pd.DataFrame] = [] 
        self.anomalo_nomes: list[str] = []

        self.normal_splits: list[np.ndarray] = []
        self.anomalo_splits: list[list[np.ndarray]] = []


        eda= Preprocessing.eda
        normal = pd.read_csv(eda.dataset.dataset_path + arquivo_normal)
        anomalos = [pd.read_csv(eda.dataset.dataset_path + fname) for fname in arquivos_anomalos]

        cols = list(set(normal.columns).difference(set(["label", 'name'])))
        normal = normal[cols]
        normal['time'] = normal['time'].map(lambda x: x/1e6)
        if filtro_savgol:
            normal = Preprocessing.eda.aplicar_filtro_savgol(normal)


        # Corrige os anômalos e atualiza a lista
        novos_anomalos = []
        for anomalo in anomalos:
            if filtro_savgol:
                anomalo = Preprocessing.eda.aplicar_filtro_savgol(anomalo)
            cols = list(set(anomalo.columns).difference(set(["label", 'name'])))
            anomalo['time'] = anomalo['time'].map(lambda x: x/1e6)
            anomalo = anomalo[cols]  # Garante que 'time' está presente
            novos_anomalos.append(anomalo)
        anomalos = novos_anomalos

        normal = eda.resampling_and_interpolate(normal, arquivo_normal)
        anomalos = [eda.resampling_and_interpolate(anomalo, nome) for anomalo,nome in zip(anomalos,arquivos_anomalos)]

        normal_cols = list(normal.columns)
        anomalos = [anomalo[normal_cols] for anomalo in anomalos] #evita um problema de ordem nas colunas

        self.normal = normal
        self.anomalos = anomalos #type:ignore
        self.anomalo_nomes = arquivos_anomalos
    
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
    
    
    @staticmethod
    def preprocessar(df:pd.DataFrame, test_splits=[0.0,0.1,0.9], window_size=60, window_overlap=10, scaler=None, fit_scaler=False) -> list[np.ndarray]:
        """Executa o preprocessamento completo de um df , sem exibir nada
        Args:
            window_size: tamanho da janela em samples continuas
            window_overlap: interseção entre uma janela e a seguinte ou à antecessora

        Returns:
            df_train, df_val, df_test, df_val, df_test: Windows já achatadas,
                com dados já normalizados, normalizados e limpos, com feature engineering pronto.
        """

        df_train, df_val,df_test = Preprocessing._train_test_split(df, *test_splits)

        if scaler is None:
            scaler = RobustScaler()
            fit_scaler = True
        if fit_scaler and len(df_train)>0:
            scaler.fit(df_train)
        
        df_train =  pd.DataFrame(
            scaler.transform(df_train),
            columns=df_train.columns) if len(df_train) > 0 else df_train
        df_val = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns)
        df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

        if len(df_train) > 0: df_train_w = Preprocessing._getFixedWindows(df_train, window_size, window_overlap)
        df_val_w = Preprocessing._getFixedWindows(df_val,  window_size, window_overlap)
        df_test_w  = Preprocessing._getFixedWindows(df_test,  window_size, window_overlap)


        if len(df_train) > 0: df_train_flat = df_train_w.reshape(df_train_w.shape[0], -1) #type:ignore
        df_val_flat = df_val_w.reshape(df_val_w.shape[0], -1)
        df_test_flat  = df_test_w.reshape(df_test_w.shape[0], -1)

        return [df_train_flat if len(df_train) > 0 else None, df_val_flat, df_test_flat] #type:ignore
    
    def preprocessar_todos(self, aplicar_savgol=True, train_splits = [0.6,0.2,0.2], test_splits=[0.0,0.1,0.9], window_size=60, window_overlap=10):
        """Preprocessa todos os datasets carregados por esta classe e os coloca em
            self.normal_splits e self.anomalo_splits
        
        Args:
            aplicar_savgol: Se deve aplicar o filtro de Savgolay
            window_size: tamanho da janela em samples continuas
            window_overlap: interseção entre uma janela e a seguinte ou à antecessora

        Returns:
            dict: dicionário com chaves 'normal' e nomes dos datasets anômalos,
                  cada um contendo uma lista [df_train, df_val, df_test]
        """
        scaler = RobustScaler()
        self.normal_splits = Preprocessing.preprocessar(self.normal, train_splits,
                                                         window_size,
                                                         window_overlap,
                                                         scaler=scaler,
                                                         fit_scaler=True)
        
        self.anomalo_splits = [ Preprocessing.
                               preprocessar(anomalo,  test_splits, window_size,
                                             window_overlap,scaler=scaler,
                                             fit_scaler=False)
                                             for anomalo in self.anomalos]
    
        return None
        

# Exemplo de uso:
pp = Preprocessing()
pp.preprocessar_todos()
pp.anomalo_splits[1][1].shape