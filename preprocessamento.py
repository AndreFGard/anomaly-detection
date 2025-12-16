from matplotlib.pylab import normal
import pandas as pd
import numpy as np
from pandas.core import apply
from scipy.stats import kurtosis, skew
import sklearn
from sklearn.preprocessing import RobustScaler
from analise_exploratoria_de_dados import EDA
import time
from analise_exploratoria_de_dados import EDA
df=None


class Preprocessing:
    eda= EDA()

    def __init__(self,arquivo_normal= "IMU_10Hz.csv", arquivos_anomalos=["IMU_hitting_platform.csv"], filtro_savgol=True):
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
        normal['time'] = normal['time'].map(lambda x: x/1e6) #type:ignore
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


        def merge_smooth_columns(df):
            # for each column, if it contains _smooth, replace the original column with it
            for col in df.columns:
                if col.endswith('_smooth'):
                    original_col = col[:-7]
                    df[original_col] = df[col]
                    df.drop(columns=[col], inplace=True)
            return df  
        normal = merge_smooth_columns(normal)
        anomalos = [merge_smooth_columns(anomalo) for anomalo in anomalos]


        normal = eda.resampling_and_interpolate(normal, arquivo_normal)
        anomalos = [eda.resampling_and_interpolate(anomalo, nome) for anomalo,nome in zip(anomalos,arquivos_anomalos)]

        normal_cols = list(normal.columns)
        anomalos = [anomalo[normal_cols] for anomalo in anomalos] #evita um problema de ordem nas colunas

        self.normal = normal
        self.anomalos = anomalos #type:ignore
        self.anomalo_nomes = arquivos_anomalos
    

    
    @staticmethod
    def _getFixedWindows(df:pd.DataFrame|np.ndarray, length, overlap):
        #drop incomplete implícito
        arr: np.ndarray = df.values if isinstance(df, pd.DataFrame) else df
        step = length - overlap
        n = arr.shape[0]

        starts = range(0, n - length + 1, step)
        windows = np.stack([arr[s:s+length] for s in starts], axis=0) #semelhante a np.array

        return windows
    
    
    @staticmethod
    def __preprocessar_DL__(df:pd.DataFrame, test_splits=[0.0,0.1,0.9], window_size=60, window_overlap=10, scaler=None, fit_scaler=False, stratifyCol=None) -> list[np.ndarray]:
        """Executa o preprocessamento completo de um df , sem exibir nada
        Args:
            window_size: tamanho da janela em samples continuas
            window_overlap: interseção entre uma janela e a seguinte ou à antecessora

        Returns:
            df_train, df_val, df_test, df_val, df_test: Windows já achatadas,
                com dados já normalizados, normalizados e limpos, com feature engineering pronto.
        """
        

        df_train, df_val,df_test = Preprocessing._train_test_split(df, *test_splits,stratifyCol=stratifyCol)
        if stratifyCol:
            #assume that we want to remove thje label (causa bug?)
            df_train = df_train.drop(columns=[stratifyCol]) if df_train is not None and len(df_train) else None
            df_val = df_val.drop(columns=[stratifyCol])
            df_test = df_test.drop(columns=[stratifyCol])
        if scaler is None:
            scaler = RobustScaler()
            fit_scaler = True
        dftrain_exists =  (df_train is not None and len(df_train))
        if fit_scaler and dftrain_exists:
            scaler.fit(df_train)
        
        
        df_train =  pd.DataFrame(
            scaler.transform(df_train),
            columns=df_train.columns) if dftrain_exists > 0 else df_train
        df_val = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns)
        df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

        if dftrain_exists: df_train_w = Preprocessing._getFixedWindows(df_train, window_size, window_overlap)
        df_val_w = Preprocessing._getFixedWindows(df_val,  window_size, window_overlap)
        df_test_w  = Preprocessing._getFixedWindows(df_test,  window_size, window_overlap)


        if dftrain_exists: df_train_flat = df_train_w.reshape(df_train_w.shape[0], -1) #type:ignore
        df_val_flat = df_val_w.reshape(df_val_w.shape[0], -1)
        df_test_flat  = df_test_w.reshape(df_test_w.shape[0], -1)

        return [df_train_flat  if dftrain_exists else None, df_val_flat, df_test_flat] #type:ignore
    
    @staticmethod
    def _train_test_split(
        df: pd.DataFrame,
        trainPer=0.6,
        valPer=0.05,
        testPer=0.35,
        stratifyCol=None
    ) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        assert trainPer + valPer + testPer <= 1.0
        n = len(df)

        if stratifyCol is None:
            train_end = int(n * trainPer)
            val_end   = train_end + int(n * valPer)

            traindf = df.iloc[:train_end].copy() if trainPer else None
            valdf   = df.iloc[train_end:val_end].copy() if valPer else None
            testdf  = df.iloc[val_end:val_end + int(n * testPer)].copy()

            return (traindf, valdf, testdf) #type:ignore

        train_idx = []
        val_idx = []
        test_idx = []

        for label, group in df.groupby(stratifyCol, sort=False):
            idx = group.index.to_numpy()
            m = len(idx)

            t_end = int(m * trainPer)
            v_end = t_end + int(m * valPer)

            train_idx.append(idx[:t_end])
            val_idx.append(idx[t_end:v_end])
            test_idx.append(idx[v_end:v_end + int(m * testPer)])

        # Merge and restore temporal order
        train_idx = np.sort(np.concatenate(train_idx)) if trainPer else None
        val_idx   = np.sort(np.concatenate(val_idx)) if valPer else None
        test_idx  = np.sort(np.concatenate(test_idx))

        traindf = df.loc[train_idx].copy() if trainPer else None
        valdf   = df.loc[val_idx].copy() if valPer else None
        testdf  = df.loc[test_idx].copy()

        return traindf, valdf, testdf #type:ignore

    from scipy.stats import kurtosis, skew
    @staticmethod
    def __add_point_engineered_features__(df:pd.DataFrame) -> pd.DataFrame:
        #add point only features
        # sens_jerk e sens_norm
        sensors = "acc gyro mag".split(' ')
        df = df.copy()
        # for sensAxis in base_features:
        #     df[f"{sensAxis}_jerk"] = df[sensAxis].diff()
        #jerk removido por variância irrisória
        for sens in sensors:
            df[f"{sens}_norm"] = np.sqrt(df[f"{sens}X"]**2 + df[f"{sens}Y"]**2 + df[f"{sens}Z"]**2)

        return df
    @staticmethod
    def _make_windows(df, length, overlap, drop_incomplete=True):
        arr = df.values  # (T, C)
        step = length - overlap
        n = arr.shape[0]

        stops = n - length + 1 if drop_incomplete else n
        starts = range(0, stops, step)

        windows = []
        for s in starts:
            w = arr[s:s + length]
            if w.shape[0] == length:
                windows.append(w)

        return np.stack(windows, axis=0)  # (N_windows, L, C)

    @staticmethod
    def __window_feature_engineering__(windows, feature_names, stats=("mean", "std", "ptp", "kurtosis", "crest", "dom_freq")):
        """
        Extracts statistical features from time-series windows for traditional ML models.
        
        Args:
            windows (np.array): Shape (N_samples, Window_Size, N_sensors)
            feature_names (list): List of sensor names strings (e.g., ['accX', 'accY'...])
            stats (tuple): List of stats to compute.
        
        Returns:
            X_feat (np.array): Shape (N_samples, N_features)
            names (list): List of feature names
        """
        feats = []
        names = []
        
        # 1. Mean (General Position/Bias)
        if "mean" in stats:
            feats.append(np.mean(windows, axis=1))
            names += [f"mean_{c}" for c in feature_names]

        # 2. Standard Deviation (Vibration Energy)
        if "std" in stats:
            feats.append(np.std(windows, axis=1))
            names += [f"std_{c}" for c in feature_names]

        # 3. RMS (Total Energy - redundancy with Mean/Std, but useful for physics)
        if "rms" in stats:
            rms = np.sqrt(np.mean(windows**2, axis=1))
            feats.append(rms)
            names += [f"rms_{c}" for c in feature_names]

        # 4. Peak-to-Peak (Amplitude of Shocks - critical for Bumps)
        if "ptp" in stats:
            feats.append(np.ptp(windows, axis=1))
            names += [f"ptp_{c}" for c in feature_names]

        # 5. Kurtosis (Impulsiveness - critical for Hits/Earthquakes)
        if "kurtosis" in stats:
            # Fisher=False makes normal distribution = 3.0. 
            # Often easier to use Fisher=True (normal = 0.0) for ML centering.
            k = kurtosis(windows, axis=1, fisher=True) 
            feats.append(k)
            names += [f"kurt_{c}" for c in feature_names]

        # 6. Skewness (Asymmetry - useful for directional crashes)
        if "skew" in stats:
            s = skew(windows, axis=1)
            feats.append(s)
            names += [f"skew_{c}" for c in feature_names]

        # 7. Crest Factor (Impact Indicator - Peak / RMS)
        if "crest" in stats:
            peak = np.max(np.abs(windows), axis=1)
            rms = np.sqrt(np.mean(windows**2, axis=1))
            # Add small epsilon to avoid division by zero
            crest = peak / (rms + 1e-9) 
            feats.append(crest)
            names += [f"crest_{c}" for c in feature_names]
            
        # 8. Dominant Frequency (Resonance - useful for Heavy Weight detection)
        if "dom_freq" in stats:
            # Perform FFT along the time axis (axis 1)
            fft_vals = np.fft.rfft(windows, axis=1)
            fft_freq = np.fft.rfftfreq(windows.shape[1])
            
            # Find index of max magnitude (ignoring DC component at index 0)
            magnitudes = np.abs(fft_vals)
            magnitudes[:, 0, :] = 0  # Zero out DC component
            
            dom_indices = np.argmax(magnitudes, axis=1) # Shape (N, C)
            
            # Map indices to actual frequencies
            dom_freqs = fft_freq[dom_indices]
            
            feats.append(dom_freqs)
            names += [f"domfreq_{c}" for c in feature_names]

        X_feat = np.concatenate(feats, axis=1)  # (N, Total_Features)
        return X_feat, names

    @staticmethod
    def __non_DL_feature_engineering_pipeline__(ds, WINDOW_SIZE=40, WINDOW_OVERLAP=10):
        #somente do ponto
        ds_pe = Preprocessing.__add_point_engineered_features__(ds)

        ds_w = Preprocessing._make_windows(ds_pe, WINDOW_SIZE, WINDOW_OVERLAP)

        #somente da janela
        ds_feat, feat_names = Preprocessing.__window_feature_engineering__(
            ds_w,
            feature_names=ds_pe.columns,
            )
        return ds_feat, feat_names
    
    @staticmethod
    def __PCA_normalize__(X_train_masked, X_val_masked=None, X_test_masked=None, variance_threshold=0.95):
        """
        Faz scaling e PCA nos splits. recebe os datasets ja somente com as features usadas
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_masked)
        
        # 2. Fit PCA
        pca = PCA(n_components=variance_threshold)
        X_train_pca = pca.fit_transform(X_train_scaled)
        
        print(f"PCA:")
        print(f"  - Features antes : {X_train_masked.shape[1]}")
        print(f"  - Features após: {X_train_pca.shape[1]}")
        print(f"  - Variância explicada: {np.sum(pca.explained_variance_ratio_):.4f}")

        X_val_pca = None
        X_test_pca = None
        
        if X_val_masked is not None:
            X_val_scaled = scaler.transform(X_val_masked)
            X_val_pca = pca.transform(X_val_scaled)
            
        if X_test_masked is not None:
            X_test_scaled = scaler.transform(X_test_masked)
            X_test_pca = pca.transform(X_test_scaled)

        return pca, scaler, X_train_pca, X_val_pca, X_test_pca
    
    @staticmethod
    def __preprocessar_non_DL__(train:pd.DataFrame|np.ndarray, val, test, pca=None, scaler=None,mask=None,WINDOW_SIZE=60,WINDOW_OVERLAP=10):
        
        train,names = Preprocessing.__non_DL_feature_engineering_pipeline__(train, WINDOW_SIZE,WINDOW_OVERLAP)
        
        val,names = Preprocessing.__non_DL_feature_engineering_pipeline__(val, WINDOW_SIZE,WINDOW_OVERLAP)
        test,names = Preprocessing.__non_DL_feature_engineering_pipeline__(test, WINDOW_SIZE,WINDOW_OVERLAP)
        assert len(train.shape) == 2 and  len(val.shape) == 2 and len(test.shape) == 2

        if mask is None:
            var = train.var(axis=0)
            VAR_THRESHOLD = 1e-5
            keep_var = var > VAR_THRESHOLD

            # Apply variance mask first
            train_var = train[:, keep_var]

            corr = np.corrcoef(train_var, rowvar=False)
            CORR_THRESHOLD = 0.95
            to_drop = set()
            for i in range(corr.shape[0]):
                for j in range(i+1, corr.shape[0]):
                    if abs(corr[i,j]) > CORR_THRESHOLD:
                        to_drop.add(j)

            keep_corr = [i not in to_drop for i in range(corr.shape[0])]

            # Final mask: first variance, then correlation
            mask = np.zeros(train.shape[1], dtype=bool)
            mask[np.where(keep_var)[0][keep_corr]] = True
        def apply_mask(arr):
            return arr[:, mask]
        
        if not pca:
            assert not scaler
            a,b,c = apply_mask(train), apply_mask(val),apply_mask(test)
            pca, scaler, train_pca,val_pca,test_pca = Preprocessing.__PCA_normalize__(
                a,b,c,variance_threshold=0.95
            )
        else:
            assert pca and scaler
            a,b,c = apply_mask(train), apply_mask(val),apply_mask(test)
            train_pca,val_pca,test_pca = (pca.transform(scaler.transform(a)),
                pca.transform(scaler.transform(b)),
                pca.transform(scaler.transform(c)))
        
        return pca,scaler,mask,train_pca,val_pca,test_pca
        

            

    def preprocessar_todos_non_deepLearning(self, aplicar_savgol=True, train_splits = [0.6,0.2,0.2], test_splits=[0.0,0.5,0.5], window_size=60, window_overlap=10):
        assert aplicar_savgol #motivos de compatibilidade de api
        anomalos = [ano.copy() for ano in self.anomalos]
        for i,ano in enumerate(anomalos):
            ano['label'] = i
        anomalo_total = pd.concat(anomalos)

        xtrain,xval,xtest = Preprocessing._train_test_split(self.normal,*train_splits)

        _,anom_val, anom_test = Preprocessing._train_test_split(anomalo_total,trainPer=0,valPer=0.5,testPer=0.5)#type:ignore
        anom_val = anom_val.drop(columns=['label'])
        anom_test = anom_test.drop(columns=['label'])

        #feature enginering e windowing

        pca,scaler,mask,xtrain,xval,xtest = Preprocessing.__preprocessar_non_DL__(xtrain,xval,
            xtest,pca=None,scaler=None,mask=None,
            WINDOW_SIZE=window_size,WINDOW_OVERLAP=window_overlap)
        pca,scaler,mask,_,anom_val,anom_test = Preprocessing.__preprocessar_non_DL__(anom_val,anom_val,
            anom_test,pca=pca,scaler=scaler,mask=mask,
            WINDOW_SIZE=window_size,WINDOW_OVERLAP=window_overlap)

        self.normal_splits = [xtrain,xval,xtest] #type:ignore
        self.anomalo_splits=[anom_val,anom_test]#type:ignore

        
    def preprocessar_todos_deepLearning(self, aplicar_savgol=True, train_splits = [0.6,0.2,0.2], test_splits=[0.0,0.5,0.5], window_size=60, window_overlap=10):
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
        assert aplicar_savgol #motivos de compatibilidade de api
        anomalos = [ano.copy() for ano in self.anomalos]
        for i,ano in enumerate(anomalos):
            ano['label'] = i
        anomalo_total = pd.concat(anomalos)



        scaler = RobustScaler()
        self.normal_splits = Preprocessing.__preprocessar_DL__(self.normal, train_splits,
                                                         window_size,
                                                         window_overlap,
                                                         scaler=scaler,
                                                         fit_scaler=True)

        self.anomalo_splits = Preprocessing.__preprocessar_DL__(anomalo_total,  test_splits, window_size,
                                             window_overlap,scaler=scaler,
                                             fit_scaler=False, stratifyCol='label')[1:]
                                            
        return None
    @staticmethod
    def resizeFlattenedWindow( flattened_windows:np.ndarray, new_window_size:int, window_overlap:int, dimensionsPerSample=9) -> np.ndarray:
        """Redimensiona  as janelas (já) achatadas para um novo tamanho"""
        assert (len(flattened_windows.shape) == 2)
        if window_overlap >= new_window_size:
            window_overlap = new_window_size//2
        
        samplesPerWindow = flattened_windows.shape[1]//dimensionsPerSample
        samples = flattened_windows.reshape(-1, dimensionsPerSample)
        
        newWindows = Preprocessing._getFixedWindows(samples, new_window_size, window_overlap)
        flattenedNewWindows = newWindows.reshape(newWindows.shape[0], -1)
        return flattenedNewWindows
    


pp = Preprocessing()
pp.preprocessar_todos_non_deepLearning()
print(pp.anomalo_splits[1].shape)