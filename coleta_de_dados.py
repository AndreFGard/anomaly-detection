import os
import pandas as pd
import kagglehub


class Dataset:
    """Classe para gerenciar coleta e carregamento de dados de IMU de braço robótico"""
    
    def __init__(self, dataset_name='hkayan/industrial-robotic-arm-imu-data-casper-1-and-2'):
        """
        Inicializa o Dataset
        
        Args:
            dataset_name: Nome do dataset no Kaggle
        """
        self.dataset_name = dataset_name
        self.dataset_path = self._obter_caminho_dataset()
        self.df_normal = None
        self.df_faulty = None
        self.lista_dfs_anomaly = None
        self.df_combined = None
        self._carregar_dados()
        self._combinar_dados()
        
    def _obter_caminho_dataset(self):
        """Obtém o caminho do dataset (local ou download)"""
        caminho = os.environ.get("DATASET_PATH")
        if not caminho:
            caminho = kagglehub.dataset_download(self.dataset_name) + '/'
        return caminho
    
    def listar_arquivos(self, caminho='/kaggle/input'):
        """Lista todos os arquivos disponíveis no diretório"""
        arquivos = []
        for dirname, _, filenames in os.walk(caminho):
            for filename in filenames:
                arquivo_completo = os.path.join(dirname, filename)
                print(arquivo_completo)
                arquivos.append(arquivo_completo)
        return arquivos
    
    def _carregar_dados(self, arquivo_normal='IMU_10Hz.csv'):
        """
        Carrega os dados normais e todos os tipos de falha disponíveis
        
        Args:
            arquivo_normal: Nome do arquivo com dados normais
            
        Returns:
            tuple: (df_normal, df_faulty, lista_dfs_anomaly)
        """
        print("--- CARREGAMENTO MANUAL DE CENÁRIOS ---")
        
        # 1. Carregar o NORMAL (df)
        # ------------------------------------------------------------------
        print("Lendo Base Normal...")
        self.df_normal = pd.read_csv(self.dataset_path + arquivo_normal)
        self.df_normal['label'] = 0
        self.df_normal['scenario'] = 'Normal'
        
        # 2. Carregar AS ANOMALIAS (faultydf)
        # ------------------------------------------------------------------
        print("Lendo Base de Falhas...")
        
        # Lista simples e direta com TODOS os arquivos de problema disponíveis
        arquivos_falha = [
            'IMU_hitting_platform.csv',   # Colisão: Plataforma
            'IMU_hitting_arm.csv',        # Colisão: Braço (Robô se batendo)
            'IMU_extra_weigth.csv',       # Mecânico: Peso Extra (Esforço)
            'IMU_earthquake.csv',         # Ambiental: Terremoto (Vibração externa)
        ]
        
        lista_dfs = []
        
        for arquivo in arquivos_falha:
            # Carrega cada um individualmente
            temp_df = pd.read_csv(self.dataset_path + arquivo)
            
            # Padroniza
            temp_df['label'] = 1             # Todo mundo aqui é erro
            temp_df['scenario'] = arquivo    # Guarda o nome pra você saber o que é
            
            lista_dfs.append(temp_df)
            print(f"-> Adicionado: {arquivo} ({len(temp_df)} linhas)")
        
        df, df_val, df_test = self.split_train_val_test()
        # Junta todos os arquivos da lista em um só DataFrame
        self.df_faulty = df_val[df_val['label'] == 1].copy()
        self.lista_dfs_anomaly = [df for scenario, df in self.df_faulty.groupby('scenario')]
        
        print("="*60)
        print(f"DATASET PRONTO:")
        print(f"-> Dados Normais: {len(self.df_normal)} linhas")
        print(f"-> Dados de Falha:  {len(self.df_faulty)} linhas (Total de 4 tipos de defeito)")
        
        return self.df_normal, self.df_faulty, self.lista_dfs_anomaly
    
    def _combinar_dados(self):
        """
        Combina os datasets normal e faulty em um único DataFrame
        
        Returns:
            pd.DataFrame: Dataset combinado
        """
        if self.df_normal is None or self.df_faulty is None:
            raise ValueError("Carregue os dados primeiro usando _carregar_dados()")
        
        self.df_combined = pd.concat([self.df_normal, self.df_faulty], 
                                      ignore_index=True)
        return self.df_combined
    
    def obter_info(self):
        """Retorna informações sobre os datasets carregados"""
        info = {}
        if self.df_normal is not None:
            info['normal'] = {
                'shape': self.df_normal.shape,
                'colunas': list(self.df_normal.columns)
            }
        if self.df_faulty is not None:
            info['faulty'] = {
                'shape': self.df_faulty.shape,
                'colunas': list(self.df_faulty.columns)
            }
        if self.df_combined is not None:
            info['combined'] = {
                'shape': self.df_combined.shape,
                'distribuicao_labels': self.df_combined['label'].value_counts().to_dict()
            }
        return info
    
    @staticmethod
    def split_sequencial(df, p_train=0.7, p_val=0.1, p_test=0.2):
        """Corta um DataFrame em 3 pedaços sequenciais baseados nas porcentagens."""
        size = len(df)
        end_train = int(size * p_train)
        end_val = int(size * (p_train + p_val))

        train = df.iloc[:end_train].copy()
        val = df.iloc[end_train:end_val].copy()
        test = df.iloc[end_val:].copy()

        return train, val, test
    
    def split_train_val_test(self):
        norm_train, norm_val, norm_test = self.split_sequencial(self.df_normal)
        # Listas para acumular os pedaços (começamos com o normal)
        final_train_list = [norm_train]
        final_val_list   = [norm_val]
        final_test_list  = [norm_test]

        print(f"1. Normal processado: {len(self.df_normal)} linhas divididas.")

        for df_falhas in self.lista_dfs_anomaly:
            f_train, f_val, f_test = self.split_sequencial(df_falhas,0,0.5,0.5)
            print(f"Falha {df_falhas.iloc[0]['scenario']} processado" )
            final_val_list.append(f_val)
            final_test_list.append(f_test)

        df_train_final = pd.concat(final_train_list, ignore_index=True)
        df_val_final = pd.concat(final_val_list, ignore_index=True)
        df_test_final = pd.concat(final_test_list, ignore_index=True)

        return df_train_final, df_val_final, df_test_final



