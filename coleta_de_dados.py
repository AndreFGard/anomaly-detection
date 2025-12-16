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
        
        # Junta todos os arquivos da lista em um só DataFrame
        self.lista_dfs_anomaly = lista_dfs
        self.df_faulty = pd.concat(lista_dfs, ignore_index=True)
        
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


