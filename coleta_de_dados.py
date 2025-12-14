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
    
    def _carregar_dados(self, arquivo_normal='IMU_10Hz.csv', 
                       arquivo_faulty='IMU_hitting_platform.csv'):
        """
        Carrega os dados normais e com falha
        
        Args:
            arquivo_normal: Nome do arquivo com dados normais
            arquivo_faulty: Nome do arquivo com dados com falha
            
        Returns:
            tuple: (df_normal, df_faulty)
        """
        # Carrega dados normais
        self.df_normal = pd.read_csv(self.dataset_path + arquivo_normal)
        self.df_normal['label'] = 0
        
        # Carrega dados com falha
        self.df_faulty = pd.read_csv(self.dataset_path + arquivo_faulty)
        self.df_faulty['label'] = 1
        
        return self.df_normal, self.df_faulty
    
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


