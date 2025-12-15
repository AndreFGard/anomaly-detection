
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, StandardScaler
from coleta_de_dados import Dataset


class EDA:
    """Classe para análise exploratória de dados de IMU de braço robótico"""
    
    def __init__(self, dataset_name='hkayan/industrial-robotic-arm-imu-data-casper-1-and-2'):
        """
        Inicializa a classe EDA com um dataset
        
        Args:
            dataset_name: Nome do dataset no Kaggle
        """
        self.dataset = Dataset(dataset_name)
        self.df_normal = None
        self.df_faulty = None
        self.df_normal_resampled = None
        self.df_faulty_resampled = None
        self.df_normal_robust_scaled = None
        self.df_faulty_robust_scaled = None
        self.df_normal_standard_scaled = None
        self.df_faulty_standard_scaled = None
        self.scaler_robust = None
        self.scaler_standard = None
        
    def carregar_e_preparar_dados(self, arquivo_normal='IMU_10Hz.csv', 
                                   arquivo_faulty='IMU_hitting_platform.csv'):
        """
        Carrega e prepara os dados para análise
        
        Args:
            arquivo_normal: Nome do arquivo com dados normais
            arquivo_faulty: Nome do arquivo com dados com falha
        """
        # Carrega os dados
        self.df_normal, self.df_faulty = self.dataset._carregar_dados(
            arquivo_normal, arquivo_faulty
        )
        
        # Conversões e limpezas iniciais
        print("\nConvertendo `time` de nanossegundos para milissegundos")
        self.df_normal['time'] = self.df_normal['time'].map(lambda x: x/1e6)
        self.df_faulty['time'] = self.df_faulty['time'].map(lambda x: x/1e6)
        
        print("\nRemovendo a coluna `name` pois não agrega valor preditivo")
        if 'name' in self.df_normal.columns:
            self.df_normal = self.df_normal.drop(columns=['name'])
        if 'name' in self.df_faulty.columns:
            self.df_faulty = self.df_faulty.drop(columns=['name'])
        
        print("\nIdentificando duplicatas")
        print(f"Duplicatas de Tempo: {self.df_normal['time'].duplicated().any()}")
        
        return self.df_normal, self.df_faulty
    
    def executar_analise_completa(self):
        """Executa a análise exploratória completa"""
        if self.df_normal is None or self.df_faulty is None:
            raise ValueError("Carregue os dados primeiro usando carregar_e_preparar_dados()")
        
        print("""## Análise Exploratória Estrutural [Classe Normal]
        - Informações básicas do dataset
        - Tipos de dados
        - Informações detalhadas
        - Estatísticas descritivas
        - Análise de valores únicos""")
        
        # Análise estrutural
        self.analise_estrutural_sensores(self.df_normal)
        
        print("""\n### Análise de Integridade Temporal
        - 10Hz é uma frequência baixa para vibrações mecânicas finas, mas aceitável para movimentos macroscópicos de braços robóticos.
        - Jitter alto (em torno de 20ms), o intervalo entre leituras varia muito. Possivelmente irá introduzir ruído ao utilizar Redes Neurais (CNN/LSTM).
        - Existem 429 momentos onde o sistema perde pacotes, com um gap máximo de 342ms, isso quebra a continuidade da janela deslizante. Será necessário fazer um Resampling para forçar um passo de tempo fixo (ex: interpolar para 100ms exatos) antes de alimentar o modelo.

        ## Análise de valores únicos e constantes
        - A coluna `name` tem apenas 1 valor único e é irrelevante para o problema de detecção de anomalias, ela não agrega valor preditivo. Será necessário fazer um drop da coluna.
        - `gyroY` tem 18% de zeros, o braço robótico passa muito tempo parado (idle). Como 18% é um valor relativamente alto, pode ser considerado definir "estar parado" como um comportamnto Normal (considerar "Zero" como classe Normal).
        - `magX` possui baixa variablidade, pode ser recomendado testar treinar com e sem ele (feature selection).

        ## Análise das estatísticas descritivas
        - O desvio padrão de `accZ` é muito baixo comparado aos outros, possuindo baixa informação preditiva. É um candidato a ser removido se precisar reduzir dimensionalidade (feature selection).
        - A mediana de `gyroY` = 0 confirma que o robô passa a maior parte do tempo parado ou em movimento linear constante (sem rotação). No entanto, uma curtose de 17 é altíssima, isso significa que a distribuição é super pontuda, o que provavelmente acontece é que o robô fica parado quase o tempo todo, mas quando se move, faz movimentos bruscos de início/fim de tarefa. Por isso, qualquer alteração nesse padrão de picos (ex: picos menores = braço lento; picos maiores = colisão) será o principal indicador de anomalia.
        - Devido à alta curtose (gyroY) e outliers naturais (máximos de 100+ no gyro), a normalização padrão irá falhar, pois o StandardScaler (Z-score) usa a média e o desvio padrão. Como o desvio padrão é inflado pelos picos, os dados ficarão em um intervalo muito pequeno. A ideia seria usar RobustScaler que usa a mediana e o intervalo interquartil (IQR), ignorando os picos extremos no cálculo da escala, preservando a forma dos picos do giroscópio.""")
        
        # Visualizações
        print("\nVisualizando `gyroY`:")
        self.plot_sensor(self.df_normal, col_sensor='gyroY')
        print("O boxplot mostra que não deve-se remover os outliers pela regra de desvio padrão (3-sigma), pois serão apagados o movimento do robô sobrando só o ruído dele padrado")
        
        # Comparação Normal vs Falha
        self.comparar_normal_vs_falha(self.df_normal, self.df_faulty, col_sensor='gyroY')
        
        # Análise de valores faltantes
        print("""\n## Análise de Valores Faltantes e Outliers
        - Identificação de valores faltantes e outliers
        - Visualizações de apoio, caso necessário
        - Análise dos mecanismos\n""")
        
        print(f"Quantidade de valores faltantes em df_normal: {self.df_normal.isnull().sum().sum()}")
        print(f"Quantidade de Valores faltantes em df_faulty: {self.df_faulty.isnull().sum().sum()}")
        print(f"\nNão serão removidos os outliers estatísticos pois são movimentos reais")
    
    def analise_estrutural_sensores(self, df, time_col='time', label_col='label'):
        """
        Realiza uma análise exploratória estrutural focada em dados de sensores (Séries Temporais).
        
        Args:
            df: DataFrame com os dados.
            time_col: Nome da coluna de tempo.
            label_col: Nome da coluna de target (anomalia/normal).
        """
        
        print("="*60)
        print("1. INFORMAÇÕES BÁSICAS E TIPOS DE DADOS")
        print("="*60)
        print(f"Dimensões do Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
        print("\nTipos de Dados (Dtypes):")
        print(df.dtypes)
        
        # Verificação de memória
        memoria = df.memory_usage(deep=True).sum() / 1024**2
        print(f"\nUso de Memória: {memoria:.2f} MB")

        print("\n" + "="*60)
        print("2. ANÁLISE DE INTEGRIDADE TEMPORAL (CRÍTICO PARA IMU)")
        print("="*60)
        
        # Ordenar por tempo para garantir análise correta
        df = df.sort_values(by=time_col)
        
        # Calcular diferenças de tempo (Delta T)
        delta_t = df[time_col].diff().dropna()
        
        mean_dt = delta_t.mean()
        std_dt = delta_t.std()
        min_dt = delta_t.min()
        max_dt = delta_t.max()
        
        print(f"- Intervalo de Amostragem Médio (Sampling Rate): {mean_dt:.6f} ms")
        print(f"- Frequência de coleta de dados: {1/(df['time'].diff().mean()/1000):.2f} Hz")
        print(f"- Jitter (Desvio Padrão do tempo): {std_dt:.6f} ms")
        print(f"- Gap Mínimo: {min_dt:.6f} ms | Gap Máximo: {max_dt:.6f} ms")
        
        # Verificar se há gaps significativos (perda de pacotes)
        # Exemplo: Se o gap for maior que 2x a média, é uma quebra de continuidade
        gaps = (delta_t > 2 * mean_dt).sum()
        print(f"- Qtd. de Gaps Temporais Significativos (> 2x média): {gaps}")
        
        print("\n" + "="*60)
        print("3. ANÁLISE DE VALORES ÚNICOS E CONSTANTES (SENSOR FREEZE)")
        print("="*60)
        
        # Separa colunas de sensores (excluindo tempo e label)
        cols_sensores = [c for c in df.columns if c not in [time_col, label_col]]
        
        resumo_unicos = pd.DataFrame({
            'Tipo': df[cols_sensores].dtypes,
            'Qtd_Unicos': df[cols_sensores].nunique(),
            'Unicos (%)': (df[cols_sensores].nunique() / len(df)) * 100,
            'Qtd_Zeros': (df[cols_sensores] == 0).sum(),
            'Zeros (%)': ((df[cols_sensores] == 0).sum() / len(df)) * 100
        })
        
        print(resumo_unicos.sort_values('Qtd_Unicos'))
        
        # Alerta para colunas com baixíssima variabilidade (Sensor travado ou irrelevante)
        cols_travadas = resumo_unicos[resumo_unicos['Qtd_Unicos'] == 1].index.tolist()
        if cols_travadas:
            print(f"\n[ALERTA] Colunas com valor constante (irrelevantes): {cols_travadas}")
        else:
            print("\n[OK] Nenhuma coluna totalmente constante detectada.")

        print("\n" + "="*60)
        print("4. ESTATÍSTICAS DESCRITIVAS DETALHADAS (MOMENTOS)")
        print("="*60)
        # Inclui Skewness e Kurtosis que são vitais para detectar desvios de normalidade em sinais
        desc = df[cols_sensores].describe().T
        desc['skewness'] = df[cols_sensores].skew()
        desc['kurtosis'] = df[cols_sensores].kurt()
        
        print(desc[['mean', 'std', 'min', '50%', 'max', 'skewness', 'kurtosis']])

        print("\n" + "="*60)
        print("5. BALANCEAMENTO DAS CLASSES (TARGET)")
        print("="*60)
        if label_col in df.columns:
            contagem = df[label_col].value_counts()
            percentual = df[label_col].value_counts(normalize=True) * 100
            
            balanceamento = pd.DataFrame({'Contagem': contagem, 'Percentual (%)': percentual})
            print(balanceamento)
            
            ratio = contagem.max() / contagem.min() if len(contagem) > 1 else 0
            print(f"\nRazão de Desbalanceamento: 1 : {ratio:.1f}")
        else:
            print(f"Coluna de target '{label_col}' não encontrada.")
    
    def plot_sensor(self, df, col_sensor, col_time='time'):
        """
        Gera um painel triplo para diagnosticar o comportamento do sensor.
        1. Série Temporal (Visão Geral)
        2. Histograma (Verificar Curtose e Zeros)
        3. Boxplot (Verificar Outliers Extremos)
        """
        
        # Copia para não alterar o original
        df_plot = df.copy()
        
        # Converter tempo para segundos para ficar legível no eixo X
        df_plot['time_sec'] = (df_plot[col_time] - df_plot[col_time].iloc[0]) / 1e3
        
        # Configuração da Figura
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2)
        
        # --- PLOT 1: SÉRIE TEMPORAL ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df_plot['time_sec'], df_plot[col_sensor], color='#1f77b4', linewidth=0.5, alpha=0.8)
        ax1.set_title(f'Série Temporal: {col_sensor}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tempo (segundos)', fontsize=12)
        ax1.set_ylabel('Valor do Sensor', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Destacar a linha do Zero (Ociosidade)
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero (Repouso)')
        ax1.legend()

        # --- PLOT 2: DISTRIBUIÇÃO (HISTOGRAMA + KDE) ---
        ax2 = fig.add_subplot(gs[1, 0])
        # Usamos escala logarítmica no Y devido a quantidade de zeros
        sns.histplot(data=df_plot, x=col_sensor, bins=100, kde=True, ax=ax2, color='#2ca02c')
        ax2.set_yscale('log') # Escala Log para ver as caudas pequenas e o pico gigante
        ax2.set_title(f'Distribuição (Escala Log): {col_sensor}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Valor do Sensor')
        ax2.set_ylabel('Frequência (Log)')
        
        # Anotação da Curtose
        kurt = df_plot[col_sensor].kurt()
        ax2.text(0.95, 0.95, f'Kurtosis: {kurt:.2f}', transform=ax2.transAxes, 
                horizontalalignment='right', verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # --- PLOT 3: BOXPLOT (DETECÇÃO DE OUTLIERS) ---
        ax3 = fig.add_subplot(gs[1, 1])
        sns.boxplot(x=df_plot[col_sensor], ax=ax3, color='#ff7f0e', fliersize=3)
        ax3.set_title(f'Boxplot de Outliers: {col_sensor}', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Valor do Sensor')
        
        plt.tight_layout()
        plt.show()
    
    def comparar_normal_vs_falha(self, df_normal, df_falha, col_sensor='gyroY'):
        """
        Plota comparativo visual entre operação Normal e Falha (Ataque/Colisão).
        """
        # 1. Ajuste de Tempo (reseta para começar do zero em ambos para facilitar visualização)
        t_norm = (df_normal['time'] - df_normal['time'].iloc[0]) / 1e3
        t_fail = (df_falha['time'] - df_falha['time'].iloc[0]) / 1e3
        
        # Recorte: Pegar apenas os primeiros 10 segundos de cada para não poluir
        mask_norm = t_norm <= 10
        mask_fail = t_fail <= 10
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
        
        # --- PLOT 1: COMPARAÇÃO NO TEMPO ---
        # Normal
        axes[0].plot(t_norm[mask_norm], df_normal.loc[mask_norm, col_sensor], 
                    color='#1f77b4', label='Normal', alpha=0.7, linewidth=1)
        axes[0].set_title(f'Padrão Normal vs. Falha ({col_sensor})', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor do Sensor')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Falha (mesmo eixo para ver a diferença de magnitude)
        axes[0].plot(t_fail[mask_fail], df_falha.loc[mask_fail, col_sensor], 
                    color='#d62728', label='Falha (Hitting Platform)', alpha=0.7, linewidth=1)
        axes[0].legend()
        
        # --- PLOT 2: COMPARAÇÃO DE DENSIDADE (KDE) ---
        # Mostra se a "forma" dos dados mudou
        sns.kdeplot(df_normal[col_sensor], ax=axes[1], color='#1f77b4', fill=True, label='Normal')
        sns.kdeplot(df_falha[col_sensor], ax=axes[1], color='#d62728', fill=True, label='Falha')
        axes[1].set_title('Mudança na Distribuição de Probabilidade', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log') # Log para ver as caudas
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def resampling_and_interpolate(self, df, df_name):
        """
        Faz resampling e interpolação dos dados para frequência fixa de 100ms
        
        Args:
            df: DataFrame com os dados
            df_name: Nome do dataset (para logging)
            
        Returns:
            DataFrame reamostrado e interpolado
        """
        print(f"\nFazendo Resampling e Interpolate de {df_name}")
        # 1. Converter tempo para Datetime (necessário para resampling)
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df = df.set_index('datetime')

        # 2. Resampling para 100ms (10Hz) - Ajuste conforme a média que vimos
        # .mean() pega todos os pontos que caíram naquele 0.1s e tira a média (reduz ruído)
        df_resampled = df.resample('100ms').mean()

        # 3. Verificar onde ficaram os buracos (NaNs gerados pelo resampling)
        print(f"Buracos gerados pelo alinhamento: {df_resampled['accX'].isnull().sum()}")

        # 4. Preencher buracos com Interpolação Linear
        # 'time' garante que a interpolação respeite a distância temporal
        df_final = df_resampled.interpolate(method='time')

        # 5. Drop nas colunas que não fazem sentido interpolar (ex: label)
        if 'label' in df_final.columns:
            df_final['label'] = df_resampled['label'].ffill().astype(int)

        print("Resampling concluído. Novo shape:", df_final.shape)
        
        return df_final
    
    def aplicar_resampling(self):
        """Aplica resampling nos datasets normal e faulty"""
        if self.df_normal is None or self.df_faulty is None:
            raise ValueError("Carregue os dados primeiro usando carregar_e_preparar_dados()")
        
        self.df_normal_resampled = self.resampling_and_interpolate(self.df_normal, "df_normal")
        self.df_faulty_resampled = self.resampling_and_interpolate(self.df_faulty, "df_faulty")
        
        return self.df_normal_resampled, self.df_faulty_resampled
    
    def _normalizar_df(self,df,fit_scaler=True,scaler=None):
        """Normaliza e retorna apenas um df"""
        if fit_scaler:
            scaler = RobustScaler()
            scaler.fit(df)
            df_scaled = scaler.transform(df)
        else:
            df_scaled = scaler.transform(df) #type:ignore
        return df_scaled,scaler

    def aplicar_normalizacao(self):
        """Aplica normalização RobustScaler e StandardScaler nos dados reamostrados"""
        if self.df_normal_resampled is None or self.df_faulty_resampled is None:
            raise ValueError("Execute o resampling primeiro usando aplicar_resampling()")
        
        # RobustScaler
        self.scaler_robust = RobustScaler()
        self.scaler_robust.fit(self.df_normal_resampled)
        
        self.df_normal_robust_scaled = self.scaler_robust.transform(self.df_normal_resampled)
        self.df_faulty_robust_scaled = self.scaler_robust.transform(self.df_faulty_resampled)
        
        # StandardScaler
        self.scaler_standard = StandardScaler()
        self.scaler_standard.fit(self.df_normal_resampled)
        
        self.df_normal_standard_scaled = self.scaler_standard.transform(self.df_normal_resampled)
        self.df_faulty_standard_scaled = self.scaler_standard.transform(self.df_faulty_resampled)
        
        print("\nNormalização concluída com RobustScaler e StandardScaler")
        
        return (self.df_normal_robust_scaled, self.df_faulty_robust_scaled, 
                self.df_normal_standard_scaled, self.df_faulty_standard_scaled)
    
    def diagnostico_preprocessamento(self, df_raw, df_resampled, df_scaled, 
                                     col_sensor='accZ', 
                                     window_sec=1.0, 
                                     scaler_name='RobustScaler'):
        """
        Gera um relatório visual comparando os estágios de pré-processamento.
        
        Args:
            df_raw: DataFrame original (Bruto).
            df_resampled: DataFrame após resampling e interpolação.
            df_scaled: DataFrame (ou array) após aplicação do Scaler.
            col_sensor: Nome da coluna do sensor para focar a análise.
            window_sec: Janela de tempo (em segundos) para o zoom do resampling.
            scaler_name: Nome do scaler usado (apenas para título).
        """
        
        # Configurar a figura
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        
        # ====================================================================
        # PARTE 1: EFEITO DO RESAMPLING (ZOOM NO TEMPO)
        # ====================================================================
        ax1 = fig.add_subplot(gs[0, :])
        
        # Preparar dados de tempo para plotagem
        # Assumindo que df_raw['time'] é int (ns) e df_resampled index é datetime
        if 'time' in df_raw.columns:
            t_raw = (df_raw['time'] - df_raw['time'].iloc[0]) / 1e3
        else:
            # Tenta usar o índice se não tiver coluna time
            t_raw = np.arange(len(df_raw)) 
            
        t_res = (df_resampled.index - df_resampled.index[0]).total_seconds()
        
        # Recorte (Zoom) para ver os detalhes
        # Pegamos apenas os primeiros 'window_sec' segundos
        mask_raw = t_raw <= window_sec
        mask_res = t_res <= window_sec
        
        # Plotar pontos originais (Scatter para mostrar o Jitter/Irregularidade)
        ax1.scatter(t_raw[mask_raw], df_raw.loc[mask_raw, col_sensor], 
                    color='black', alpha=0.6, s=30, label='Original (Raw Points)', zorder=3)
        
        # Plotar linha reamostrada (Linha + X para mostrar a grade fixa)
        ax1.plot(t_res[mask_res], df_resampled.loc[mask_res, col_sensor], 
                 color='#1f77b4', linewidth=2, marker='x', markersize=8, 
                 label='Resampled (10Hz Grid)', alpha=0.8, zorder=2)
        
        ax1.set_title(f'1. Efeito do Resampling: Regularização do Tempo ({col_sensor}) - Zoom de {window_sec}s', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tempo (segundos)')
        ax1.set_ylabel('Valor do Sensor (Físico)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # ====================================================================
        # PARTE 2: EFEITO DO SCALER (DISTRIBUIÇÃO)
        # ====================================================================
        
        # Preparar o df_scaled se ele for um numpy array (saída comum do sklearn)
        if isinstance(df_scaled, np.ndarray):
            # Tenta encontrar o índice da coluna se for array
            try:
                col_idx = df_resampled.columns.get_loc(col_sensor)
                data_scaled = df_scaled[:, col_idx]
            except:
                data_scaled = df_scaled[:, 0] # Fallback
        else:
            data_scaled = df_scaled[col_sensor]

        # Plot A: Distribuição Original (Resampled)
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(df_resampled[col_sensor], kde=True, ax=ax2, color='#1f77b4', bins=50)
        ax2.set_title(f'2a. Distribuição ANTES do Scaler\n(Unidades Físicas Reais)', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'{col_sensor} Original')
        
        # Plot B: Distribuição Escalada
        ax3 = fig.add_subplot(gs[1, 1])
        sns.histplot(data_scaled, kde=True, ax=ax3, color='#2ca02c', bins=50)
        ax3.set_title(f'2b. Distribuição DEPOIS do {scaler_name}\n(Unidades Relativas)', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'{col_sensor} Scaled')
        
        # Adicionar estatísticas de texto para comparação
        orig_mean, orig_std = df_resampled[col_sensor].mean(), df_resampled[col_sensor].std()
        scale_mean, scale_std = np.mean(data_scaled), np.std(data_scaled)
        
        txt = (f"Original:\nMédia={orig_mean:.2f}\nStd={orig_std:.2f}\nMin={df_resampled[col_sensor].min():.2f}\nMax={df_resampled[col_sensor].max():.2f}")
        ax2.text(0.95, 0.95, txt, transform=ax2.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        txt_sc = (f"Scaled:\nMédia={scale_mean:.2f}\nStd={scale_std:.2f}\nMin={np.min(data_scaled):.2f}\nMax={np.max(data_scaled):.2f}")
        ax3.text(0.95, 0.95, txt_sc, transform=ax3.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()
    
    def analise_univariada_alvo(self, df_normal=None, df_falha=None):
        """
        Realiza a análise univariada comparando atributos vs. alvo (Normal vs Falha).
        Gera estatísticas de separação e visualizações.
        
        Args:
            df_normal: DataFrame com dados normais (usa self.df_normal_resampled se None)
            df_falha: DataFrame com dados de falha (usa self.df_faulty_resampled se None)
            
        Returns:
            DataFrame com estatísticas de ranking dos sensores
        """
        if df_normal is None:
            df_normal = self.df_normal_resampled
        if df_falha is None:
            df_falha = self.df_faulty_resampled
            
        if df_normal is None or df_falha is None:
            raise ValueError("Execute o resampling primeiro ou forneça os DataFrames")
        
        # 1. Preparar dados
        # Remover colunas não-sensor (time, label, datetime, name)
        cols_ignore = ['time', 'label', 'datetime', 'name']
        sensores = [c for c in df_normal.columns if c not in cols_ignore]
        
        stats_list = []
        
        # Configuração dos Plots
        # Vamos fazer um grid de 3 colunas
        n_cols = 3
        n_rows = (len(sensores) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()
        
        print(f"Analisando {len(sensores)} sensores...\n")
        
        i = -1  # Initialize i to handle empty sensores list
        for i, sensor in enumerate(sensores):
            # Dados das duas classes
            data_norm = df_normal[sensor].dropna()
            data_fail = df_falha[sensor].dropna()
            
            # --- A. ESTATÍSTICA ---
            # 1. Diferença de Médias
            mean_diff = abs(data_fail.mean() - data_norm.mean())
            
            # 2. Razão de Variância (Quantas vezes a vibração aumentou?)
            # Adicionamos um epsilon pequeno para evitar divisão por zero
            var_ratio = data_fail.var() / (data_norm.var() + 1e-9)
            
            # 3. Teste Kolmogorov-Smirnov (Poder de Separação Geral)
            # statistic: 0 a 1 (quanto maior, melhor separa as classes)
            ks_stat, p_value = ks_2samp(data_norm, data_fail)
            
            stats_list.append({
                'Sensor': sensor,
                'KS_Statistic': ks_stat, # Principal métrica de separação
                'Variance_Ratio': var_ratio,
                'Mean_Diff': mean_diff,
                'P_Value': p_value
            })
            
            # --- B. VISUALIZAÇÃO (Violin Plot) ---
            # Criamos um mini-df temporário para o seaborn
            df_temp = pd.DataFrame({
                'Valor': np.concatenate([data_norm, data_fail]),
                'Estado': ['Normal'] * len(data_norm) + ['Falha'] * len(data_fail)
            })
            
            sns.violinplot(data=df_temp, x='Estado', y='Valor', ax=axes[i], 
                           palette={'Normal': '#1f77b4', 'Falha': '#d62728'}, split=False)
            
            axes[i].set_title(f'{sensor}\nKS Stat: {ks_stat:.3f}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].grid(True, alpha=0.3)

        # Remover axes vazios se houver
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        
        # --- C. TABELA DE RANKING ---
        df_stats = pd.DataFrame(stats_list)
        # Ordenar pelo KS Statistic (Melhor separador primeiro)
        df_stats = df_stats.sort_values(by='KS_Statistic', ascending=False).reset_index(drop=True)
        
        print("="*60)
        print("RANKING DE IMPORTÂNCIA DOS SENSORES (Baseado em KS-Test)")
        print("="*60)
        print("KS_Statistic: 1.0 = Separação Perfeita | 0.0 = Indistinguível")
        print("Variance_Ratio: > 1.0 = Falha aumentou a variabilidade")
        print("-" * 60)
        print(df_stats[['Sensor', 'KS_Statistic', 'Variance_Ratio', 'Mean_Diff']])
        
        return df_stats
    
    def plotSensors(self, dfPlot, step=1, suptitle="SENSORES AO LONGO DE 1 MINUTO\n", 
                    startTimeIdx=None, endTimeIdx=None):
        """
        Plota os sensores (acelerômetro, giroscópio e magnetômetro) ao longo do tempo
        
        Args:
            dfPlot: DataFrame com os dados
            step: Passo para amostragem dos dados
            suptitle: Título do gráfico
            startTimeIdx: Índice inicial (ou None para usar o primeiro)
            endTimeIdx: Índice final (ou None para usar o último)
        """
        df = dfPlot.iloc[::step]
        fig = plt.figure(figsize=(25, 15))

        if not any((startTimeIdx, endTimeIdx)):
            startTimeIdx, endTimeIdx = df['time'].iloc[0], df['time'].iloc[-1]
        else:
            startTimeIdx, endTimeIdx = df['time'].iloc[startTimeIdx], df['time'].iloc[endTimeIdx]

        # Helper to plot three axes in the same subplot
        def plotSensorsSameGraph(ax, cols, title, x="time"):
            for col in cols:
                mask = (df['time'] >= startTimeIdx) & (df['time'] < endTimeIdx)
                ax.plot(df[mask][x], df[mask][col], label=col)

            ax.set_title(title, fontsize=18)
            ax.set_xlabel(x)
            ax.set_ylabel("value")
            ax.legend(loc='lower left')

        # === Subplots ===
        ax1 = fig.add_subplot(3, 1, 1)
        plotSensorsSameGraph(ax1,
                             cols=["accX", "accY", "accZ"],
                             title="Accelerometer (X, Y, Z)")

        ax2 = fig.add_subplot(3, 1, 2)
        plotSensorsSameGraph(ax2,
                             cols=["gyroZ", "gyroX", "gyroY"],
                             title="Gyroscope (X, Y, Z)")

        ax3 = fig.add_subplot(3, 1, 3)
        plotSensorsSameGraph(ax3,
                             cols=["magZ", "magY", "magX"],
                             title="Magnetometer (X, Y, Z)")
        plt.suptitle(suptitle, fontsize='18')
        plt.tight_layout()
        plt.show()
    
    def aplicar_filtro_savgol(self, df, cols=None, window_length=8, polyorder=2):
        """
        Aplica o filtro Savitzky-Golay para suavizar os sinais dos sensores
        
        Args:
            df: DataFrame com os dados
            cols: Lista de colunas para aplicar o filtro (None = todas as colunas de sensores)
            window_length: Tamanho da janela do filtro
            polyorder: Ordem do polinômio
            
        Returns:
            DataFrame com as colunas suavizadas (_smooth)
        """
        df = df.copy()
        
        if cols is None:
            cols = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"]
        
        for col in cols:
            if col in df.columns:
                df[col + "_smooth"] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
        
        print(f"Filtro Savitzky-Golay aplicado em {len(cols)} colunas")
        return df
    
    def plot_raw_vs_smooth(self, df, step=10):
        """
        Plota comparação entre dados brutos e suavizados
        
        Args:
            df: DataFrame com dados brutos e suavizados (_smooth)
            step: Passo para amostragem dos dados
        """
        sensor_groups = [
            ("Accelerometer", ["accX", "accY", "accZ"]),
            ("Gyroscope", ["gyroX", "gyroY", "gyroZ"]),
            ("Magnetometer", ["magX", "magY", "magZ"])
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(32, 24))
        df = df.iloc[::step]

        for row, (title, cols) in enumerate(sensor_groups):
            for col, axis in enumerate(cols):
                ax = axes[row][col]

                if axis not in df.columns:
                    continue
                    
                raw = df[axis]
                smooth_col = axis + "_smooth"
                
                if smooth_col in df.columns:
                    smooth = df[smooth_col]
                    ax.plot(df["time"], raw, label="raw", alpha=0.35)
                    ax.plot(df["time"], smooth, label="smooth", linewidth=2)
                else:
                    ax.plot(df["time"], raw, label="raw")

                ax.set_title(f"{title} — {axis}", fontsize=16)
                ax.set_xlabel("time")
                ax.set_ylabel("value")
                ax.legend()

        plt.tight_layout()
        plt.show()
    
    def executar_pipeline_completo(self, mostrar_diagnostico=True, mostrar_analise_univariada=True,
                                   mostrar_sensores=False, mostrar_savgol=False, aplicar_filtro=False):
        """
        Executa o pipeline completo de pré-processamento e análise
        
        Args:
            mostrar_diagnostico: Se True, mostra diagnóstico de pré-processamento
            mostrar_analise_univariada: Se True, executa análise univariada
            mostrar_sensores: Se True, plota visualização dos sensores
            aplicar_filtro: Se True, aplica filtro Savitzky-Golay
            
        Returns:
            dict com os resultados do pipeline
        """
        if self.df_normal is None or self.df_faulty is None:
            raise ValueError("Carregue os dados primeiro usando carregar_e_preparar_dados()")
        
        resultados = {}
        
        # 1. Aplicar resampling
        print("\n" + "="*60)
        print("ETAPA 1: RESAMPLING")
        print("="*60)
        self.aplicar_resampling()
        resultados['df_normal_resampled'] = self.df_normal_resampled
        resultados['df_faulty_resampled'] = self.df_faulty_resampled
        
        # 2. Aplicar normalização
        print("\n" + "="*60)
        print("ETAPA 2: NORMALIZAÇÃO")
        print("="*60)
        scaled_data = self.aplicar_normalizacao()
        resultados['scaled_data'] = {
            'robust': (self.df_normal_robust_scaled, self.df_faulty_robust_scaled),
            'standard': (self.df_normal_standard_scaled, self.df_faulty_standard_scaled)
        }
        
        # 3. Diagnóstico de pré-processamento
        if mostrar_diagnostico:
            print("\n" + "="*60)
            print("ETAPA 3: DIAGNÓSTICO DE PRÉ-PROCESSAMENTO")
            print("="*60)
            print("\nDiagnóstico com RobustScaler:")
            self.diagnostico_preprocessamento(
                self.df_normal, 
                self.df_normal_resampled, 
                self.df_normal_robust_scaled, 
                col_sensor='accZ',
                scaler_name='RobustScaler'
            )
            print("\nDiagnóstico com StandardScaler:")
            self.diagnostico_preprocessamento(
                self.df_normal, 
                self.df_normal_resampled, 
                self.df_normal_standard_scaled, 
                col_sensor='accZ',
                scaler_name='StandardScaler'
            )
        
        # 4. Análise univariada
        if mostrar_analise_univariada:
            print("\n" + "="*60)
            print("ETAPA 4: ANÁLISE UNIVARIADA")
            print("="*60)
            df_stats = self.analise_univariada_alvo()
            resultados['stats_univariada'] = df_stats
        
        # 5. Visualização de sensores
        if mostrar_sensores:
            print("\n" + "="*60)
            print("ETAPA 5: VISUALIZAÇÃO DE SENSORES")
            print("="*60)
            self.plotSensors(self.df_normal, startTimeIdx=20050, endTimeIdx=20050 + 60*10)
        
        # 6. Aplicar filtro Savitzky-Golay
        if aplicar_filtro:
            print("\n" + "="*60)
            print("ETAPA 6: APLICAÇÃO DE FILTRO SAVITZKY-GOLAY")
            print("="*60)
            df_filtered = self.aplicar_filtro_savgol(self.df_normal)
            if mostrar_savgol: self.plot_raw_vs_smooth(df_filtered)
            resultados['df_filtered'] = df_filtered
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETO FINALIZADO!")
        print("="*60)
        
        return resultados


# Exemplo de uso
# if __name__ == "__main__":
#     # Instancia a classe EDA
#     eda = EDA()
    
#     # Carrega e prepara os dados
#     df_normal, df_faulty = eda.carregar_e_preparar_dados()
    
#     # Executa a análise exploratória completa
#     eda.executar_analise_completa()
    
#     # Pipeline completo de pré-processamento (método integrado na classe)
#     resultados = eda.executar_pipeline_completo(
#         mostrar_diagnostico=True,
#         mostrar_analise_univariada=True,
#         mostrar_sensores=True,
#         aplicar_filtro=True
#     )
    
#     print(resultados)
    
    # OU executar etapas individuais:
    # eda.aplicar_resampling()
    # eda.aplicar_normalizacao()
    # eda.diagnostico_preprocessamento(eda.df_normal, eda.df_normal_resampled, 
    #                                   eda.df_normal_robust_scaled, col_sensor='accZ')
    # df_stats = eda.analise_univariada_alvo()
    # eda.plotSensors(eda.df_normal, startTimeIdx=20050, endTimeIdx=20050 + 60*10)
    # df_filtered = eda.aplicar_filtro_savgol(eda.df_normal)
    # eda.plot_raw_vs_smooth(df_filtered)

