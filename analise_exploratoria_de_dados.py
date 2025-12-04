# %%
def EDA(df, faultydf):
    print("""## Análise Exploratória Estrutural [Classe Normal]
        - Informações básicas do dataset
        - Tipos de dados
        - Informações detalhadas
        - Estatísticas descritivas
        - Análise de valores únicos""")
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    df.head()

    print("\nConvertendo `time` de nanossegundos para milissegundos para facilitar a análise de integridade temporal")
    df['time'] = (df['time'].map(lambda x: x/1e6))
    faultydf['time'] = faultydf['time'].map(lambda x: x/1e6)

    print("\nRemovendo a coluna `name` pois não agrega valor preditivo")
    df = df.drop(columns=['name'])

    print("\nIdentificando duplicatas")
    print(f"Duplicatas de Tempo: {df['time'].duplicated().any()}")

    def analise_estrutural_sensores(df, time_col='time', label_col='label'):
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

    analise_estrutural_sensores(df, time_col='time', label_col='label')
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

    def plot_sensor(df, col_sensor, col_time='time'):
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

    print("\nVisualizando `gyroY`:")
    plot_sensor(df, col_sensor='gyroY')
    print("O boxplot mostra que não deve-se remover os outliers pela regra de desvio padrão (3-sigma), pois serão apagados o movimento do robô sobrando só o ruído dele padrado")

    # TODO
    """ Montar EDA para faultydf; 
        verificar se gyroY de falha tem picos diferentes
        comparar magZ em df e faultydf (se forem iguais, ele é irrelevante para detecção e pode ser descartado)
        ambos precisam ter a mesma frequência para entrar no modelo
        verificar se possui os mesmos defeitos físicos (jitter, gaps) """

    def comparar_normal_vs_falha(df_normal, df_falha, col_sensor='gyroY'):
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

    if 'name' in faultydf.columns: faultydf = faultydf.drop(columns=['name'])
    
    comparar_normal_vs_falha(df, faultydf, col_sensor='gyroY')

    #TODO
    """"O que procurar neste novo gráfico:

    Mudança de Amplitude: A linha vermelha (falha) atinge picos muito maiores que 50? (Indicaria impacto forte).

    Mudança de Frequência: A linha vermelha oscila muito mais rápido ou fica "travada" em valores estranhos?

    Deslocamento de Distribuição (Gráfico de baixo): Se a curva vermelha estiver deslocada para a direita ou esquerda em relação à azul, significa que a média mudou (drift). Se a curva vermelha for "mais gorda", a variância (vibração) aumentou.

    Essas pistas visuais vão ditar quais features seu modelo precisa (ex: se a amplitude mudou, max/min são boas features. Se a frequência mudou, FFT é melhor)."""""

    print("""\n## Análise de Valores Faltantes e Outliers
        - Identificação de valores faltantes e outliers
        - Visualizações de apoio, caso necessário
        - Análise dos mecanismos\n""")

    print(f"Quantidade de valores faltantes em df: {df.isnull().sum().sum()}")
    print(f"Quantidade de Valores faltantes em faultydf: {faultydf.isnull().sum().sum()}")

    print(f"\nNão serão removidos os outliers estatísticos pois são movimentos reais")
#%%