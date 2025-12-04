#%%
from coleta_de_dados import coletar_dados
from analise_exploratoria_de_dados import EDA
!pip install kagglehub

# %%
df, faultydf = coletar_dados()

#%%
EDA(df, faultydf)

#%%
