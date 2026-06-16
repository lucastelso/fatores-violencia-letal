#%% Importando os pacotes
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import seaborn as sns

import pandas as pd
import numpy as np

from utilitarios.visualizacao import EstatisticasDescritivas
sns.set(style="whitegrid", context="notebook", font_scale=1.2)

#%% carregando os dados (Path evita conflitos entre "/" e "\")
df_unificado_filtrado = pd.read_parquet(Path(r'dados_brutos/base_de_dados_filtrada.parquet'))
base_taxas_de_violencia_2010_2022= pd.read_parquet(Path(r'dados_brutos/taxas_de_violencia_2008_2022.parquet'))
base_taxas_de_violencia_2010_2022_filtrada = pd.read_parquet(Path(r'dados_brutos/taxas_de_violencia_2008_2022_filtrada.parquet'))

# %% Tratando os dados
df_unificado_filtrado.columns = [col.lower() for col in df_unificado_filtrado.columns]
base_taxas_de_violencia_2010_2022.columns = [col.lower() for col in base_taxas_de_violencia_2010_2022.columns]
base_taxas_de_violencia_2010_2022_filtrada.columns = [col.lower() for col in base_taxas_de_violencia_2010_2022_filtrada.columns]

#%% Instanciando a classe para estatisticas descritivas
ed_bd_filtr = EstatisticasDescritivas(df_unificado_filtrado)


#%% Estatisticas das taxas de violencia ao longo do tempo
base_taxas_de_violencia_2010_2022_filtrada.iloc[:,5:].describe().round(3)


#%% 

df_plot = base_taxas_de_violencia_2010_2022_filtrada.iloc[:, 7:-1].melt(var_name='Ano', value_name='Taxa')
df_plot_ed_ano = EstatisticasDescritivas(df_plot)
df_plot_ed_ano.plot_box_uf(
    titulo='Distribuição das Taxas de Violencia Letal por ano',
    y_label='taxa',
    x='Ano', 
    y='Taxa',
    hue='Ano'
    ) 


#%% Histograma da taxa de 2010

# Tamanho da figura
plt.figure(figsize=(12, 6))

# Cor azul escuro
azul_escuro = "#1f4e79"

# Histograma com azul escuro
ax = sns.histplot(
    data=df_unificado_filtrado,
    x='taxa_violencia',
    bins=30,
    kde=True,
    color=azul_escuro,
    edgecolor='black',
    alpha=0.85
)

# Título e rótulos
plt.title("Distribuição da Taxa de Mortes Violentas Intencionais em 2010", fontsize=18, weight='bold')
plt.xlabel("Taxa de Violência", fontsize=14)
plt.ylabel("Frequência", fontsize=14)

ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{int(y):,}'))

# Personalizando spines (bordas)
ax = plt.gca()  # pega o eixo atual
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1)    # controla grossura
ax.spines['bottom'].set_linewidth(1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(False)

# Layout final
plt.tight_layout()
plt.show()


#%% Violencia letal por região brasileira

ed_bd_filtr.plot_box_uf('Taxa de Mortes Violêntas Intencionais em 2010, segundo Região brasileira\n', 
            'Taxa de Mortes Violentas Intencionais\npor 100.000 habitantes', 
            x='regiao',
            y='taxa_violencia',
            hue='regiao'
            )

#%% Violencia letal por UF

ed_bd_filtr.plot_box_uf(
    titulo='Taxa de mortes violentas intencionais em 2010 segundo UF',
    y_label='Taxa de mortes violentas intencionais \npor 100.000 habitantes',
    x='uf',
    y='taxa_violencia'
)

plt.show()
df_unificado_filtrado['taxa_violencia'].describe().round(3)

#%%
"""
A partir daqui são boxplots das variáveis de interesse, divididas por UF
"""

ed_bd_filtr.plot_box_uf(
    titulo='Índice de Desenvolvimento Humano Municipal em 2010 segundo UF\n',
    y_label='Indice de Desenvolvimento Humano Municipal',
    x='uf',
    y='idhm_2010'
)
plt.show()

#%%

ed_bd_filtr.plot_box_uf(
    titulo='Proporção de Pessoas Vulneráveis à Pobreza em 2010 segundo UF\n',
    y_label='Indice de GINI',
    x='uf',
    y='prop_vulneraveis_pobreza'
)
plt.show()

#%%

ed_bd_filtr.plot_box_uf(
    titulo='Índice de Desenvolvimento Humano Municiapal em 2010 segundo UF\n',
    y_label='Percentual de residências com água\ne rede de esgoto inadequadas',
    x='uf',
    y='agua-esgotamento_inadequados'
)
plt.show()

#%%
ed_bd_filtr.plot_box_uf(
    titulo='Percentual de desempregados com 18 anos ou mais em 2010 segundo UF\n',
    y_label='Percentual de desempregados\ncom 18 anos ou mais',
    x='uf',
    y='taxa_desocup_18_mais'
)
plt.show()

#%%
ed_bd_filtr.plot_box_uf(
    titulo='Proporção de pessoas vulneráveis a pobreza em 2010 segundo UF\n',
    y_label='Proporção de pessoas vulneráveis a pobreza',
    x='uf',
    y='prop_vulneraveis_pobreza'
)
plt.show()

#%%
ed_bd_filtr.plot_box_uf(
    titulo='Índice de Desenvolvimento Humano Municiapal em 2010 segundo UF\n',
    y_label='Proporção de Analfabetos\ncom 15 anos ou mais',
    x='uf',
    y='analfabetismo_15_anos'
)
plt.show()

#%%
ed_bd_filtr.plot_box_uf(
    titulo='Proporção de desempregados adultos em 2010 segundo UF\n',
    y_label='Proporção de desempregados\n com 18 anos ou mais',
    x='uf',
    y='taxa_desocup_18_mais',
)
plt.show()


# FIM! :-)