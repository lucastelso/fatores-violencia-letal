#%% Importando pacotes
import pandas as pd
from utilitarios.visualizacao import (
    DiagnosticoMulticolinearidade, 
    EstatisticasDescritivas, 
    VisualizadorExploratorio
)

from utilitarios.pca import ExtratorPCATematico


df_unificado_filtrado = pd.read_parquet(r'dados_brutos/base_de_dados_filtrada.parquet')
df_unificado_filtrado.set_index('COD', inplace=True)


#%% Realizando o diagnóstico de multicolinearidade

variaveis = [
    "POP_TOTAL_RESIDENTE", "POP_URBANA_RESIDENTE", "ENERGIA_ELETRICA", "GINI_2010","THEIL_2010",
    "ENS_FUND_25_ANOS_OU_MAIS", "ENS_MEDIO_25_ANOS_OU_MAIS", "COLETA_LIXO", "POP_AGUA_ENCANADA",
    "REDE_ESGOTO","AGUA-ESGOTAMENTO_INADEQUADOS","POP_ECON_ATIVA","PROP_POBRES",
    "PROP_VULNERAVEIS_POBREZA","RENDA_PERCAPITA_5_MAIS_POBRE",
    "PROP_EXTREM_POBRES","TAXA_DESOCUP_18_MAIS","ANALFABETISMO_15_ANOS",
]

diag_multicolinearidade = DiagnosticoMulticolinearidade(
    df=df_unificado_filtrado, 
    features=variaveis
    )

diag_multicolinearidade.plotar_matriz_correlacao(decimais=1)
diag_multicolinearidade.calcular_vif()


#%% Definição temática das variáveis
"""
Na célula seguinte, as variáveis originais são transformadas 
em fatores ortogonais a partir da análise de componentes principais. 
Não se trata de uma transformação global, mas de uma transformação temática, 
de modo que as variáveis do mesmo tema são agrupadas em um unico fator
"""

df_unificado_filtrado.info()
# %% TRANSFORMAÇÃO DOS COMPONENTES

temas = {
    'POPULACAO': [
        'POP_TOTAL_RESIDENTE', "POP_URBANA_RESIDENTE"
    ],
    'EDUCACAO': [
        'ANALFABETISMO_15_ANOS','ENS_FUND_25_ANOS_OU_MAIS', 'ENS_MEDIO_25_ANOS_OU_MAIS'
    ],
    'INFRAESTRUTURA': [
        'COLETA_LIXO','ENERGIA_ELETRICA','AGUA-ESGOTAMENTO_INADEQUADOS','POP_AGUA_ENCANADA','REDE_ESGOTO'
    ],
    'POBREZA_DESIGUALDADE': [
        'PROP_POBRES','PROP_VULNERAVEIS_POBREZA',"PROP_EXTREM_POBRES",'THEIL_2010', 'GINI_2010', "RENDA_PERCAPITA_5_MAIS_POBRE"
    ],

    'EMPREGO': [
        'TAXA_DESOCUP_18_MAIS','POP_ECON_ATIVA'
    ]
}

extrator = ExtratorPCATematico(
    dicionario_temas=temas, 
    temas_inverter_sinal=['INFRAESTRUTURA', 'EMPREGO']
    )

df_pca_completo = extrator.fit_transform(df_unificado_filtrado)
extrator.exibir_relatorio()

#%%
colunas_finais = ['TAXA_VIOLENCIA'] + [f'FATOR_{t}' for t in temas.keys()]
df_final_modelagem = df_pca_completo[colunas_finais]

df_final_modelagem

#%% REFAZENDO O DIAGNÓSTICO DE MULTICOLINEARIDADE


diag_multicolinearidade_fatores = DiagnosticoMulticolinearidade(
    df=df_final_modelagem,
     features=df_final_modelagem.columns.to_list())

diag_multicolinearidade_fatores.plotar_matriz_correlacao(2)
diag_multicolinearidade_fatores.calcular_vif()


#%%  
"""
A partir da celular de cima, o diagnóstico já está completo 
e o DataFrame final para modelagem estatística está pronto.

As celulas que se seguem são visualizações dos novos dados
"""

#%% Análise Univariada

variavel_alvo = 'TAXA_VIOLENCIA'
fatores_extraidos = [col for col in df_final_modelagem.columns if col != variavel_alvo]
variaveis_distribuicao = fatores_extraidos + [variavel_alvo]

# Instancia o visualizador com o DF processado pela nossa classe PCA
viz = VisualizadorExploratorio(df_pca_completo)

# Gera os plots de forma completamente agnóstica à quantidade de variáveis
viz.plotar_distribuicoes(variaveis=variaveis_distribuicao, hue='REGIAO', max_colunas=3)
viz.plotar_relacionamentos(preditores=fatores_extraidos, alvo=variavel_alvo, hue='REGIAO', max_colunas=3)



#%% Visualização dos novos fatores, agora por unidade federativa

# Instanciando o visualizador
ed = EstatisticasDescritivas(df_pca_completo)

ed.plot_box_uf(
    "FATOR_EMPREGO", 
    "FATOR_EMPREGO", 
    x='UF',
    y='FATOR_EMPREGO',
    hue='UF')

#%%
ed.plot_box_uf(
    "FATOR_EDUCACAO", 
    "EDUCAÇAO", 
    x='UF',
    y='FATOR_EDUCACAO',
    hue='UF')
#%%
ed.plot_box_uf(
    "FATOR_POBREZA_DESIGUALDADE", 
    "FATOR_POBREZA_DESIGUALDADE", 
    x='UF',
    y='FATOR_POBREZA_DESIGUALDADE',
    hue='UF')
#%%
ed.plot_box_uf(
    "FATOR_POPULACAO", 
    "POPULAÇÃO",
    x='UF',
    y='FATOR_POPULACAO',
    hue='UF'
    )
#%%
ed.plot_box_uf(
    "FATOR INFRAESTRUTURA", 
    "FATOR_INFRAESTURUTRA", 
    x='UF',
    y='FATOR_INFRAESTRUTURA',
    hue='UF'
    )
#%%
ed.plot_box_uf(
    "Emprego", 
    "Fator Emprego", 
    x='UF',
    y='FATOR_EMPREGO',
    hue='UF'
    )

# FIM! :-) 