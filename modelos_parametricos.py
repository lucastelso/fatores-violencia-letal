
#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import boxcox
from models import AvaliadorRegressaoParametrica, DiagnosticadorMultinivel
from utilitarios.outros_utils import criar_path

#%%

df_PCA = pd.read_parquet(criar_path("fatores_pca.parquet"))
df_unificado_filtrado = pd.read_parquet(criar_path("base_de_dados_filtrada.parquet"))
base_taxas_de_violencia_2010_2022= pd.read_parquet(criar_path('taxas_de_violencia_2008_2022.parquet'))
base_taxas_de_violencia_2010_2022_filtrada = pd.read_parquet(criar_path('taxas_de_violencia_2008_2022_filtrada.parquet'))

#%% DEFININDO O x e o y dos modelos e intanciando o avaliador do modelos
X = df_PCA.iloc[:,1:6]
y = df_PCA.iloc[:,0]


#%% Instancia o Avaliador (Alpha = 0.05)
avaliador_param = AvaliadorRegressaoParametrica(alpha=0.05)
df_PCA_parametrico = df_PCA.copy()

formula_rlm = "TAXA_VIOLENCIA ~ FATOR_POBREZA_DESIGUALDADE + FATOR_INFRAESTRUTURA + FATOR_EDUCACAO + FATOR_EMPREGO + FATOR_POPULACAO"


#%% Instanciar todos os modelos 

# Regressão Linear Multipla
rlm = sm.OLS.from_formula(
    formula=formula_rlm,
    data=df_PCA_parametrico
).fit()

# Regressão Linear Multipla com transformação logaritimica da variável dependente
df_PCA_parametrico_log1 = df_PCA.copy()
df_PCA_parametrico_log1['TAXA_VIOLENCIA'] = np.log1p(df_PCA_parametrico['TAXA_VIOLENCIA'])

log_rlm = sm.OLS.from_formula(
    formula=formula_rlm,
    data=df_PCA_parametrico_log1
).fit()

# Regressão Linear Multipla com transformações de box-cox
df_PCA_parametrico_box = df_PCA.copy()
df_PCA_parametrico_box['TAXA_VIOLENCIA'], lambda_bc = boxcox(df_PCA_parametrico_box['TAXA_VIOLENCIA'] + 1) # +1 para evitar div por 0

# Modelo OLS com fórmula
rlm_bc = sm.OLS.from_formula(
    formula=formula_rlm,
    data=df_PCA_parametrico_box
).fit()

# Regressão Linear Múltipla Multinível com Transformações de Box-Cox
formula_bc = (
    'TAXA_VIOLENCIA ~ FATOR_POBREZA_DESIGUALDADE + FATOR_INFRAESTRUTURA + '
    'FATOR_EDUCACAO + FATOR_EMPREGO + FATOR_POPULACAO'
)

modelo_bc_2niveis = sm.MixedLM.from_formula(
    formula=formula_bc,
    data=df_PCA_parametrico_box,
    groups=df_PCA_parametrico_box["UF"],
    re_formula='1 + FATOR_EMPREGO'
)
resultado_bc_2niveis = modelo_bc_2niveis.fit()


#%% 1. REGRESSÃO LINEAR MÚLTIPLA

y_real = df_PCA_parametrico['TAXA_VIOLENCIA'].to_numpy()
y_predito_ols = rlm.fittedvalues.to_numpy()
residuos_ols = rlm.resid.to_numpy()
exog_ols = rlm.model.exog # Matriz de variáveis independentes original


avaliador_param.diagnostico_quantitativo(
    residuos_ols, 
    y_predito_ols, 
    exog=exog_ols, 
    nome_contexto="Regressão Linear Múltipla"
    )

avaliador_param.plotar_previsao_vs_real(
    y_real, 
    y_predito_ols, 
    hue_labels=df_PCA_parametrico['REGIAO']
    )

avaliador_param.plotar_painel_residuos(
    residuos_ols, 
    y_predito_ols, 
    titulo="OLS Padrão"
    )

print(rlm.summary())


#%% 2. REGRESSÃO LINEAR MÚLTIPLA COM TRANSFORMAÇÃO LOGARÍTIMA

y_real = df_PCA_parametrico_log1['TAXA_VIOLENCIA'].to_numpy()
y_predito_ols = log_rlm.fittedvalues.to_numpy()
residuos_ols = log_rlm.resid.to_numpy()
exog_ols = log_rlm.model.exog # Matriz de variáveis independentes original


avaliador_param.diagnostico_quantitativo(
    residuos_ols, 
    y_predito_ols, 
    exog=exog_ols, 
    nome_contexto="Regressão Linear Múltipla com transformação logaritimica"
    )

avaliador_param.plotar_previsao_vs_real(
    y_real, 
    y_predito_ols, 
    hue_labels=df_PCA_parametrico_log1['REGIAO'], 
    titulo='Regressão Linear Múltipla com Transformação Logaritimica'
    )

avaliador_param.plotar_painel_residuos(
    residuos_ols, 
    y_predito_ols, 
    titulo="Regressão Linear Múltipla com Transformação Logaritimica"
    )


#%% 3.REGRESSÃO LINEAR MÚLTIPLA COM TRANSFORMAÇÕES DE BOX-COX

y_real = df_PCA_parametrico_box['TAXA_VIOLENCIA'].to_numpy()
y_predito_ols = rlm_bc.fittedvalues.to_numpy()
residuos_ols = rlm_bc.resid.to_numpy()
exog_ols = rlm_bc.model.exog # Matriz de variáveis independentes original


avaliador_param.diagnostico_quantitativo(
    residuos_ols, 
    y_predito_ols, 
    exog=exog_ols, 
    nome_contexto="Regressão Linear Múltipla com transformação logaritimica"
    )

avaliador_param.plotar_previsao_vs_real(
    y_real, 
    y_predito_ols, 
    hue_labels=df_PCA_parametrico_box['REGIAO'], 
    titulo='Regressão Linear Múltipla com Transformações de Box-Cox'
    )

avaliador_param.plotar_painel_residuos(
    residuos_ols, 
    y_predito_ols, 
    titulo="Regressão Linear Múltipla com Transformações de Box-Cox"
    )

#%% 4. REGRESSÃO LINEAR MÚLTIPLA MULTINÍVEL COM TRANSFORMAÇÕES DE BOX-COX

diag_hlm = DiagnosticadorMultinivel()
diag_hlm.renderizar_previsao_boxcox(
    modelo_bc_ajustado=resultado_bc_2niveis,
    df_completo=df_PCA,
    coluna_alvo='TAXA_VIOLENCIA',
    coluna_estrato='REGIAO',
    lambda_bc=lambda_bc
    )

#%% Segunda parte da RLMM-BC

y_predito_bc = resultado_bc_2niveis.fittedvalues.to_numpy()
residuos_bc = resultado_bc_2niveis.resid.to_numpy()
avaliador_param.diagnostico_quantitativo(residuos_bc, y_predito_bc, nome_contexto="Modelo Hierárquico (Nível 1)")
avaliador_param.plotar_painel_residuos(residuos_bc, y_predito_bc, titulo="Multinível Box-Cox (Nível 1)")
# avaliador_param.plotar_efeitos_aleatorios_multinivel(resultado_bc_2niveis.random_effects)
diag_hlm.diagnostico_efeitos_aleatorios_blups(modelo_ajustado=resultado_bc_2niveis, alpha=0.05)

#%% Resultados do RLMM-BC

print(resultado_bc_2niveis.summary())