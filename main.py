"""
ADICIONAR:
    1. Visualizar dados brutos;
    2. Transformar dados brutos em fatores;
    3. Visualizar fatores
    4. Modelos paramétricos
    5. Modelo não-paramétrico
    
"""
import pandas as pd
from utilitarios.outros_utils import CustomFormatter
from models import AvaliadorRegressaoParametrica, DiagnosticadorMultinivel
from scipy.stats import boxcox
import statsmodels.api as sm

logger = CustomFormatter.get_logger("PIPELINE")

logger.info('INICIANDO PIPELINE!')

# CARREGANDO OS DADOS

path_dados = r"dados_brutos/fatores_pca.parquet"
logger.info(f"Lendo dados: {path_dados}")
fatores = pd.read_parquet(path=path_dados)
logger.info("Dados encontrados com sucesso!")


logger.warning("VERIFIQUE A INTEGRIDADE DOS DADOS:")

print(fatores.sample(10))

choice = input(logger.warning('Os dados estão integros?[Y/n]'))

while choice not in ['Y', 'N', 'y', 'n']:
    choice = input(logger.error("Responda com 'Y' para 'SIM' ou 'n' para 'NAO"))

if choice in ['N', 'n']:
    raise logger.error('PIPELINE ENCERRADO, DADOS NÃO ESTÃO INTEGROS')

logger.info('Continuando...')
logger.warning("Fazendo calculo da taxa de violencia em função de indicadores socioeconômicos")

formula_bc = (
    'TAXA_VIOLENCIA ~ FATOR_POBREZA_DESIGUALDADE + FATOR_INFRAESTRUTURA + '
    'FATOR_EDUCACAO + FATOR_EMPREGO + FATOR_POPULACAO'
)

fatores_bc_rlmm = fatores.copy()
avaliador_param = AvaliadorRegressaoParametrica()
diag_hlm = DiagnosticadorMultinivel()

fatores_bc_rlmm['TAXA_VIOLENCIA'], lambda_bc = boxcox(fatores_bc_rlmm['TAXA_VIOLENCIA'] + 1)

modelo_bc_2niveis = sm.MixedLM.from_formula(
    formula=formula_bc,
    data=fatores_bc_rlmm,
    groups=fatores["UF"],
    re_formula='1 + FATOR_EMPREGO'
)

resultado_bc_2niveis = modelo_bc_2niveis.fit()



y_predito_bc = resultado_bc_2niveis.fittedvalues.to_numpy()
residuos_bc = resultado_bc_2niveis.resid.to_numpy()
print(avaliador_param.diagnostico_quantitativo(
    residuos_bc, y_predito_bc, 
    nome_contexto="Modelo Hierárquico (Nível 1)")
    )

print(f"R² de Nakagawa:{diag_hlm.r2_nakagawa_schielzeth(resultado_bc_2niveis)}")
print(avaliador_param.plotar_painel_residuos(residuos_bc, y_predito_bc, titulo="Multinível Box-Cox (Nível 1)"))
print(avaliador_param.plotar_efeitos_aleatorios_multinivel(resultado_bc_2niveis.random_effects))
print(resultado_bc_2niveis.summary())