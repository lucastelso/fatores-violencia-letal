import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import Literal

ColunasDataset = Literal[
    'cod','regiao','uf', 'municipio', 'pop_total_residente', 'pop_rural_residente',
    'pop_urbana_residente','pop_homem_residente', 'pop_mulher_residente', 'pop_cor_branca',
    'pop_cor_negra','idhm_2010', 'gini_2010','theil_2010','esperanca_vida_nascer','analfabetismo_15_anos',
    'ens_fund_25_anos_ou_mais','ens_medio_25_anos_ou_mais','ens_superior_25_anos_ou_mais','pop_agua_encanada',
    'agua-esgotamento_inadequados', 'coleta_lixo','rede_esgoto','energia_eletrica', 'renda_percapita',
    'renda_percapita_5_mais_pobre', 'renda_percapita_2°_quinto_mais_pobre','renda_percapita_3°_quinto_mais_pobre',
    'renda_percapita_4°_quinto_mais_pobre','prop_extrem_pobres','prop_pobres','prop_vulneraveis_pobreza',
    'renda_top_10_porcento_maior_renda','renda_40_porcento_mais_pobre', 'homicidios_datasus','homicidios_feminino',
    'homicidios_masculino','homicidios_15_29_anos_datasus','homicidios_masculino_15_29_anos','pop_econ_ativa',
    'formal_trab','pop_ocup_fund_completo','pop_ocup_medio_completo','taxa_desocup_18_mais','taxa_violencia'
    ]

class EstatisticasDescritivas:
    """
    Classe para encapsular Análise Exploratória de Dados.
    Mantém o DataFrame como estado interno para evitar repetição de passagem de parâmetros.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # self._validar_schema()
    """        
    def _validar_schema(self) -> None:
        "Método privado para garantir que o DF instanciado tem as colunas básicas esperadas."
        colunas_necessarias = ['taxa_violencia'] # Exemplo mínimo
        faltantes = [col for col in colunas_necessarias if col not in self.df.columns]
        if faltantes:
            raise ValueError(f"O DataFrame instanciado não possui as colunas obrigatórias: {faltantes}")
    """
    def plot_box_uf(
        self,
        titulo: str, 
        y_label: str, 
        x: ColunasDataset, 
        y: ColunasDataset, 
        hue: ColunasDataset = 'uf',
        palette: str = 'viridis'
    ) -> None:
        """
        Gera um boxplot customizado usando a interface Orientada a Objetos do Matplotlib.
        """
        # Validação em runtime: garante que a string passada realmente existe no DF atual
        # for col in [x, y, hue]:
        #    if col not in self.df.columns:
        #        raise KeyError(f"A coluna '{col}' não foi encontrada no DataFrame.")

        # Interface Orientada a Objetos (Melhor prática / Evita memory leak)
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Gerando o plot no eixo (ax) específico
        sns.boxplot(
            data=self.df, 
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            ax=ax
        )
        
        # Customização do eixo (Substituindo o estado global plt.title, plt.ylabel, etc)
        ax.set_title(titulo, fontsize=15, pad=15)
        ax.set_ylabel(y_label)
        ax.set_xlabel("") # Intencionalmente vazio conforme seu design original
        
        # Ajuste de Spines (Bordas)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1)

        # Ajuste inteligente de Grid: Mostrar apenas linhas horizontais (facilita leitura de boxplots)
        ax.yaxis.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
        ax.set_axisbelow(True) # Garante que o grid fique atrás dos boxplots
        
        # plt.show() libera a figura e entrega para a interface
        plt.show()


class DiagnosticoMulticolinearidade:
    """
    Classe para diagnóstico visual e quantitativo de multicolinearidade.
    Ideal para inspeção antes de modelagens lineares.
    """
    
    def __init__(self, df: pd.DataFrame, features: list):
        self.df = df[features].dropna() # Cópia limpa para diagnóstico
        self.features = features

    def plotar_matriz_correlacao(self, decimais:int, titulo: str = 'Correlação entre Features') -> None:
        """Renderiza o heatmap de correlação de Pearson de forma segura."""
        fig, ax = plt.subplots(figsize=(13, 13))
        
        # Calcula correlação apenas das features numéricas
        corr_matrix = self.df.corr()
        
        # Máscara para esconder a diagonal superior (redundante)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt=f".{decimais}f", 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        ax.set_title(titulo, pad=20, fontsize=14)
        #plt.tight_layout()

    def calcular_vif(self) -> pd.DataFrame:
        """
        Calcula o Variance Inflation Factor (VIF).
        Matematicamente: VIF_i = 1 / (1 - R_i^2).
        Adiciona a constante obrigatória para evitar R^2 enviesado na origem.
        """
        # Adiciona intercepto para o cálculo correto via OLS interno do statsmodels
        X_const = add_constant(self.df)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_const.columns
        # List comprehension é ok aqui, mas array do numpy é repassado ao backend do C
        vif_data["VIF"] = [
            variance_inflation_factor(X_const.values, i) 
            for i in range(X_const.shape[1])
        ]
        
        # Remove a constante do output final, pois ela não é uma feature do domínio
        vif_data = vif_data[vif_data["Feature"] != "const"]
        
        return vif_data.round(2).sort_values(by="VIF", ascending=False).reset_index(drop=True)