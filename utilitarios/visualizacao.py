import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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