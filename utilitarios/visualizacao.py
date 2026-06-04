import math
import pandas as pd
import numpy as np

from typing import Literal, Optional, List

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score, mean_squared_error




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

class VisualizadorExploratorio:
    """
    Classe dedicada à Análise Exploratória de Dados (EDA) modular.
    Gerencia automaticamente a alocação de grids e o ciclo de vida das figuras do Matplotlib.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def _calcular_geometria_grid(self, num_plots: int, max_colunas: int = 3) -> tuple:
        """
        Calcula as dimensões ideais da matriz de subplots usando teoria dos conjuntos e função teto.
        Retorna (linhas, colunas).
        """
        colunas = min(num_plots, max_colunas)
        linhas = math.ceil(num_plots / colunas)
        return linhas, colunas

    def _remover_eixos_vazios(self, axes: iter, num_plots_reais: int) -> None:
        """Desliga a renderização de subplots excedentes criados pela matriz."""
        for i in range(num_plots_reais, len(axes)):
            axes[i].set_visible(False)

    def plotar_distribuicoes(
        self, 
        variaveis: List[str], 
        hue: Optional[str] = None, 
        max_colunas: int = 3
    ) -> None:
        """
        Gera histogramas com KDE para analisar a densidade probabilística das features.
        Fundamental para checar o comportamento dos fatores gerados pela PCA.
        """
        linhas, colunas = self._calcular_geometria_grid(len(variaveis), max_colunas)
        fig, axes = plt.subplots(linhas, colunas, figsize=(5 * colunas, 4 * linhas))
        axes = axes.flatten() if len(variaveis) > 1 else [axes]

        for i, var in enumerate(variaveis):
            # Validação fail-fast
            if var not in self.df.columns:
                raise KeyError(f"Feature '{var}' não encontrada no DataFrame.")
                
            sns.histplot(
                data=self.df, 
                x=var, 
                hue=hue, 
                kde=True, 
                ax=axes[i], 
                bins=20,
                alpha=0.6,
                palette='Set2' if hue else None
            )
            
            axes[i].set_title(f'Distribuição: {var.replace("_", " ").title()}', fontsize=12, pad=10)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequência' if i % colunas == 0 else '') # Y-label apenas nas bordas
            axes[i].spines[['top', 'right']].set_visible(False)

        self._remover_eixos_vazios(axes, len(variaveis))
        
        fig.suptitle('Análise Univariada: Densidade das Variáveis', fontsize=16, y=1.02)
        fig.tight_layout()
        plt.show()

    def plotar_relacionamentos(
        self, 
        preditores: List[str], 
        alvo: str, 
        hue: Optional[str] = None, 
        max_colunas: int = 3
    ) -> None:
        """
        Gera scatter plots bidimensionais mapeando Preditores (X) vs Alvo (y).
        Adiciona uma reta de tendência (Regressão Linear) para diagnóstico de linearidade e heterocedasticidade.
        """
        linhas, colunas = self._calcular_geometria_grid(len(preditores), max_colunas)
        fig, axes = plt.subplots(linhas, colunas, figsize=(5 * colunas, 4 * linhas))
        axes = axes.flatten() if len(preditores) > 1 else [axes]

        for i, fator in enumerate(preditores):
            # 1. Plot da dispersão com a estratificação (hue)
            sns.scatterplot(
                data=self.df, 
                x=fator, 
                y=alvo, 
                hue=hue, 
                ax=axes[i], 
                alpha=0.6,
                palette='Set2' if hue else None
            )
            
            # 2. Sobreposição da reta de regressão global (sem hue) para capturar o "efeito principal"
            sns.regplot(
                data=self.df, 
                x=fator, 
                y=alvo, 
                ax=axes[i], 
                scatter=False, 
                color='black', 
                line_kws={'linestyle':'--', 'linewidth': 1.5}
            )

            axes[i].set_title(f'{fator.replace("FATOR_", "")} vs {alvo.replace("TAXA_", "")}', fontsize=12)
            axes[i].set_xlabel("Score do Fator PCA")
            axes[i].set_ylabel(alvo.replace("_", " ").title() if i % colunas == 0 else '')
            axes[i].spines[['top', 'right']].set_visible(False)
            
            # Limpa legendas redundantes (mantém apenas a do primeiro plot, se existir)
            if hue and i > 0 and axes[i].get_legend() is not None:
                axes[i].get_legend().remove()

        # Ajuste fino da legenda do primeiro plot (caso exista)
        if hue and axes[0].get_legend() is not None:
            sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))

        self._remover_eixos_vazios(axes, len(preditores))
        
        fig.suptitle(f'Análise Bivariada: Preditores vs {alvo}', fontsize=16, y=1.02)
        fig.tight_layout()
        plt.show()

class DiagnosticoRegressao:
    """
    Classe padronizada para avaliação visual de modelos de regressão contínua.
    Garante identidade visual e rigor estatístico comparativo entre paradigmas.
    """
    
    def __init__(self):
        # Configurações globais de estilo limpo herdadas do seu design
        sns.set_theme(style="white") 

    def _aplicar_identidade_visual(self, ax: plt.Axes) -> None:
        """
        Aplica o padrão arquitetural de eixos e grids.
        Isola a manipulação da interface orientada a objetos (evitando o plt.gca()).
        """
        # Spines (Bordas)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Grid Inteligente: Tracejado, leve e atrás dos dados
        ax.grid(True, color='lightgrey', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    def _calcular_limites_dinamicos(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Calcula limites automáticos com uma margem de respiro de 5% para os eixos."""
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margem = (max_val - min_val) * 0.05
        return (min_val - margem, max_val + margem)

    def plotar_treino_teste_lado_a_lado(
        self,
        y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        hue_train: Optional[pd.Series] = None,
        hue_test: Optional[pd.Series] = None,
        ano_treino: str = "2010",
        ano_teste: str = "2011",
        titulo_modelo: str = "Random Forest"
    ) -> None:
        """
        Renderiza o diagnóstico bifásico (Treino vs Teste/Out-of-Time).
        Exige que as predições sejam feitas a priori, garantindo a separação de responsabilidades.
        """
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        
        dados_plot = [
            (axs[0], y_train_true, y_train_pred, hue_train, f'Treino ({ano_treino})'),
            (axs[1], y_test_true, y_test_pred, hue_test, f'Teste Out-of-Time ({ano_teste})')
        ]

        # Descobrindo o limite global para manter a mesma escala em ambos os gráficos
        limite_global = self._calcular_limites_dinamicos(
            np.concatenate([y_train_true, y_test_true]),
            np.concatenate([y_train_pred, y_test_pred])
        )

        for ax, y_true, y_pred, hue, label_split in dados_plot:
            # Métricas
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Scatter Plot com estratificação regional (hue)
            sns.scatterplot(
                x=y_true, 
                y=y_pred, 
                hue=hue,
                ax=ax,
                s=50,
                alpha=0.7,
                palette='Set1' if hue is not None else None,
                edgecolor='k',
                linewidth=0.5
            )
            
            # Linha de Tendência do Modelo (Regressão Linear nos resíduos)
            sns.regplot(
                x=y_true, 
                y=y_pred, 
                ax=ax, 
                scatter=False, 
                color='red', 
                line_kws={'linewidth': 2},
                label='Tendência das Previsões'
            )
            
            # A Matemática Pura: Reta de Previsão Perfeita (y = x)
            ax.plot(
                [limite_global[0], limite_global[1]], 
                [limite_global[0], limite_global[1]], 
                color='black', linestyle='--', linewidth=1.5, label='Previsão Perfeita ($y = \\hat{y}$)'
            )

            # Estilização
            ax.set_title(f'{label_split}\n$R^2$ = {r2:.3f} | RMSE = {rmse:.1f}', fontsize=13, pad=15)
            ax.set_xlabel('Taxa de Violência Real (Observada)', fontsize=11)
            ax.set_ylabel('Taxa de Violência Prevista', fontsize=11)
            
            # Força a proporção geométrica 1:1 e trava os limites
            ax.set_aspect('equal', 'box')
            ax.set_xlim(limite_global)
            ax.set_ylim(limite_global)
            
            self._aplicar_identidade_visual(ax)
            
            # Arrumando a legenda (mantém apenas uma limpa)
            if ax.get_legend() is not None:
                if ax == axs[1]: # Legenda completa apenas no gráfico da direita
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))
                else:
                    ax.get_legend().remove()

        fig.suptitle(f'Diagnóstico de Previsão Espaço-Temporal: {titulo_modelo}', fontsize=16, y=1.05)
        fig.tight_layout()
        plt.show()
    
    def plotar_desempenho_por_estrato(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        estratos: np.ndarray,
        nome_estrato: str = "Região",
        max_colunas: int = 3
    ) -> None:
        """
        Gera visualizações de 'Small Multiples' para avaliar o viés algorítmico 
        em diferentes categorias (ex: Regiões).
        """
        import math
        
        # Identifica as categorias únicas (ex: as 5 regiões do Brasil)
        categorias = np.unique(estratos)
        num_plots = len(categorias)
        
        # Calcula a geometria do grid de forma inteligente
        colunas = min(num_plots, max_colunas)
        linhas = math.ceil(num_plots / colunas)
        
        # sharex e sharey garantem que a escala seja idêntica para não distorcer a percepção
        fig, axes = plt.subplots(
            linhas, colunas, 
            figsize=(5.5 * colunas, 5 * linhas), 
            sharex=True, sharey=True
        )
        
        # Achatamento da matriz de eixos para facilitar iteração
        axes = axes.flatten() if num_plots > 1 else [axes]
        
        # Limites globais baseados na variação total de TODAS as regiões juntas
        limite_global = self._calcular_limites_dinamicos(y_true, y_pred)
        
        # R2 do país inteiro
        r2_global = r2_score(y_true, y_pred)
        
        for i, categoria in enumerate(categorias):
            ax = axes[i]
            
            # MÁSCARA SEGURA (NumPy boolean indexing estrito)
            mask = (estratos == categoria)
            y_real_cat = y_true[mask]
            y_pred_cat = y_pred[mask]
            
            # Trava de segurança: se a região não tiver municípios no split
            if len(y_real_cat) < 2:
                ax.set_title(f'{categoria}\n(Dados Insuficientes)', fontsize=12)
                self._aplicar_identidade_visual(ax)
                continue
                
            r2_local = r2_score(y_real_cat, y_pred_cat)
            rmse_local = np.sqrt(mean_squared_error(y_real_cat, y_pred_cat))
            
            # Scatter Plot Local
            sns.scatterplot(
                x=y_real_cat, 
                y=y_pred_cat, 
                ax=ax, 
                s=45,
                alpha=0.7,
                color='steelblue', # Mantém uma cor sóbria já que os títulos dizem a região
                edgecolor='k',
                linewidth=0.5
            )
            
            # Linha de Tendência do erro local
            sns.regplot(
                x=y_real_cat, 
                y=y_pred_cat, 
                ax=ax, 
                scatter=False, 
                color='red', 
                line_kws={'linewidth': 1.5}
            )
            
            # Reta Matemática Perfeita
            ax.plot(
                [limite_global[0], limite_global[1]], 
                [limite_global[0], limite_global[1]], 
                color='black', linestyle='--', linewidth=1.5
            )
            
            # Identidade Visual e Limites Quadrados
            ax.set_aspect('equal', 'box')
            ax.set_xlim(limite_global)
            ax.set_ylim(limite_global)
            self._aplicar_identidade_visual(ax)
            
            ax.set_title(f'{categoria}\n$R^2$ = {r2_local:.3f} | RMSE = {rmse_local:.1f}', fontsize=12, pad=10)
            
            # Inteligência de Labels: Apenas nas extremidades do Grid para reduzir poluição visual
            if i >= (linhas - 1) * colunas:
                ax.set_xlabel('Taxa Real (Observada)', fontsize=11)
            if i % colunas == 0:
                ax.set_ylabel('Taxa Prevista', fontsize=11)

        # Desligar os subplots sobressalentes gerados pela matriz matemática
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
            
        fig.suptitle(
            f'Avaliação de Desempenho Espacial: {nome_estrato}\n($R^2$ Global = {r2_global:.3f})', 
            fontsize=16, y=1.02, fontweight='bold'
        )
        fig.tight_layout()
        plt.show()

    def plotar_decadencia_temporal(
        self,
        anos_x: np.ndarray,
        r2_y: np.ndarray,
        limite_inferior: float = 0.0
    ) -> None:
        """
        Plota a curva de degradação preditiva do modelo ao longo do tempo (Model Decay).
        Mostra visualmente como a eficácia do modelo treinado em T0 cai em T+n.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # A Linha de Tendência de Degradação
        sns.lineplot(
            x=anos_x, 
            y=r2_y, 
            marker='o', 
            markersize=8,
            linewidth=2.5,
            color='crimson',
            ax=ax
        )
        
        # Rótulos precisos acima dos marcadores (sem hardcoding mágico de +0.02)
        margem_y = (max(r2_y) - min(r2_y)) * 0.05
        for x_val, y_val in zip(anos_x, r2_y):
            ax.text(
                x_val, 
                y_val + margem_y, 
                f'{y_val:.3f}', 
                ha='center', 
                va='bottom', 
                fontsize=11, 
                fontweight='bold',
                color='black'
            )

        # Identidade Visual Herdada
        self._aplicar_identidade_visual(ax)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y') # Grid apenas horizontal
        
        # Configuração de Eixos
        ax.set_title('Decadência Temporal Preditiva (Model Decay)', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Ano de Previsão Out-of-Time', fontsize=12, labelpad=10)
        ax.set_ylabel('Coeficiente de Determinação ($R^2$)', fontsize=12, labelpad=10)
        
        # Força os ticks a serem números inteiros (anos não têm decimais)
        ax.set_xticks(anos_x)
        ax.set_ylim(limite_inferior, min(1.0, max(r2_y) + 0.1))
        
        fig.tight_layout()
        plt.show()

    def plotar_importancia_variaveis(
        self,
        nomes_features: np.ndarray,
        importancias: np.ndarray,
        desvios_padrao: Optional[np.ndarray] = None,
        titulo: str = 'Importância dos Fatores (Permutation Importance)'
    ) -> None:
        """
        Gráfico horizontal rigoroso para análise de impacto dos preditores.
        Renderiza com o colormap 'viridis' mapeado dinamicamente sobre a magnitude da importância.
        """

        # Ordenação descendente
        indices_ordenados = np.argsort(importancias)
        nomes_ordenados = nomes_features[indices_ordenados]
        importancias_ordenadas = importancias[indices_ordenados]
        xerr = desvios_padrao[indices_ordenados] if desvios_padrao is not None else None
        
        # Normaliza as importâncias para uma escala de 0 a 1
        norm = mcolors.Normalize(vmin=importancias_ordenadas.min(), vmax=importancias_ordenadas.max())
        # Acessa o colormap nativo moderno (evitando métodos descontinuados)
        cmap = mpl.colormaps['viridis']
        # Mapeia as cores baseadas no peso matemático
        cores_viridis = cmap(norm(importancias_ordenadas))
        # ----------------------------------------

        fig, ax = plt.subplots(figsize=(14, 10)) # Tamanho ajustado conforme sua preferência
        
        bars = ax.barh(
            nomes_ordenados, 
            importancias_ordenadas,
            xerr=xerr,
            align='center',
            color=cores_viridis, # Injeta a matriz de cores
            edgecolor='black',
            linewidth=1.2,
            capsize=6, 
            alpha=0.9
        )
        
        # Adiciona rótulos percentuais à direita das barras (formatados)
        for bar in bars:
            largura = bar.get_width()
            ax.text(
                largura + (largura * 0.02), 
                bar.get_y() + bar.get_height() / 2, 
                f'{largura * 100:.2f}%', # Exibindo como percentual conforme você havia feito
                va='center', 
                ha='left', 
                fontsize=14,
                fontweight='bold'
            )

        self._aplicar_identidade_visual(ax)
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')
        ax.set_axisbelow(True)
        
        # Estilização herdada
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        
        ax.set_title(titulo, fontsize=20, pad=20, fontweight='bold')
        ax.set_xlabel('Importância Relativa (Redução do Erro %)', fontsize=16, labelpad=15)
        ax.set_xlim(0, max(importancias_ordenadas) * 1.15)
        
        fig.tight_layout()
        plt.show()