import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, learning_curve, RepeatedKFold
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import seaborn as sns

class AvaliadorRandomForest:
    """
    Classe modular para otimização e avaliação de curvas de aprendizado 
    para modelos Random Forest em bases sensíveis a outliers.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.melhor_modelo_ = None
        self.melhores_params_ = None
        
    def otimizar_hiperparametros(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        param_distributions: Dict[str, Any],
        n_iter: int = 500,
        cv_splits: int = 5
    ) -> RandomForestRegressor:
        """
        Executa a busca de hiperparâmetros em todo o dataset usando Cross-Validation robusta.
        """
        print("🚀 Iniciando Otimização de Hiperparâmetros...")
        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # RepeatedKFold reduz a variância causada por outliers em bases com N pequeno
        cv_strategy = RepeatedKFold(n_splits=cv_splits, n_repeats=3, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='r2',
            cv=cv_strategy,
            n_jobs=-1, # Aloca todos os cores disponíveis
            random_state=self.random_state,
            verbose=1
        )
        
        search.fit(X, y)
        
        self.melhor_modelo_ = search.best_estimator_
        self.melhores_params_ = search.best_params_
        
        print(f"\nOtimização Concluída!")
        print(f"Melhor R² no CV: {search.best_score_:.4f}")
        print(f"Melhores Parâmetros: {self.melhores_params_}")
        
        return self.melhor_modelo_

    def gerar_curva_aprendizado(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_splits: int = 5,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Avalia o impacto do volume de dados na performance do modelo já otimizado.
        Substitui todos os antigos loops manuais pesados por C otimizado no backend.
        """
        if self.melhor_modelo_ is None:
            raise NotFittedError("Você precisa rodar 'otimizar_hiperparametros' primeiro.")
            
        print("\nCalculando Curva de Aprendizado...")
        cv_strategy = RepeatedKFold(n_splits=cv_splits, n_repeats=3, random_state=self.random_state)
        
        # A magia do sklearn: ele faz o particionamento e os fits automaticamente
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator=self.melhor_modelo_,
            X=X, 
            y=y,
            train_sizes=train_sizes,
            cv=cv_strategy,
            scoring='r2',
            n_jobs=-1
        )
        
        return train_sizes_abs, train_scores, test_scores

    def plotar_curva_aprendizado(
        self, 
        train_sizes_abs: np.ndarray, 
        train_scores: np.ndarray, 
        test_scores: np.ndarray
    ) -> None:
        """
        Renderiza o gráfico de diagnóstico de viés/variância.
        """
        # Calculando médias e desvios para construir a banda de confiança
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.set_title("Curva de Aprendizado (Random Forest Otimizada)", fontsize=14, pad=15)
        ax.set_xlabel("Número de Instâncias de Treino", fontsize=12)
        ax.set_ylabel("R² (Coeficiente de Determinação)", fontsize=12)
        
        # Banda de variância do Treino
        ax.fill_between(
            train_sizes_abs, 
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, 
            alpha=0.1, color="blue"
        )
        # Linha média do Treino
        ax.plot(train_sizes_abs, train_scores_mean, 'o-', color="blue", label="Score de Treino")

        # Banda de variância da Validação Cruzada (Teste)
        ax.fill_between(
            train_sizes_abs, 
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, 
            alpha=0.1, color="orange"
        )
        # Linha média do Teste
        ax.plot(train_sizes_abs, test_scores_mean, 's-', color="orange", label="Score de Validação Cruzada")

        # Linha de baseline (R2 = 0)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Estilização
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")
        
        plt.tight_layout()
        plt.show()

class AvaliadorRegressaoParametrica:
    """
    Módulo de diagnóstico rigoroso para modelos OLS e MixedLM (Multinível).
    Encapsula testes de pressupostos e padronização visual.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        sns.set_theme(style="white")

    def _aplicar_identidade_visual(self, ax: plt.Axes, grid_horizontal: bool = True, grid_vertical: bool = True) -> None:
        """Centraliza a formatação corporativa dos eixos, eliminando o plt.gca()."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        ax.grid(False)
        if grid_horizontal:
            ax.yaxis.grid(True, color='lightgrey', linestyle='--', alpha=0.7)
        if grid_vertical:
            ax.xaxis.grid(True, color='lightgrey', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    def diagnostico_quantitativo(
        self, 
        residuos: np.ndarray, 
        valores_ajustados: np.ndarray, 
        exog: Optional[np.ndarray] = None,
        nome_contexto: str = "Nível 1"
    ) -> None:
        """
        Executa os testes formais de Normalidade (Shapiro/Anderson) e Homocedasticidade (Breusch-Pagan).
        """
        print(f"\n{'='*60}")
        print(f" DIAGNÓSTICO QUANTITATIVO: {nome_contexto.upper()} ".center(60))
        print(f"{'='*60}")
        
        # 1. Teste de Normalidade (Anderson-Darling é superior para N > 50)
        print("\n[1] NORMALIDADE DOS RESÍDUOS")
        ad_stat, crit_vals, sig_levels = stats.anderson(residuos, dist='norm')
        print(f"Estatística Anderson-Darling: {ad_stat:.3f}")
        
        is_normal = True
        for sl, cv in zip(sig_levels, crit_vals):
            status = "REJEITADO (Não-Normal)" if ad_stat > cv else "ACEITO (Normal)"
            if sl == (self.alpha * 100) and ad_stat > cv:
                is_normal = False
            print(f"  > Nível {sl}% -> Valor Crítico = {cv:.3f} | {status}")
            
        # Opcional: Shapiro-Wilk como dupla checagem
        stat_sw, p_sw = stats.shapiro(residuos)
        print(f"Shapiro-Wilk: p-valor = {p_sw:.4f} -> {'Normal' if p_sw > self.alpha else 'Não-Normal'}")

        # 2. Teste de Homocedasticidade (Breusch-Pagan)
        print("\n[2] HOMOCEDASTICIDADE (Breusch-Pagan)")
        # Se exog não for fornecido, criamos um proxy usando fitted_values e constante
        matriz_exog = exog if exog is not None else sm.add_constant(valores_ajustados)
        
        bp_test = het_breuschpagan(residuos, matriz_exog)
        lm_stat, p_lm = bp_test[0], bp_test[1]
        
        print(f"Estatística LM: {lm_stat:.3f} | p-valor: {p_lm:.4f}")
        if p_lm > self.alpha:
            print(">> Conclusão: Resíduos Homocedásticos (Variância Constante).")
        else:
            print(">> ALERTA: Resíduos Heterocedásticos. Considere matriz de covariância robusta (ex: HC3) ou Box-Cox.")
        print(f"{'-'*60}\n")

    def plotar_previsao_vs_real(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        hue_labels: pd.Series, 
        titulo: str = "Regressão Linear Múltipla (OLS)"
    ) -> None:
        """
        Plota o ajuste do modelo, sobrepondo predições e valores reais com a reta perfeita matemática.
        Elimina *hardcoding* de limites, ajustando-se automaticamente à escala linear, log ou Box-Cox.
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Limites dinâmicos universais baseados nos dados
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margem = (max_val - min_val) * 0.05
        limites = [min_val - margem, max_val + margem]

        # 1. Dispersão Estratificada
        sns.scatterplot(
            x=y_true, 
            y=y_pred,
            hue=hue_labels,
            palette='Set1',
            s=55,
            alpha=0.75,
            edgecolor='k',
            ax=ax
        )

        # 2. Reta de Regressão do Modelo (Em vermelho)
        sns.regplot(
            x=y_true, 
            y=y_pred,
            scatter=False,
            color='crimson',
            line_kws={'linewidth': 2},
            label='Tendência do Ajuste',
            ax=ax
        )

        # 3. Reta Matemática Pura de 45°
        ax.plot(
            limites, limites, 
            color='black', linestyle='--', linewidth=1.5, label='Previsão Perfeita ($y = \\hat{y}$)'
        )

        # Estilização Paramétrica
        self._aplicar_identidade_visual(ax)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        
        # R2 extraído dinamicamente
        r2_ajustado = np.corrcoef(y_true, y_pred)[0, 1]**2
        
        ax.set_title(f'{titulo}\n$R^2$ = {r2_ajustado:.3f}', fontsize=16, pad=15, loc='left', fontweight='bold')
        ax.set_xlabel('Taxa de Violência Real (Observada)', fontsize=12)
        ax.set_ylabel('Taxa de Violência Prevista', fontsize=12)
        
        # Move legenda e remove frame
        ax.legend(frameon=False, loc='upper left')
        if ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        plt.show()

    def plotar_painel_residuos(self, residuos: np.ndarray, valores_ajustados: np.ndarray, titulo: str = "Nível 1") -> None:
        """
        Gera um painel triplo (1x3) para inspeção anatômica dos resíduos de forma corporativa.
        1. Histograma (Densidade).
        2. Q-Q Plot (Cauda e Normalidade).
        3. Scatter (Variância / Homocedasticidade).
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Histograma
        sns.histplot(residuos, kde=True, ax=axes[0], color='steelblue', alpha=0.6)
        axes[0].set_title('Densidade dos Resíduos', fontsize=13)
        axes[0].set_xlabel('Valor do Resíduo')
        self._aplicar_identidade_visual(axes[0], grid_vertical=False)

        # 2. Q-Q Plot
        sm.qqplot(residuos, line='45', ax=axes[1], color='steelblue', alpha=0.6)
        axes[1].set_title('Q-Q Plot (Normalidade)', fontsize=13)
        self._aplicar_identidade_visual(axes[1])

        # 3. Resíduos vs Fitted (Homocedasticidade)
        sns.scatterplot(x=valores_ajustados, y=residuos, ax=axes[2], color='darkorange', alpha=0.7, edgecolor='k')
        axes[2].axhline(0, color='crimson', linestyle='--', linewidth=1.5)
        axes[2].set_title('Resíduos vs Valores Ajustados', fontsize=13)
        axes[2].set_xlabel('Valores Previstos (Fitted)')
        axes[2].set_ylabel('Resíduos Condicionais')
        self._aplicar_identidade_visual(axes[2])

        fig.suptitle(f'Diagnóstico Estrutural de Resíduos: {titulo}', fontsize=16, y=1.05, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plotar_efeitos_aleatorios_multinivel(self, dict_efeitos_aleatorios: Dict[Any, pd.Series]) -> None:
        """
        Exclusivo para o Modelo Multinível (MixedLM). Avalia o Nível 2.
        Recebe o dicionário resultante de `modelo.random_effects`.
        """
        # Converte o output padrão do statsmodels MixedLM para DataFrame
        df_re = pd.DataFrame.from_dict(dict_efeitos_aleatorios, orient='index')
        
        for coluna in df_re.columns:
            valores = df_re[coluna].to_numpy()
            
            # Se a amostra de grupos (ex: regiões) for pequena, adapta o plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.histplot(valores, kde=True, ax=axes[0], color='teal', bins=min(10, len(valores)))
            axes[0].set_title(f'Histograma: Efeito Aleatório ({coluna})')
            self._aplicar_identidade_visual(axes[0], grid_vertical=False)
            
            sm.qqplot(valores, line='s', ax=axes[1], color='teal', alpha=0.8) # line='s' usa variância padronizada
            axes[1].set_title(f'Q-Q Plot: Efeito Aleatório ({coluna})')
            self._aplicar_identidade_visual(axes[1])
            
            fig.suptitle(f'Diagnóstico Hierárquico Nível 2 (Efeitos Aleatórios)', fontsize=15, y=1.02)
            plt.tight_layout()
            plt.show()