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
from scipy.special import inv_boxcox



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
            Executa os testes formais de Normalidade (Anderson-Darling, Shapiro-Wilk e Shapiro-Francia) 
            e Homocedasticidade (Breusch-Pagan).
            """
            
            try:
                from statstests.tests import shapiro_francia
                possui_statstests = True
            except ImportError:
                possui_statstests = False
                
            print(f"\n{'='*60}")
            print(f" DIAGNÓSTICO QUANTITATIVO: {nome_contexto.upper()} ".center(60))
            print(f"{'='*60}")
            
            # -------------------------------------------------------------
            # 1. TESTES DE NORMALIDADE
            # -------------------------------------------------------------
            print("\n[1] NORMALIDADE DOS RESÍDUOS")
            
            # 1.1 Anderson-Darling (Sempre bom manter como base estrutural para N > 50)
            ad_stat, crit_vals, sig_levels = stats.anderson(residuos, dist='norm')
            print(f"\nAnderson-Darling: Estatística = {ad_stat:.3f}")
            for sl, cv in zip(sig_levels, crit_vals):
                status = "REJEITADO (Não-Normal)" if ad_stat > cv else "ACEITO (Normal)"
                print(f"  > Nível {sl}% -> Valor Crítico = {cv:.3f} | {status}")

            # 1.2 Shapiro-Wilk (Executado simultaneamente como benchmark)
            stat_sw, p_sw = stats.shapiro(residuos)
            status_sw = "ACEITO (Normal)" if p_sw > self.alpha else "REJEITADO (Não-Normal)"
            print(f"\nShapiro-Wilk ($W$): Estatística = {stat_sw:.4f} | p-valor = {p_sw:.4f} -> {status_sw}")
                
            # 1.3 Shapiro-Francia (A extração blindada)
            # 1.3 Shapiro-Francia (A extração blindada por Chave)
            if possui_statstests:
                residuos_serie = pd.Series(residuos, name='Residuos') if isinstance(residuos, np.ndarray) else residuos
                
                import sys, io
                # Bloqueia o print "poluído" nativo da biblioteca statstests para o terminal ficar limpo
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    resultado_sf = shapiro_francia(residuos_serie)
                finally:
                    sys.stdout = old_stdout
                
                # Extração estrita via dicionário (Padrão do pacote)
                if isinstance(resultado_sf, dict):
                    estatistica_w = float(resultado_sf.get('statistics W', 0.0))
                    p_valor_sf = float(resultado_sf.get('p-value', 0.0))
                elif isinstance(resultado_sf, pd.DataFrame):
                    # Se em alguma atualização futura virar DataFrame
                    col_p = [c for c in resultado_sf.columns if 'p' in str(c).lower()]
                    col_w = [c for c in resultado_sf.columns if 'w' in str(c).lower()]
                    estatistica_w = float(resultado_sf[col_w[0]].iloc[0])
                    p_valor_sf = float(resultado_sf[col_p[0]].iloc[0])
                else:
                    estatistica_w, p_valor_sf = float(resultado_sf[0]), float(resultado_sf[-1]) # Pega o último para evitar o Z
                
                status_sf = "ACEITO (Normal)" if p_valor_sf > self.alpha else "REJEITADO (Não-Normal)"
                print(f"Shapiro-Francia ($W'$): Estatística = {estatistica_w:.4f} | p-valor = {p_valor_sf:.4e} -> {status_sf}")

            # -------------------------------------------------------------
            # 2. TESTE DE HOMOCEDASTICIDADE
            # -------------------------------------------------------------
            print("\n[2] HOMOCEDASTICIDADE (Breusch-Pagan)")
            matriz_exog = exog if exog is not None else sm.add_constant(valores_ajustados)
            
            bp_test = het_breuschpagan(residuos, matriz_exog)
            lm_stat, p_lm = bp_test[0], bp_test[1]
            
            print(f"Estatística LM: {lm_stat:.3f} | p-valor: {p_lm:.4f}")
            if p_lm > self.alpha:
                print(">> Conclusão: Resíduos Homocedásticos (A variância é constante).")
            else:
                print(">> ALERTA ESTRUTURAL: Heterocedasticidade detectada.")
                print(">> Justifica-se o uso de transformação Box-Cox, Log ou regressão robusta multinível.")
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

class DiagnosticadorMultinivel:
    """
    Extensão especializada para Modelos Hierárquicos (MixedLM).
    Extrai e renderiza as métricas da matriz de covariância estrutural.
    """

    @staticmethod
    def r2_nakagawa_schielzeth(modelo_ajustado: sm.regression.mixed_linear_model.MixedLMResults) -> Tuple[float, float]:
        """
        Calcula o R² de Nakagawa para Modelos Mistos usando os componentes de variância nativos do statsmodels.
        Garante a separação perfeita entre efeitos fixos e variâncias hierárquicas.
        """
        # 1. Variância Residual Pura (Nível 1 - e)
        var_residual = modelo_ajustado.scale
        
        # 2. Variância dos Efeitos Aleatórios (Nível 2 - Grupos)
        # Extraímos do vetor de parâmetros as variâncias de grupo ('Group Var'). 
        # Cuidado: Modelos de 'slope' aleatório (FATOR_EMPREGO Var) exigem traço da matriz, mas 
        # para o impacto do R² Condicional padrão, somamos as variâncias não-estruturais.
        componentes_vc = [col for col in modelo_ajustado.params.index if 'Var' in col]
        var_aleatoria = modelo_ajustado.params[componentes_vc].sum()
        
        # 3. Variância dos Efeitos Fixos (Apenas X_beta, sem interceptos aleatórios)
        # modelo_ajustado.model.exog é a matriz X. modelo_ajustado.fe_params são os betas.
        pred_fixa = np.dot(modelo_ajustado.model.exog, modelo_ajustado.fe_params)
        var_fixa = np.var(pred_fixa, ddof=1) # ddof=1 para estimador não-viesado da variância
        
        # 4. A Matemática de Nakagawa
        var_total = var_fixa + var_aleatoria + var_residual
        r2_marginal = var_fixa / var_total
        r2_condicional = (var_fixa + var_aleatoria) / var_total
        
        return r2_marginal, r2_condicional

    def renderizar_previsao_boxcox(
        self,
        modelo_bc_ajustado: sm.regression.mixed_linear_model.MixedLMResults,
        df_completo: pd.DataFrame,
        coluna_alvo: str,
        coluna_estrato: str,
        lambda_bc: float
    ) -> None:
        """
        Realiza o de-transformation de Box-Cox e plota o diagnóstico de aderência visual,
        enriquecido com as métricas absolutas de Nakagawa.
        """
        # 1. Prepara as Séries (Alinhamento Indexado Automático)
        y_real = df_completo[coluna_alvo]
        y_pred_boxcox = modelo_bc_ajustado.fittedvalues
        
        # 2. De-Transformation Segura (Voltando à taxa de homicídios bruta)
        y_pred_revertido = inv_boxcox(y_pred_boxcox, lambda_bc)
        # Correção do shift -1, assumino que usou-se box-cox + 1 para evitar divisão por 0
        y_pred_revertido = y_pred_revertido - 1 

        # 3. Calcula Métricas Sólidas
        r2_marg, r2_cond = self.r2_nakagawa_schielzeth(modelo_bc_ajustado)

        # 4. Configuração Arquitetural do Plot
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        sns.set_theme(style="white")
        
        # Limites Universais (Com respiro de 5)
        min_val = min(y_real.min(), y_pred_revertido.min()) - 5
        max_val = max(y_real.max(), y_pred_revertido.max()) + 5
        limites = [min_val, max_val]

        # Scatter
        sns.scatterplot(
            x=y_real, 
            y=y_pred_revertido, 
            hue=df_completo[coluna_estrato],
            palette='Set1',
            s=60,
            alpha=0.85,
            edgecolor='k',
            linewidth=0.6,
            ax=ax
        )

        # Reta de Ajuste (Com os dados na escala humana)
        sns.regplot(
            x=y_real, 
            y=y_pred_revertido, 
            scatter=False,
            color='crimson',
            line_kws={'linewidth': 2, 'linestyle': '-'},
            label='Curva de De-Transformation'
        )

        # Previsão Perfeita Geométrica
        ax.plot(limites, limites, color='black', linestyle='--', linewidth=1.5, label='Previsão Perfeita ($y = \\hat{y}$)')

        # Identidade Visual e Textos
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        ax.set_aspect('equal', 'box')
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        
        titulo_formatado = (
            f"MODELO HIERÁRQUICO (BOX-COX $\\lambda = {lambda_bc:.3f}$)\n"
            f"Nakagawa $R^2$ Marginal = {r2_marg:.3f} (Fatores Estruturais)\n"
            f"Nakagawa $R^2$ Condicional = {r2_cond:.3f} (Fatores + Efeito Geográfico)"
        )
        
        ax.set_title(titulo_formatado, fontsize=14, pad=15, loc='left', fontweight='bold')
        ax.set_xlabel('Taxa de Violência Real (Mortes/100k)', fontsize=12)
        ax.set_ylabel('Taxa Prevista Revertida', fontsize=12)
        
        ax.legend(frameon=False, loc='upper left')
        if ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        plt.show()