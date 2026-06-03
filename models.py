import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, learning_curve, RepeatedKFold

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
