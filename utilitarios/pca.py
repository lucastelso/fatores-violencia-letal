import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ExtratorPCATematico(BaseEstimator, TransformerMixin):
    """
    Transformador Scikit-Learn customizado para realizar PCA em blocos temáticos.
    Garante o fit correto nos dados de treino e aplicação segura nos dados de teste,
    preservando rigorosamente os índices do Pandas.
    """
    
    def __init__(self, dicionario_temas: dict, temas_inverter_sinal: list = None):
        self.dicionario_temas = dicionario_temas
        self.temas_inverter_sinal = temas_inverter_sinal or []
        
        # Dicionários internos para persistir o estado de cada modelo
        self._scalers = {}
        self._pcas = {}
        self.loadings_ = {}
        self.variancia_explicada_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende os parâmetros estatísticos (média, variância, autovetores) para cada tema.
        """
        for tema, variaveis in self.dicionario_temas.items():
            # Validação de schema
            faltantes = [v for v in variaveis if v not in X.columns]
            if faltantes:
                raise KeyError(f"Variáveis ausentes no tema '{tema}': {faltantes}")

            # Subsetting e dropna estrito para o fit
            X_tema = X[variaveis].dropna()
            
            # Instancia e treina o StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_tema)
            self._scalers[tema] = scaler
            
            # Instancia e treina o PCA
            pca = PCA(n_components=1, random_state=42) # random_state para reprodutibilidade
            pca.fit(X_scaled)
            self._pcas[tema] = pca
            
            # Extrai métricas
            self.variancia_explicada_[tema] = pca.explained_variance_ratio_[0]
            
            # Gestão de Loadings e Inversão de Sinal (Regra de Negócio)
            multiplicador = -1 if tema in self.temas_inverter_sinal else 1
            loadings = pca.components_[0] * multiplicador
            
            self.loadings_[f'FATOR_{tema}'] = pd.Series(loadings, index=variaveis)

        return self # Retorna self para encadeamento de métodos do Sklearn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações estatísticas baseadas no fit anterior.
        Garante alinhamento de índices.
        """
        X_out = X.copy()
        
        for tema, variaveis in self.dicionario_temas.items():
            # Máscara booleana para identificar linhas sem NaN nestas colunas específicas
            mask_validos = X_out[variaveis].notna().all(axis=1)
            X_tema_validos = X_out.loc[mask_validos, variaveis]
            
            if X_tema_validos.empty:
                X_out[f'FATOR_{tema}'] = pd.NA
                continue
                
            # Recupera os modelos treinados
            scaler = self._scalers[tema]
            pca = self._pcas[tema]
            
            # Transformação matemática
            X_scaled = scaler.transform(X_tema_validos)
            fator_raw = pca.transform(X_scaled).flatten()
            
            # Inversão de sinal baseada na regra de negócio
            multiplicador = -1 if tema in self.temas_inverter_sinal else 1
            fator_final = fator_raw * multiplicador
            
            # Injeção segura utilizando os índices originais!
            fator_series = pd.Series(fator_final, index=X_tema_validos.index)
            X_out.loc[mask_validos, f'FATOR_{tema}'] = fator_series
            
        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def exibir_relatorio(self) -> None:
        """Resultados da extração."""
        print("="*50)
        print(" RELATÓRIO DE EXTRAÇÃO DE FATORES (PCA) ".center(50))
        print("="*50)
        for tema in self.dicionario_temas.keys():
            print(f"\n[FATOR: {tema}]")
            print(f"Variância Explicada: {self.variancia_explicada_[tema]:.2%}")
            print(f"Composição (Loadings):")
            print(self.loadings_[f'FATOR_{tema}'].to_string())
        print("\n" + "="*50)