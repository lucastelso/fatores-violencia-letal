from typing import Literal
import pickle
from pathlib import Path


modelos = Literal[
    'modelo_novo_683_a94.pkl', 'melhor_rf.pkl'
    ]

def carregar_modelo(
        model:str=modelos, 
        path:str='./modelos_salvos/'
        ):
    """Carrega um modelo pré-treinado localizado em alguma pasta do projeto"""
    if not path:
        raise print('forneça um caminho')
    if not model:
        raise print('pelo menos um modelo deve ser apontado')
    
    with open(Path(f"{path}/{model}"), 'rb') as file:
        found_model = pickle.load(file)
    
    return found_model