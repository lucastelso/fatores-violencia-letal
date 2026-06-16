from typing import Literal
import pickle
from pathlib import Path
import logging
import sys

PATH_DADOS = "dados_brutos"

def criar_path(
        tabela: str,
        base: str = "dados_brutos",
        ) -> Path:
    return Path(fr"{base}/{tabela}")

modelos = Literal[
    'modelo_rf_683.pkl', 'melhor_rf.pkl'
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

class CustomFormatter(logging.Formatter):
    """Logger"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"
    
    # Formato limpo e informativo: Timestamp [NÍVEL] (Arquivo:Linha) -> Mensagem
    log_format = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) -> %(message)s"

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: cyan + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

    def get_logger(name: str = __name__) -> logging.Logger:
        """Retorna um logger customizado e configurado para o terminal."""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)

            # Configura o output para o stdout (terminal)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(CustomFormatter())
            
            logger.addHandler(stdout_handler)
            
        return logger