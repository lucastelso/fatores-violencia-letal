# Nome do ambiente virtual
VENV = .venv
PYTHON = python3.11
PIP = $(VENV)/bin/pip

.PHONY: setup clean

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "=> Criando o ambiente virtual leve com $(PYTHON)..."
	$(PYTHON) -m venv $(VENV)
	@echo "=> Atualizando o pip..."
	$(PIP) install --upgrade pip
	@echo "=> Instalando as dependências de forma robusta..."
	$(PIP) install -r requirements.txt
	@echo "=> Tudo pronto! Para ativar o ambiente, execute: source $(VENV)/bin/activate"

clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
