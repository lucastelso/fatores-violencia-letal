# Nome do ambiente virtual
VENV = .venv
PYTHON = python3.11
PIP = $(VENV)/bin/pip

.PHONY: setup clean

# Comando principal: cria o venv, atualiza o pip e instala os pacotes
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "=> Criando o ambiente virtual com $(PYTHON)..."
	$(PYTHON) -m venv $(VENV)
	@echo "=> Atualizando o pip..."
	$(PIP) install --upgrade pip
	@echo "=> Instalando as dependências do projeto..."
	$(PIP) install -r requirements.txt
	@echo "=> Tudo pronto! Para ativar o ambiente, execute: source $(VENV)/bin/activate"

# Comando para limpar o ambiente se quiser recomeçar do zero
clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete