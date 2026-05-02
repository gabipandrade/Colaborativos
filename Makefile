PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit
MODEL = qwen2.5:3b
OLLAMA_MODELS = /home/rafael/ollama-models

.PHONY: help venv install ollama pull web run train clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make venv      - Cria o ambiente virtual"
	@echo "  make install   - Instala as dependências Python"
	@echo "  make ollama    - Inicia o servidor do Ollama"
	@echo "  make pull      - Baixa o modelo qwen2.5:3b"
	@echo "  make train     - Treina os modelos K-Means e Isolation Forest"
	@echo "  make web       - Inicia a interface Streamlit"
	@echo "  make run       - Executa a versão terminal Colaborativo1.0.py"
	@echo "  make clean     - Remove arquivos temporários Python"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install langgraph langchain langchain-ollama streamlit pandas scikit-learn joblib plotly

ollama:
	OLLAMA_MODELS=$(OLLAMA_MODELS) ollama serve

pull:
	OLLAMA_MODELS=$(OLLAMA_MODELS) ollama pull $(MODEL)

train:
	OLLAMA_MODELS=$(OLLAMA_MODELS) $(VENV)/bin/python Treinamento.py

web:
	OLLAMA_MODELS=$(OLLAMA_MODELS) $(STREAMLIT) run Interface.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete