# Quick Start - Sistema RAG Antifraude

## ⚡ Inicio Rápido (5 minutos)

### 1️⃣ Dependências

```bash
cd ~/Colaborativos
source .venv/bin/activate

# Instalar dependências
pip install langgraph langchain langchain-ollama langchain-community langchain-chroma langchain-text-splitters chromadb
```

### 2️⃣ Iniciar Ollama (Terminal 1)

```bash
OLLAMA_MODELS=/home/gabi/ollama-models ollama serve
```

Aguarde aparecer:
```
Listening on 127.0.0.1:11434
```

### 3️⃣ Rodar o Agent (Terminal 2)

```bash
cd ~/Colaborativos
source .venv/bin/activate
python Colaborativo1.0.py
```

Você verá:
```
[SISTEMA] Inicializando RAG...
[RAG] Carregando arquivo: politicas_antifraude.txt
[RAG] Carregando arquivo: relatorios_fraude.txt
[SISTEMA] RAG inicializado!

Sistema colaborativo de detecção de fraude iniciado.
Digite 'sair' para encerrar.

Usuário: 
```

---

## 🧪 Teste Rápido

Copie e cole estas perguntas no agente:

### Teste 1: Análise de Transação
```
Verificar transação: João em São Paulo, R$ 500, eletrônicos, noturno
```

**O que esperar:** Agent usa `verificar_transacao()`, analisa, responde se é normal ou suspeita.

### Teste 2: Consultar Políticas
```
Qual é o limite de PIX para cliente Premium?
```

**O que esperar:** Agent usa `consultar_politicas_antifraude()`, busca RAG, retorna limite.

### Teste 3: Buscar Histórico
```
Quais fraudes foram detectadas por mudança de localização?
```

**O que esperar:** Agent usa `consultar_relatorios_fraude()`, encontra Relatório #001.

### Teste 4: Caso Complexo
```
Analisar: Maria em Madrid, R$ 3.500, roupas, 23:00
```

**O que esperar:** Agent faz análise em cadeia:
1. Verifica transação (Score médio)
2. Consulta políticas (horário noturno)
3. Busca histórico (Relatório #003: falso positivo)
4. Solicita confirmação

---

## 📊 Arquivos de Dados

Você tem 2 arquivos de dados prontos para usar:

### politicas_antifraude.txt
- 10 seções com políticas do banco
- 45+ chunks indexados
- Pronto para buscas sobre: limites, indicadores, padrões

### relatorios_fraude.txt
- 12 casos de fraude documentados
- 28+ chunks indexados
- Pronto para buscas sobre: tipos, padrões, histórico

---

## 🛠️ Troubleshooting

### Erro: "Dependências RAG não disponíveis"
```bash
pip install langchain-text-splitters langchain-chroma chromadb
python Colaborativo1.0.py
```

### Erro: "Arquivo não encontrado"
Verifique que estão na pasta:
```bash
ls -la /home/gabi/Colaborativos/politicas_*.txt
ls -la /home/gabi/Colaborativos/relatorios_*.txt
```

### Erro: "Ollama não respondendo"
```bash
# Terminal 1 - Reinicie Ollama:
OLLAMA_MODELS=/home/gabi/ollama-models ollama serve

# Verifique:
curl http://localhost:11434/api/tags
```

### Vector store corrompido
```bash
rm -rf /home/gabi/Colaborativos/vdb
python Colaborativo1.0.py
# Vai recriar os índices automaticamente
```

---

## 📖 Documentação Completa

- **RAG_DOCUMENTATION.md** - Guia técnico detalhado
- **IMPLEMENTATION_SUMMARY.md** - Resumo da implementação
- **README.md** - Documentação geral do projeto

---

## 🚀 O que Vem Depois

Ideias para expandir:

1. **Adicionar mais políticas** - Críme financeiro, compliance, LGPD
2. **PDFs reais** - Use `search_pdf` para documentos PDF do banco
3. **Web UI** - Crie dashboard com Streamlit ou FastAPI
4. **Logging** - Registre todas as decisões para audit
5. **Métricas** - Acompanhe precisão vs. falsos positivos
6. **Integração** - Conecte com sistema real de transações

---

## ✅ Checklist

- [x] Arquivos de dados criados
- [x] RAG integrado no Colaborativo1.0.py
- [x] Vector store Chroma funcionando
- [x] Embeddings Ollama carregando
- [x] Tools RAG disponíveis
- [x] Documentação completa
- [x] Exemplos de uso pronto

**Status: ✅ Pronto para usar!**

---

## 💡 Dica Pro

O vector store é persistido em `./vdb/`. Na próxima vez que rodar:
- Não recupera dos arquivos de texto
- Usa o índice já criado
- Mais rápido (~3x)

Para forçar reconstrução: `rm -rf ./vdb`

---

Sucesso! 🎉
