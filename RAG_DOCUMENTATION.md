# Sistema RAG Antifraude - Documentação

## Visão Geral

O sistema **Colaborativo1.0.py** agora possui um **RAG (Retrieval-Augmented Generation) completo** que consulta dois arquivos de conhecimento:

1. **politicas_antifraude.txt** - Políticas e procedimentos do banco
2. **relatorios_fraude.txt** - Histórico e padrões de fraude

## Arquivos de Dados

### 1. politicas_antifraude.txt
Documento com:
- Definições de fraude
- Limites de transação por perfil de cliente
- Indicadores de risco (scores)
- Padrões de uso esperados
- Procedimentos de verificação
- Períodos de risco elevado
- Procedimentos de recuperação pós-fraude

**Usado para:** Validar limites, entender critérios de risco, aplicar políticas corretas

### 2. relatorios_fraude.txt
Arquivo com:
- 12 relatórios históricos de fraude (alguns confirmados, alguns falsos positivos)
- Padrões identificados no histórico
- Tipos de fraude: identidade, cartão clonado, social engineering, etc.
- Resultados e lições aprendidas

**Usado para:** Comparar transações com casos históricos, identificar padrões

## Novas Ferramentas (Tools)

### ✓ consultar_politicas_antifraude(pergunta)
Busca informações nas políticas antifraude usando RAG semântico.

**Exemplos:**
```
- "Qual é o limite de compra online para cliente padrão?"
- "Como identificar fraude de cartão clonado?"
- "Quais são os indicadores de ALTO RISCO?"
- "O que fazer em caso de fraude confirmada?"
```

### ✓ consultar_relatorios_fraude(pergunta)
Busca casos históricos e padrões de fraude.

**Exemplos:**
```
- "Quais fraudes foram detectadas por mudança de localização?"
- "Qual é o padrão de teste de limite em fraudes?"
- "Como foram detectadas fraudes em massa?"
- "Que tipos de fraude temos neste banco?"
```

## Como Funciona o RAG

```
1. Startup: 
   - Lê politicas_antifraude.txt e relatorios_fraude.txt
   - Divide em chunks (800 caracteres com 200 de sobreposição)
   - Cria embeddings com Ollama (modelo: nomic-embed-text)
   - Armazena em Chroma (./vdb/) com coleções separadas

2. Durante execução:
   - Usuário faz pergunta → agente
   - Agente decide se precisa consultar RAG
   - Se usar tool, busca similaridade semântica
   - Retorna top-5 trechos mais relevantes
   - Agente responde com contexto

3. Integração:
   - Tools funcionam com LangGraph
   - Agente escolhe automaticamente qual tool usar
   - Pode combinar múltiplas ferramentas em uma resposta
```

## Fluxo de Análise de Transação

```
Usuário: "Verificar transação de João em São Paulo, R$ 500, eletrônicos, noturno"

1. Agente recebe pergunta
2. Execute: verificar_transacao() → análise local
3. Se Score ≥ 80 (CRÍTICO):
   - Execute: consultar_politicas_antifraude() → padrão esperado
   - Execute: consultar_relatorios_fraude() → casos históricos
   - Execute: solicitar_confirmacao_cliente() → confirmação
   - Execute: gerar_relatorio_banco() → documentar decision

4. Resposta: "Transação suspeita por X motivos. Comparar com políticas: ... Histórico: ..."
```

## Dependências Necessárias

O projeto requer que todos estes módulos estejam instalados via pip:

```bash
pip install langgraph langchain langchain-ollama langchain-community langchain-chroma langchain-text-splitters chromadb
```

## Executando

```bash
# Terminal 1 - Iniciar Ollama
OLLAMA_MODELS=/home/gabi/ollama-models ollama serve

# Terminal 2 - Rodar o agente
cd ~/Colaborativos
source .venv/bin/activate
python Colaborativo1.0.py
```

## Exemplos de Uso

### Exemplo 1: Consultar Política
```
Usuário: Qual é o limite de transferência PIX para cliente Premium?
Agente: [usa consultar_politicas_antifraude]
Resposta: Limite de transferência PIX para Premium é R$ 100.000 por transação...
```

### Exemplo 2: Comparar com Histórico
```
Usuário: Houve fraude com mudança de localização geográfica neste banco?
Agente: [usa consultar_relatorios_fraude]
Resposta: Sim, no Relatório #001, João Silva teve uma tentativa em Manaus quando normalmente 
transaciona em São Paulo. O sistema bloqueou automaticamente...
```

### Exemplo 3: Análise Completa
```
Usuário: Verificar transação de Ana em Madrid, R$ 3.500, roupas, 23:00
Agente: 
  1. [usa verificar_transacao] → Score 78 (MÉDIO RISCO)
  2. [usa consultar_politicas_antifraude] → comparar com límites
  3. [usa consultar_relatorios_fraude] → buscar casos similares
  4. [usa solicitar_confirmacao_cliente] → pedir confirmação
  
Resposta: "Transação de MÉDIO RISCO detectada. Score 78. Motivos: horário 
anômalo (noturno), localização internacional. Segundo nossas políticas, transações 
internacionais exigem confirmação. Consultei o histórico: Caso #003 foi falso positivo 
(cliente em férias). Enviamos confirmação para Ana..."
```

## Estrutura do Projeto Agora

```
Colaborativos/
├── Colaborativo1.0.py          (agente principal com RAG)
├── 05_rag_agent.py             (referência - exemplo básico)
├── politicas_antifraude.txt    (dados: políticas do banco)
├── relatorios_fraude.txt       (dados: histórico de fraudes)
├── README.md                   (você está aqui)
├── .venv/                      (virtual environment)
├── vdb/                        (vector database - Chroma, criado ao rodar)
└── .git/
```

## Saída de Inicialização

Quando você roda o sistema, verá:

```
[SISTEMA] Inicializando RAG...
[RAG] Carregando arquivo: politicas_antifraude.txt
[RAG] Dividindo documento em chunks...
[RAG] 45 chunks criados de politicas_antifraude.txt
[RAG] Criando vector store com embeddings Ollama...
[RAG] ✓ Retriever criado com sucesso para politicas_antifraude.txt

[RAG] Carregando arquivo: relatorios_fraude.txt
[RAG] Dividindo documento em chunks...
[RAG] 28 chunks criados de relatorios_fraude.txt
[RAG] Criando vector store com embeddings Ollama...
[RAG] ✓ Retriever criado com sucesso para relatorios_fraude.txt

[SISTEMA] RAG inicializado!

Sistema colaborativo de detecção de fraude iniciado.
Paradigma de IA: IA preditiva + RAG (Retrieval-Augmented Generation).
Digite 'sair' para encerrar.
```

## Solução de Problemas

### "Dependências RAG não disponíveis"
```bash
# Solução:
pip install langchain-text-splitters langchain-chroma chromadb
python Colaborativo1.0.py
```

### "Arquivo não encontrado"
- Verifique que `politicas_antifraude.txt` e `relatorios_fraude.txt` estão na mesma pasta do script
- Use nomes exatos, case-sensitive

### "Ollama não respondendo"
```bash
# Terminal separado:
OLLAMA_MODELS=/home/gabi/ollama-models ollama serve

# Se modelo não existe:
/home/gabi/ollama/bin/ollama pull nomic-embed-text
```

### Vector store inválido ou corrompido
```bash
# Limpar cache e reconstruir:
rm -rf ./vdb
python Colaborativo1.0.py
# Sistema vai recriar os indices
```

## Próximos Passos (Sugestões)

1. **Adicionar mais dados**: Crie mais documentos de políticas ou casos de fraude
2. **Integrar PDFs reais**: Use o `search_pdf` para adicionar documentos em PDF
3. **Fine-tuning de chunks**: Ajuste `CHUNK_SIZE` e `CHUNK_OVERLAP` para melhor semântica
4. **Mudar modelo de embedding**: Teste outros modelos do Ollama (mistral, llama2, etc)
5. **Persistência**: Os vectores são salvos em `vdb/` - podem ser reusados entre execuções
6. **Logging**: Adicione logging estruturado para auditar decisões do agente

---

**Versão**: 1.1 com RAG completo
**Data**: 2025
**Status**: ✓ Funcional com Ollama
