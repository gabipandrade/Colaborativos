import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    RAG_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        RAG_AVAILABLE = True
    except ImportError:
        RecursiveCharacterTextSplitter = None
        Chroma = None
        OllamaEmbeddings = None
        RAG_AVAILABLE = False


# ============================================================
# Configuração do RAG
# ============================================================

PDF_PATH = "file.pdf"
POLITICAS_ANTIFRAUDE_PATH = "politicas_antifraude.txt"
RELATORIOS_FRAUDE_PATH = "relatorios_fraude.txt"

CHROMA_DIR = "./vdb"
POLITICAS_COLLECTION = "anti_fraud_policies"
RELATORIOS_COLLECTION = "fraud_reports"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVER_K = 5
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Retrievers (inicializados como None)
pdf_retriever = None
politicas_retriever = None
relatorios_retriever = None


# ============================================================
# Utilitários de recuperação
# ============================================================

def _normalizar_texto(texto: str) -> str:
    return texto.lower().strip()


def _pontuar_relevancia(pergunta: str, documento: str) -> int:
    pergunta_normalizada = _normalizar_texto(pergunta)
    documento_normalizado = _normalizar_texto(documento)
    termos = [termo for termo in pergunta_normalizada.replace(",", " ").replace(".", " ").split() if len(termo) > 2]

    pontuacao = 0
    for termo in termos:
        if termo in documento_normalizado:
            pontuacao += 2

    if pergunta_normalizada and pergunta_normalizada in documento_normalizado:
        pontuacao += 5

    return pontuacao


# ============================================================
# Funções de RAG para arquivos de texto
# ============================================================

def build_text_retriever(file_path: str, collection_name: str, chroma_dir: str = CHROMA_DIR):
    """
    Constrói um retriever vetorial para um arquivo de texto local.
    """
    if not RAG_AVAILABLE:
        print(f"[AVISO] Dependências RAG não disponíveis. Não será possível indexar {file_path}")
        return None

    if not os.path.exists(file_path):
        print(f"[AVISO] Arquivo {file_path} não encontrado.")
        return None

    try:
        print(f"[RAG] Carregando arquivo: {file_path}")
        
        # Carregar arquivo de texto
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Criar documento
        from langchain_core.documents import Document
        documents = [Document(page_content=content, metadata={"source": file_path})]
        
        print(f"[RAG] Dividindo documento em chunks...")
        
        # Dividir em chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[RAG] {len(chunks)} chunks criados de {file_path}")
        
        # Criar embeddings e vector store
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        print(f"[RAG] Criando vector store com embeddings Ollama...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
        
        print(f"[RAG] ✓ Retriever criado com sucesso para {file_path}")
        return retriever
        
    except Exception as erro:
        print(f"[ERRO] Falha ao criar retriever para {file_path}: {erro}")
        return None


def build_pdf_retriever(pdf_path: str = PDF_PATH):
    """
    Constrói um retriever vetorial para um PDF local, se as dependências existirem.
    """
    if not RAG_AVAILABLE:
        return None

    if not os.path.exists(pdf_path):
        return None

    try:
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(pages)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="tutorial_rag",
        )
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
    except Exception as erro:
        print(f"[ERRO] Falha ao criar retriever PDF: {erro}")
        return None


# Inicializar retrievers na startup
print("[SISTEMA] Inicializando RAG...")
pdf_retriever = build_pdf_retriever()
politicas_retriever = build_text_retriever(POLITICAS_ANTIFRAUDE_PATH, POLITICAS_COLLECTION)
relatorios_retriever = build_text_retriever(RELATORIOS_FRAUDE_PATH, RELATORIOS_COLLECTION)
print("[SISTEMA] RAG inicializado!")


def recuperar_contexto_banco(pergunta: str, limite: int = 3) -> str:
    """
    Recupera trechos relevantes do banco de dados simulado para compor o contexto do agente.
    """

    documentos = []

    for cliente, perfil in banco_de_dados["clientes"].items():
        documentos.append(
            (
                f"cliente:{cliente}",
                f"Cliente {cliente}. Cidade padrão: {perfil['cidade_padrao']}. "
                f"Valor médio: R$ {perfil['valor_medio']:.2f}. "
                f"Categorias comuns: {', '.join(perfil['categorias_comuns'])}. "
                f"Horário comum: {perfil['horario_comum']}."
            )
        )

    for indice, relatorio in enumerate(banco_de_dados["relatorios_fraude"], start=1):
        documentos.append(
            (
                f"relatorio:{indice}",
                f"Relatório {indice}. Cliente {relatorio['cliente']}. Cidade: {relatorio['cidade']}. "
                f"Categoria: {relatorio['categoria']}. Horário: {relatorio['horario']}. "
                f"Classificação: {relatorio['classificacao']}. Pontuação: {relatorio['pontuacao_risco']}."
            )
        )

    for indice, confirmacao in enumerate(banco_de_dados["confirmacoes_cliente"], start=1):
        documentos.append(
            (
                f"confirmacao:{indice}",
                f"Confirmação {indice}. Cliente {confirmacao['cliente']}. "
                f"Status: {confirmacao['status']}. Descrição: {confirmacao['descricao_transacao']}."
            )
        )

    documentos_pontuados = [
        (chave, texto, _pontuar_relevancia(pergunta, texto))
        for chave, texto in documentos
    ]
    documentos_pontuados = [item for item in documentos_pontuados if item[2] > 0]
    documentos_pontuados.sort(key=lambda item: item[2], reverse=True)

    if not documentos_pontuados:
        return (
            "Nenhum registro diretamente relevante foi encontrado no banco de dados simulado. "
            "Considere os perfis de clientes conhecidos e o histórico geral disponível."
        )

    partes = ["Contexto recuperado do banco de dados:"]
    for chave, texto, pontuacao in documentos_pontuados[:limite]:
        partes.append(f"- [{chave}] (relevância {pontuacao}): {texto}")

    return "\n".join(partes)


def extrair_ultima_mensagem_usuario(mensagens):
    for mensagem in reversed(mensagens):
        if isinstance(mensagem, HumanMessage):
            return mensagem.content
    return ""


# ============================================================
# Banco de dados simulado
# ============================================================

banco_de_dados = {
    "clientes": {
        "Joao": {
            "cidade_padrao": "Sao Carlos",
            "valor_medio": 120.0,
            "categorias_comuns": ["mercado", "farmacia", "restaurante"],
            "horario_comum": "diurno"
        },
        "Ana": {
            "cidade_padrao": "Campinas",
            "valor_medio": 250.0,
            "categorias_comuns": ["roupas", "mercado", "transporte"],
            "horario_comum": "diurno"
        },
        "Carlos": {
            "cidade_padrao": "Sao Paulo",
            "valor_medio": 500.0,
            "categorias_comuns": ["eletronicos", "restaurante", "viagem"],
            "horario_comum": "noturno"
        }
    },
    "relatorios_fraude": [],
    "confirmacoes_cliente": []
}


# ============================================================
# Função auxiliar: classificação preditiva simples
# ============================================================

def classificar_transacao(cliente, valor, cidade, categoria, horario):
    """
    Simula uma IA preditiva baseada em agrupamentos simples.
    A ideia é comparar a transação atual com o padrão histórico do cliente.
    """

    if cliente not in banco_de_dados["clientes"]:
        return {
            "status": "suspeita",
            "pontuacao_risco": 80,
            "motivos": [
                "Cliente não encontrado no banco de dados histórico.",
                "Não há padrão de comportamento para comparação."
            ],
            "acao_predita": "fraude"
        }

    perfil = banco_de_dados["clientes"][cliente]

    risco = 0
    motivos = []

    # Comparação por cidade
    if cidade != perfil["cidade_padrao"]:
        risco += 30
        motivos.append(
            f"Cidade diferente do padrão. Esperado: {perfil['cidade_padrao']}, recebido: {cidade}."
        )

    # Comparação por valor
    if valor > perfil["valor_medio"] * 3:
        risco += 35
        motivos.append(
            f"Valor muito acima da média. Média: R$ {perfil['valor_medio']:.2f}, recebido: R$ {valor:.2f}."
        )

    # Comparação por categoria
    if categoria not in perfil["categorias_comuns"]:
        risco += 20
        motivos.append(
            f"Categoria incomum para o cliente. Categoria recebida: {categoria}."
        )

    # Comparação por horário
    if horario != perfil["horario_comum"]:
        risco += 15
        motivos.append(
            f"Horário diferente do padrão. Esperado: {perfil['horario_comum']}, recebido: {horario}."
        )

    if risco >= 50:
        status = "suspeita"
        acao_predita = "fraude"
    else:
        status = "normal"
        acao_predita = "compra legítima"

    if not motivos:
        motivos.append("A transação coincide com o padrão histórico do cliente.")

    return {
        "status": status,
        "pontuacao_risco": risco,
        "motivos": motivos,
        "acao_predita": acao_predita
    }


# ============================================================
# Ferramentas do React Agent
# ============================================================

@tool
def verificar_transacao(cliente: str, valor: float, cidade: str, categoria: str, horario: str) -> str:
    """
    Verifica se uma transação bancária, PIX ou compra online é suspeita de fraude.

    Use esta ferramenta quando o usuário informar uma transação que precisa ser analisada.
    Parâmetros:
    - cliente: nome do cliente
    - valor: valor da transação
    - cidade: cidade onde ocorreu a transação
    - categoria: categoria da compra, como mercado, farmacia, eletronicos, viagem
    - horario: diurno ou noturno
    """

    resultado = classificar_transacao(cliente, valor, cidade, categoria, horario)

    texto = "Resultado da verificação da IA preditiva:\n"
    texto += f"Cliente: {cliente}\n"
    texto += f"Valor: R$ {valor:.2f}\n"
    texto += f"Cidade: {cidade}\n"
    texto += f"Categoria: {categoria}\n"
    texto += f"Horário: {horario}\n\n"

    texto += f"Ação predita pela IA: {resultado['acao_predita']}\n"
    texto += f"Pontuação de risco: {resultado['pontuacao_risco']}/100\n"
    texto += f"Classificação: {resultado['status'].upper()}\n\n"

    texto += "Justificativa baseada no padrão aprendido:\n"
    for motivo in resultado["motivos"]:
        texto += f"- {motivo}\n"

    if resultado["status"] == "suspeita":
        texto += (
            "\nConclusão: a transação não coincide com o padrão real esperado para o cliente. "
            "Ela deve ser tratada como possível fraude e enviada para confirmação do cliente."
        )
    else:
        texto += (
            "\nConclusão: a transação coincide com o padrão real esperado. "
            "Não há indícios fortes de fraude."
        )

    return texto


@tool
def gerar_relatorio_banco(cliente: str, valor: float, cidade: str, categoria: str, horario: str) -> str:
    """
    Gera um relatório para o banco explicando por que a transação foi classificada como suspeita ou normal.

    Use esta ferramenta quando for necessário registrar a justificativa da decisão para o banco.
    """

    resultado = classificar_transacao(cliente, valor, cidade, categoria, horario)

    relatorio = {
        "cliente": cliente,
        "valor": valor,
        "cidade": cidade,
        "categoria": categoria,
        "horario": horario,
        "acao_predita": resultado["acao_predita"],
        "pontuacao_risco": resultado["pontuacao_risco"],
        "classificacao": resultado["status"],
        "motivos": resultado["motivos"]
    }

    banco_de_dados["relatorios_fraude"].append(relatorio)

    texto = "Relatório enviado ao banco:\n"
    texto += f"Cliente: {cliente}\n"
    texto += f"Classificação: {resultado['status'].upper()}\n"
    texto += f"Pontuação de risco: {resultado['pontuacao_risco']}/100\n"
    texto += "Fatores considerados pela IA:\n"

    for motivo in resultado["motivos"]:
        texto += f"- {motivo}\n"

    return texto


@tool
def solicitar_confirmacao_cliente(cliente: str, descricao_transacao: str) -> str:
    """
    Solicita ao cliente uma confirmação de que a transação suspeita foi realmente feita por ele.

    Use esta ferramenta quando a IA identificar uma transação suspeita e for necessário confirmar com o cliente.
    """

    mensagem = {
        "cliente": cliente,
        "descricao_transacao": descricao_transacao,
        "status": "aguardando confirmação"
    }

    banco_de_dados["confirmacoes_cliente"].append(mensagem)

    return (
        f"Confirmação enviada ao cliente {cliente}. "
        f"Mensagem: Identificamos uma transação suspeita: {descricao_transacao}. "
        f"Confirme se essa compra foi realizada por você."
    )


@tool
def consultar_banco_de_dados() -> str:
    """
    Consulta o banco de dados simulado, incluindo perfis de clientes, relatórios e confirmações pendentes.

    Use esta ferramenta quando o usuário pedir para visualizar os dados armazenados.
    """

    texto = "Banco de dados simulado:\n\n"

    texto += "Perfis de clientes:\n"
    for cliente, perfil in banco_de_dados["clientes"].items():
        texto += f"- {cliente}: {perfil}\n"

    texto += "\nRelatórios de fraude registrados:\n"
    if banco_de_dados["relatorios_fraude"]:
        for i, relatorio in enumerate(banco_de_dados["relatorios_fraude"], start=1):
            texto += f"{i}. {relatorio}\n"
    else:
        texto += "Nenhum relatório registrado.\n"

    texto += "\nConfirmações enviadas ao cliente:\n"
    if banco_de_dados["confirmacoes_cliente"]:
        for i, confirmacao in enumerate(banco_de_dados["confirmacoes_cliente"], start=1):
            texto += f"{i}. {confirmacao}\n"
    else:
        texto += "Nenhuma confirmação enviada.\n"

    return texto


@tool
def consultar_rag_banco_de_dados(pergunta: str) -> str:
    """
    Consulta o banco de dados simulado com recuperação de contexto relevante.

    Use esta ferramenta quando for necessário obter trechos específicos do banco de dados
    que ajudem a responder uma pergunta ou analisar uma transação.
    """

    return recuperar_contexto_banco(pergunta)


@tool
def search_pdf(query: str) -> str:
    """
    Busca informações relevantes em um PDF local indexado via RAG.

    Use esta ferramenta quando o usuário perguntar sobre o conteúdo de um documento PDF.
    """

    if pdf_retriever is None:
        if not RAG_AVAILABLE:
            return (
                "RAG PDF indisponível porque as dependências necessárias não estão instaladas. "
                "O agente continua funcionando com o banco simulado."
            )

        return (
            f"Nenhum PDF foi indexado. Coloque um arquivo em {PDF_PATH} para ativar a busca."
        )

    try:
        docs = pdf_retriever.invoke(query)
    except Exception as erro:
        return f"Falha ao consultar o PDF indexado: {erro}"

    if not docs:
        return "Nenhuma informação relevante foi encontrada no PDF indexado."

    resultados = []
    for indice, doc in enumerate(docs, start=1):
        pagina = doc.metadata.get("page", "?")
        resultados.append(f"[Trecho {indice}, página {pagina}]:\n{doc.page_content}")

    return "\n\n".join(resultados)


@tool
def consultar_politicas_antifraude(pergunta: str) -> str:
    """
    Consulta o documento de POLÍTICAS ANTIFRAUDE do banco usando RAG.
    
    Use esta ferramenta para buscar informações sobre:
    - Definições de fraude
    - Limites de transação
    - Indicadores de risco
    - Padrões de uso esperados
    - Procedimentos de verificação
    - Períodos de risco elevado
    
    Exemplo: "Qual é o limite de compra online para cliente padrão?"
    """
    
    if politicas_retriever is None:
        return (
            f"Retriever de políticas antifraude não está disponível. "
            f"Verifique se o arquivo {POLITICAS_ANTIFRAUDE_PATH} existe e as dependências RAG estão instaladas."
        )
    
    try:
        docs = politicas_retriever.invoke(pergunta)
    except Exception as erro:
        return f"Falha ao consultar políticas antifraude: {erro}"
    
    if not docs:
        return "Nenhuma informação relevante foi encontrada nas políticas antifraude."
    
    resultados = [f"Contexto das políticas antifraude para sua pergunta:\n"]
    for indice, doc in enumerate(docs, start=1):
        resultados.append(f"[Informação {indice}]:\n{doc.page_content}\n")
    
    return "\n".join(resultados)


@tool
def consultar_relatorios_fraude(pergunta: str) -> str:
    """
    Consulta o documento de RELATÓRIOS HISTÓRICOS DE FRAUDE usando RAG.
    
    Use esta ferramenta para buscar informações sobre:
    - Casos históricos de fraude
    - Padrões de fraude detectados
    - Tipos de fraude (identidade, cartão clonado, etc)
    - Resultados de investigações passadas
    - Lições aprendidas do histórico
    
    Exemplo: "Quais fraudes foram detectadas por mudança de localização?"
    """
    
    if relatorios_retriever is None:
        return (
            f"Retriever de relatórios de fraude não está disponível. "
            f"Verifique se o arquivo {RELATORIOS_FRAUDE_PATH} existe e as dependências RAG estão instaladas."
        )
    
    try:
        docs = relatorios_retriever.invoke(pergunta)
    except Exception as erro:
        return f"Falha ao consultar relatórios de fraude: {erro}"
    
    if not docs:
        return "Nenhuma informação relevante foi encontrada nos relatórios de fraude."
    
    resultados = [f"Contexto dos relatórios de fraude para sua pergunta:\n"]
    for indice, doc in enumerate(docs, start=1):
        resultados.append(f"[Relatório {indice}]:\n{doc.page_content}\n")
    
    return "\n".join(resultados)


tools = [
    verificar_transacao,
    gerar_relatorio_banco,
    solicitar_confirmacao_cliente,
    consultar_banco_de_dados,
    consultar_rag_banco_de_dados,
    search_pdf,
    consultar_politicas_antifraude,
    consultar_relatorios_fraude
]


# ============================================================
# Modelo local com Ollama
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

llm_com_tools = llm.bind_tools(tools)


# ============================================================
# Nó do agente
# ============================================================

def agente(state: MessagesState):
    """
    Nó principal do React Agent.

    O agente decide se deve:
    1. responder diretamente;
    2. chamar uma ferramenta para verificar transação;
    3. gerar relatório ao banco;
    4. solicitar confirmação ao cliente.
    """

    system_prompt = SystemMessage(
        content=(
            "Você é um agente colaborativo de detecção de fraude bancária.\n\n"

            "O sistema possui dois tipos principais de usuários: Banco e Cliente.\n"
            "O projeto é detectar fraudes em transações bancárias, PIX ou compras online.\n\n"

            "Fluxo esperado do sistema:\n"
            "1. O cliente realiza uma ação de compra.\n"
            "2. O banco pede para a IA verificar essa ação de compra.\n"
            "3. A IA consulta o banco de dados histórico.\n"
            "4. A IA prediz se a ação parece legítima ou suspeita.\n"
            "5. Se a predição coincidir com o padrão real esperado, a transação é considerada normal.\n"
            "6. Se a predição não coincidir com o padrão real esperado, a transação é considerada suspeita.\n"
            "7. Em caso de suspeita, o sistema deve solicitar confirmação ao cliente e gerar relatório ao banco.\n\n"

            "Ferramentas disponíveis:\n"
            "- verificar_transacao: use sempre que houver dados de uma compra/transação para análise.\n"
            "- gerar_relatorio_banco: use quando precisar registrar a justificativa para o banco.\n"
            "- solicitar_confirmacao_cliente: use quando a transação for suspeita.\n"
            "- consultar_banco_de_dados: use quando o usuário pedir para visualizar os dados internos.\n"
            "- consultar_rag_banco_de_dados: use para buscar contexto relevante no banco simulado antes de responder.\n"
            "- consultar_politicas_antifraude: use para buscar informações sobre políticas, limites e indicadores de risco.\n"
            "- consultar_relatorios_fraude: use para buscar casos históricos e padrões de fraude detectados.\n"
            "- search_pdf: use para consultar um PDF indexado quando necessário.\n\n"

            "Importante:\n"
            "✓ Quando uma transação for claramente suspeita, use as ferramentas para verificar, gerar relatório e solicitar confirmação.\n"
            "✓ Consulte SEMPRE o documento de POLÍTICAS ANTIFRAUDE para respaldar suas análises.\n"
            "✓ Use o documento de RELATÓRIOS FRAUDE para comparar com padrões históricos.\n"
            "✓ Se a pergunta mencionar políticas, limites, ou padrões de risco, use consultar_politicas_antifraude.\n"
            "✓ Se a pergunta mencionar fraudes históricas ou padrões, use consultar_relatorios_fraude.\n"
            "✓ Quando a transação parecer normal, explique que ela coincide com o padrão esperado.\n"
            "✓ Responda em português brasileiro de forma clara, organizada e objetiva."
        )
    )

    contexto_rag = recuperar_contexto_banco(extrair_ultima_mensagem_usuario(state["messages"]))
    contexto_prompt = SystemMessage(
        content=(
            "Contexto recuperado automaticamente do banco de dados simulado:\n"
            f"{contexto_rag}"
        )
    )

    resposta = llm_com_tools.invoke([system_prompt, contexto_prompt] + state["messages"])

    return {"messages": [resposta]}


# ============================================================
# Construção do grafo LangGraph
# ============================================================

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agente", agente)
graph_builder.add_node("ferramentas", ToolNode(tools))

graph_builder.add_edge(START, "agente")

graph_builder.add_conditional_edges(
    "agente",
    tools_condition,
    {
        "tools": "ferramentas",
        "__end__": "__end__"
    }
)

graph_builder.add_edge("ferramentas", "agente")

graph = graph_builder.compile()


# ============================================================
# Loop principal
# ============================================================

def main():
    print("Sistema colaborativo de detecção de fraude iniciado.")
    print("Paradigma de IA: IA preditiva.")
    print("Digite 'sair' para encerrar.\n")

    historico = []

    while True:
        entrada = input("Usuário: ")

        if entrada.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o sistema.")
            break

        historico.append(HumanMessage(content=entrada))

        try:
            resultado = graph.invoke({"messages": historico})
            historico = resultado["messages"]

            ultima_resposta = historico[-1]

            print("\nAgente:")
            print(ultima_resposta.content)
            print()

        except Exception as erro:
            print("\nErro ao executar o agente:")
            print(erro)
            print(
                "\nVerifique se o Ollama está rodando em outro terminal com:\n"
                "OLLAMA_MODELS=/home/rafael/ollama-models ollama serve\n"
            )


if __name__ == "__main__":
    main()