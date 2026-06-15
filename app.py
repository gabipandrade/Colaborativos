import os
import hashlib
from datetime import datetime

import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from modelos_dados import (
        DF_TRANSACOES,
        MODELO_CLUSTER,
        MODELO_ISOLATION,
        PERFIS_CLUSTERS,
        PERFIS_CONTAS,
        TRANSACOES_CONSULTA,
        analisar_nova_transacao,
        analisar_transacao_por_id,
        base_disponivel,
        consultar_base_modelos,
        formatar_resultado as formatar_resultado_modelos,
    )

    MODELOS_EXTERNOS_DISPONIVEIS = base_disponivel()
    ERRO_MODELOS_EXTERNOS = None
except Exception as erro_modelos:
    DF_TRANSACOES = None
    MODELO_CLUSTER = None
    MODELO_ISOLATION = None
    PERFIS_CLUSTERS = None
    PERFIS_CONTAS = {}
    TRANSACOES_CONSULTA = []
    MODELOS_EXTERNOS_DISPONIVEIS = False
    ERRO_MODELOS_EXTERNOS = str(erro_modelos)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    RAG_AVAILABLE = True
except ImportError:
    PyPDFLoader = None
    Document = None
    RecursiveCharacterTextSplitter = None
    Chroma = None
    OllamaEmbeddings = None
    RAG_AVAILABLE = False


# ============================================================
# Configuracao do RAG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLITICAS_ANTIFRAUDE_PATH = os.path.join(BASE_DIR, "politicas_antifraude.txt")
RELATORIOS_FRAUDE_PATH = os.path.join(BASE_DIR, "relatorios_fraude.txt")
PDF_DIR = os.path.join(BASE_DIR, "documentos_pdf")
CHROMA_DIR = os.path.join(BASE_DIR, "vdb")
POLITICAS_COLLECTION = "streamlit_anti_fraud_policies"
RELATORIOS_COLLECTION = "streamlit_fraud_reports"
PDF_COLLECTION = "streamlit_uploaded_pdfs"
MAX_PDF_DOCUMENTS = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVER_K = 5
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


@st.cache_resource(show_spinner=False)
def build_text_retriever(file_path: str, collection_name: str):
    if not RAG_AVAILABLE:
        return None

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as arquivo:
            content = arquivo.read()

        documents = [Document(page_content=content, metadata={"source": file_path})]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
        )
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
    except Exception as erro:
        st.session_state[f"rag_error_{collection_name}"] = str(erro)
        return None


def nome_arquivo_seguro(nome_arquivo: str) -> str:
    nome_base = os.path.basename(nome_arquivo).replace(" ", "_")
    return "".join(
        caractere
        for caractere in nome_base
        if caractere.isalnum() or caractere in ("-", "_", ".")
    ) or "documento.pdf"


def listar_pdfs_disponiveis() -> list[str]:
    if not os.path.isdir(PDF_DIR):
        return []

    arquivos = [
        os.path.join(PDF_DIR, nome)
        for nome in sorted(os.listdir(PDF_DIR))
        if nome.lower().endswith(".pdf")
    ]
    return arquivos[:MAX_PDF_DOCUMENTS]


def salvar_pdfs_enviados(arquivos_enviados) -> list[str]:
    if not arquivos_enviados:
        return listar_pdfs_disponiveis()

    os.makedirs(PDF_DIR, exist_ok=True)
    caminhos_salvos = []
    for arquivo in arquivos_enviados[:MAX_PDF_DOCUMENTS]:
        nome_seguro = nome_arquivo_seguro(arquivo.name)
        caminho = os.path.join(PDF_DIR, nome_seguro)
        with open(caminho, "wb") as destino:
            destino.write(arquivo.getbuffer())
        caminhos_salvos.append(caminho)

    return caminhos_salvos


@st.cache_resource(show_spinner=False)
def build_pdf_retriever(pdf_paths: tuple[str, ...]):
    if not RAG_AVAILABLE or PyPDFLoader is None:
        return None

    caminhos_validos = [caminho for caminho in pdf_paths if os.path.exists(caminho)]
    if not caminhos_validos:
        return None

    try:
        documentos = []
        for caminho in caminhos_validos[:MAX_PDF_DOCUMENTS]:
            paginas = PyPDFLoader(caminho).load()
            for pagina in paginas:
                pagina.metadata["source"] = caminho
            documentos.extend(paginas)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documentos)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        assinatura = hashlib.sha256("|".join(caminhos_validos).encode("utf-8")).hexdigest()[:12]
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=f"{PDF_COLLECTION}_{assinatura}",
        )
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
    except Exception as erro:
        st.session_state["rag_error_pdfs"] = str(erro)
        return None


def consultar_documento_rag(
    pergunta: str,
    retriever,
    nome_documento: str,
    caminho_documento: str,
    incluir_busca_textual_atualizada: bool = False,
) -> str:
    if retriever is None:
        return buscar_texto_local(pergunta, caminho_documento, nome_documento)

    try:
        docs = retriever.invoke(pergunta)
    except Exception as erro:
        return f"Falha ao consultar {nome_documento}: {erro}\n\n{buscar_texto_local(pergunta, caminho_documento, nome_documento)}"

    if not docs:
        return buscar_texto_local(pergunta, caminho_documento, nome_documento)

    resultados = [f"Contexto recuperado de {nome_documento}:"]
    for indice, doc in enumerate(docs, start=1):
        resultados.append(f"[Trecho {indice}]\n{doc.page_content}")

    if incluir_busca_textual_atualizada:
        resultados.append(
            "Busca textual atualizada no arquivo persistente:\n"
            f"{buscar_texto_local(pergunta, caminho_documento, nome_documento)}"
        )

    return "\n\n".join(resultados)


def consultar_pdfs_rag(pergunta: str, pdf_paths: list[str] | tuple[str, ...] | None = None) -> str:
    caminhos = tuple((pdf_paths or listar_pdfs_disponiveis())[:MAX_PDF_DOCUMENTS])
    if not caminhos:
        return "Nenhum PDF disponível. Envie até 5 arquivos PDF na aba Agente para consultar documentos."

    retriever = build_pdf_retriever(caminhos)
    nomes = ", ".join(os.path.basename(caminho) for caminho in caminhos)
    if retriever is None:
        erro = st.session_state.get("rag_error_pdfs")
        detalhe = f" Detalhe: {erro}" if erro else ""
        return f"RAG de PDFs indisponível para os arquivos: {nomes}.{detalhe}"

    try:
        docs = retriever.invoke(pergunta)
    except Exception as erro:
        return f"Falha ao consultar PDFs: {erro}"

    if not docs:
        return f"Nenhum trecho relevante foi encontrado nos PDFs: {nomes}."

    resultados = [f"Contexto recuperado dos PDFs ({nomes}):"]
    for indice, doc in enumerate(docs, start=1):
        fonte = os.path.basename(str(doc.metadata.get("source", "PDF")))
        pagina = doc.metadata.get("page")
        referencia = f"{fonte}, página {pagina + 1}" if isinstance(pagina, int) else fonte
        resultados.append(f"[Trecho {indice} - {referencia}]\n{doc.page_content}")

    return "\n\n".join(resultados)


politicas_retriever = build_text_retriever(POLITICAS_ANTIFRAUDE_PATH, POLITICAS_COLLECTION)
relatorios_retriever = build_text_retriever(RELATORIOS_FRAUDE_PATH, RELATORIOS_COLLECTION)


def buscar_texto_local(pergunta: str, caminho_documento: str, nome_documento: str) -> str:
    if not os.path.exists(caminho_documento):
        return f"{nome_documento} indisponivel. O arquivo nao foi encontrado em {caminho_documento}."

    with open(caminho_documento, "r", encoding="utf-8") as arquivo:
        texto = arquivo.read()

    blocos = [bloco.strip() for bloco in texto.split("\n\n") if bloco.strip()]
    termos = [
        termo.lower()
        for termo in pergunta.replace(",", " ").replace(".", " ").split()
        if len(termo) > 2
    ]

    pontuados = []
    for bloco in blocos:
        bloco_normalizado = bloco.lower()
        pontuacao = sum(bloco_normalizado.count(termo) for termo in termos)
        pontuados.append((pontuacao, bloco))

    pontuados.sort(key=lambda item: item[0], reverse=True)
    selecionados = [bloco for pontuacao, bloco in pontuados[:RETRIEVER_K] if pontuacao > 0]

    if not selecionados:
        selecionados = blocos[:RETRIEVER_K]

    resultados = [f"Contexto recuperado de {nome_documento} por busca textual:"]
    for indice, bloco in enumerate(selecionados, start=1):
        resultados.append(f"[Trecho {indice}]\n{bloco}")

    return "\n\n".join(resultados)


def criar_id_relatorio(relatorio: dict) -> str:
    partes = [
        str(relatorio.get("tipo", "analise")),
        str(relatorio.get("cliente", "")),
        str(relatorio.get("valor", "")),
        str(relatorio.get("cidade", "")),
        str(relatorio.get("categoria", "")),
        str(relatorio.get("horario", "")),
        str(relatorio.get("classificacao", "")),
        str(relatorio.get("resposta_cliente", "")),
        str(relatorio.get("pontuacao_risco", relatorio.get("pontuacao_risco_final", ""))),
    ]
    conteudo = "|".join(partes)
    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()[:16]


def formatar_relatorio_persistente(relatorio: dict) -> str:
    relatorio_id = relatorio.get("id_relatorio") or criar_id_relatorio(relatorio)
    motivos = relatorio.get("motivos", [])
    linhas = [
        "",
        "---",
        "",
        f"RELATÓRIO DINÂMICO #{relatorio_id}",
        f"Data de registro: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tipo: {relatorio.get('tipo', 'analise de transacao')}",
        f"Cliente: {relatorio.get('cliente', 'indisponivel')}",
        f"Valor: R$ {float(relatorio.get('valor', 0.0)):.2f}",
        f"Categoria: {relatorio.get('categoria', 'indisponivel')}",
        f"Localização: {relatorio.get('cidade', 'indisponivel')}",
        f"Horário: {relatorio.get('horario', 'indisponivel')}",
        f"Classificação: {relatorio.get('classificacao', 'indisponivel')}",
    ]

    if "pontuacao_risco" in relatorio:
        linhas.append(f"Score: {relatorio['pontuacao_risco']}/100")
    if "pontuacao_risco_inicial" in relatorio:
        linhas.append(f"Score inicial: {relatorio['pontuacao_risco_inicial']}/100")
    if "pontuacao_risco_final" in relatorio:
        linhas.append(f"Score final: {relatorio['pontuacao_risco_final']}/100")
    if relatorio.get("resposta_cliente"):
        linhas.append(f"Resposta do cliente: {relatorio['resposta_cliente']}")
    if relatorio.get("acao_recomendada"):
        linhas.append(f"Ação recomendada: {relatorio['acao_recomendada']}")
    if relatorio.get("acao_predita"):
        linhas.append(f"Ação predita: {relatorio['acao_predita']}")

    linhas.append("Motivos:")
    if motivos:
        linhas.extend(f"- {motivo}" for motivo in motivos)
    else:
        linhas.append("- Nenhum motivo registrado.")

    linhas.append("")
    return "\n".join(linhas)


def persistir_relatorio_fraude(relatorio: dict):
    relatorio_id = relatorio.get("id_relatorio") or criar_id_relatorio(relatorio)
    relatorio["id_relatorio"] = relatorio_id

    conteudo_atual = ""
    if os.path.exists(RELATORIOS_FRAUDE_PATH):
        with open(RELATORIOS_FRAUDE_PATH, "r", encoding="utf-8") as arquivo:
            conteudo_atual = arquivo.read()

    marcador = f"RELATÓRIO DINÂMICO #{relatorio_id}"
    if marcador in conteudo_atual:
        return relatorio_id

    with open(RELATORIOS_FRAUDE_PATH, "a", encoding="utf-8") as arquivo:
        arquivo.write(formatar_relatorio_persistente(relatorio))

    return relatorio_id


def registrar_relatorio_fraude(relatorio: dict):
    relatorio_id = persistir_relatorio_fraude(relatorio)
    relatorio["id_relatorio"] = relatorio_id

    for relatorio_existente in banco_de_dados["relatorios_fraude"]:
        if relatorio_existente.get("id_relatorio") == relatorio_id:
            return relatorio_existente

    banco_de_dados["relatorios_fraude"].append(relatorio)
    return relatorio


# ============================================================
# Banco de dados simulado GLOBAL
# ============================================================

banco_de_dados = {
    "clientes": PERFIS_CONTAS,
    "relatorios_fraude": [],
    "confirmacoes_cliente": [],
    "historico_cliente": [],
    "clientes_modelos": PERFIS_CONTAS,
    "transacoes_modelos": TRANSACOES_CONSULTA,
}


if "historico" not in st.session_state:
    st.session_state.historico = []
if "relatorios_fraude" not in st.session_state:
    st.session_state.relatorios_fraude = []
if "confirmacoes_cliente" not in st.session_state:
    st.session_state.confirmacoes_cliente = []
if "historico_cliente" not in st.session_state:
    st.session_state.historico_cliente = []
if "eventos_colaborativos" not in st.session_state:
    st.session_state.eventos_colaborativos = []
if "mensagens_participantes" not in st.session_state:
    st.session_state.mensagens_participantes = []
if "ultima_analise_banco" not in st.session_state:
    st.session_state.ultima_analise_banco = None
if "ultimo_resultado_modelos" not in st.session_state:
    st.session_state.ultimo_resultado_modelos = None

banco_de_dados["relatorios_fraude"] = st.session_state.relatorios_fraude
banco_de_dados["confirmacoes_cliente"] = st.session_state.confirmacoes_cliente
banco_de_dados["historico_cliente"] = st.session_state.historico_cliente


def valores_unicos_base(coluna: str, fallback: list[str]) -> list[str]:
    if DF_TRANSACOES is None or coluna not in DF_TRANSACOES.columns:
        return fallback

    valores = sorted(DF_TRANSACOES[coluna].dropna().astype(str).unique().tolist())
    return valores or fallback


def valor_texto_padrao(chave: str, valor_padrao: str) -> str:
    if chave not in st.session_state:
        st.session_state[chave] = valor_padrao
    return st.session_state[chave]


def contas_disponiveis() -> list[str]:
    contas = list(banco_de_dados["clientes"].keys())
    if contas:
        return contas
    return ["CONTA_DESCONHECIDA"]


def perfil_conta(account_id: str) -> dict:
    return banco_de_dados["clientes"].get(account_id, {})


def horario_para_hora(horario: str) -> int:
    return 10 if horario == "diurno" else 22


def analisar_transacao_banco_com_modelos(
    cliente: str,
    valor: float,
    cidade: str,
    categoria: str,
    horario: str,
    canal: str,
    idade: int,
    ocupacao: str,
    duracao: float,
    tentativas_login: int,
    saldo: float,
    dispositivo: str,
    ip: str,
    comerciante: str,
) -> dict:
    if not MODELOS_EXTERNOS_DISPONIVEIS:
        return classificar_transacao(cliente, valor, cidade, categoria, horario)

    data_transacao = f"2026-06-15 {horario_para_hora(horario):02d}:00:00"
    df_nova, resultado_modelo = analisar_nova_transacao(
        account_id=cliente,
        transaction_amount=valor,
        transaction_type=categoria,
        location=cidade,
        channel=canal,
        customer_age=idade,
        customer_occupation=ocupacao,
        transaction_duration=duracao,
        login_attempts=tentativas_login,
        account_balance=saldo,
        device_id=dispositivo,
        ip_address=ip,
        merchant_id=comerciante,
        transaction_date=data_transacao,
    )

    motivos = resultado_modelo.get("motivos", [])
    if resultado_modelo.get("cluster_predito") is not None:
        motivos.append(f"Cluster predito pelos modelos: {resultado_modelo['cluster_predito']}.")

    return {
        "status": resultado_modelo.get("status", "normal"),
        "pontuacao_risco": int(resultado_modelo.get("pontuacao_risco", 0)),
        "motivos": motivos or ["Análise realizada pelos modelos da base."],
        "acao_predita": resultado_modelo.get("acao_predita", "transação analisada pelos modelos"),
        "resultado_modelo": resultado_modelo,
        "df_modelo": df_nova,
    }


def registrar_evento(origem: str, destino: str, descricao: str):
    st.session_state.eventos_colaborativos.append({
        "origem": origem,
        "destino": destino,
        "descricao": descricao,
    })


def registrar_mensagem(origem: str, destino: str, cliente: str, mensagem: str):
    st.session_state.mensagens_participantes.append({
        "origem": origem,
        "destino": destino,
        "cliente": cliente,
        "mensagem": mensagem,
        "status": "enviada",
    })
    registrar_evento(origem, destino, f"Mensagem sobre {cliente}: {mensagem}")


def registrar_confirmacao_pedida_pelo_banco(cliente: str, valor: float, cidade: str, categoria: str, horario: str):
    for confirmacao in banco_de_dados["confirmacoes_cliente"]:
        transacao = confirmacao.get("transacao", {})
        mesma_transacao = (
            confirmacao.get("cliente") == cliente
            and float(transacao.get("valor", 0.0)) == float(valor)
            and transacao.get("cidade") == cidade
            and transacao.get("categoria") == categoria
            and transacao.get("horario") == horario
        )
        if mesma_transacao and confirmacao.get("status") == "aguardando confirmação":
            return confirmacao

    descricao = (
        f"Compra de R$ {valor:.2f} em {cidade}, categoria {categoria}, "
        f"no horário {horario}."
    )
    mensagem = {
        "cliente": cliente,
        "descricao_transacao": descricao,
        "status": "aguardando confirmação",
        "origem": "Banco",
        "transacao": {
            "valor": valor,
            "cidade": cidade,
            "categoria": categoria,
            "horario": horario,
        },
    }
    banco_de_dados["confirmacoes_cliente"].append(mensagem)
    registrar_mensagem(
        "Banco",
        cliente,
        cliente,
        f"Precisamos confirmar se você reconhece esta transação: {descricao}",
    )
    registrar_evento("Banco", cliente, f"Banco solicitou confirmação da transação: {descricao}")
    return mensagem


def atualizar_historico_cliente_apos_falso_positivo(cliente: str, valor: float, cidade: str, categoria: str, horario: str):
    if cliente not in banco_de_dados["clientes"]:
        return

    for evento in banco_de_dados["historico_cliente"]:
        if (
            evento.get("cliente") == cliente
            and evento.get("valor") == valor
            and evento.get("cidade") == cidade
            and evento.get("categoria") == categoria
            and evento.get("horario") == horario
        ):
            return

    perfil = banco_de_dados["clientes"][cliente]
    perfil["valor_medio"] = round((perfil["valor_medio"] * 0.8) + (valor * 0.2), 2)

    if categoria and categoria not in perfil["categorias_comuns"]:
        perfil["categorias_comuns"].append(categoria)

    banco_de_dados["historico_cliente"].append({
        "cliente": cliente,
        "valor": valor,
        "cidade": cidade,
        "categoria": categoria,
        "horario": horario,
        "resultado": "falso positivo confirmado pelo cliente",
    })


def recalcular_risco_apos_resposta(cliente: str, valor: float, cidade: str, categoria: str, horario: str, resposta_cliente: str):
    resultado_inicial = classificar_transacao(cliente, valor, cidade, categoria, horario)
    risco_inicial = resultado_inicial["pontuacao_risco"]
    motivos = list(resultado_inicial["motivos"])

    resposta_normalizada = resposta_cliente.lower().strip()
    cliente_confirmou = resposta_normalizada in ["confirmada pelo cliente", "confirmada", "sim", "reconhecida"]
    cliente_negou = resposta_normalizada in ["negada pelo cliente", "negada", "nao", "não", "nao reconhecida", "não reconhecida"]

    if cliente_confirmou:
        risco_final = max(0, risco_inicial - 45)
        classificacao_final = "normal" if risco_final < 50 else "revisao manual"
        acao_final = "compra legítima confirmada"
        motivos.append("Cliente confirmou que reconhece a transação; risco reduzido na reanálise.")
        if resultado_inicial["status"] == "suspeita":
            atualizar_historico_cliente_apos_falso_positivo(cliente, valor, cidade, categoria, horario)
            motivos.append("Histórico do cliente atualizado porque a suspeita inicial foi falso positivo.")
    elif cliente_negou:
        risco_final = min(100, risco_inicial + 35)
        classificacao_final = "fraude confirmada"
        acao_final = "bloquear transação e abrir contestação"
        motivos.append("Cliente negou reconhecer a transação; risco elevado na reanálise.")
    else:
        risco_final = risco_inicial
        classificacao_final = "pendente"
        acao_final = "aguardar validação operacional"
        motivos.append("Resposta do cliente não foi conclusiva; manter pendência para análise manual.")

    return {
        "cliente": cliente,
        "valor": valor,
        "cidade": cidade,
        "categoria": categoria,
        "horario": horario,
        "resposta_cliente": resposta_cliente,
        "pontuacao_risco_inicial": risco_inicial,
        "pontuacao_risco_final": risco_final,
        "classificacao_inicial": resultado_inicial["status"],
        "classificacao_final": classificacao_final,
        "acao_final": acao_final,
        "motivos": motivos,
    }


def gerar_relatorio_final_reanalise(reanalise: dict) -> dict:
    for relatorio_existente in banco_de_dados["relatorios_fraude"]:
        if (
            relatorio_existente.get("tipo") == "relatorio final de reanalise"
            and relatorio_existente.get("cliente") == reanalise["cliente"]
            and relatorio_existente.get("valor") == reanalise["valor"]
            and relatorio_existente.get("cidade") == reanalise["cidade"]
            and relatorio_existente.get("categoria") == reanalise["categoria"]
            and relatorio_existente.get("horario") == reanalise["horario"]
            and relatorio_existente.get("resposta_cliente") == reanalise["resposta_cliente"]
        ):
            return relatorio_existente

    relatorio = {
        "tipo": "relatorio final de reanalise",
        "cliente": reanalise["cliente"],
        "valor": reanalise["valor"],
        "cidade": reanalise["cidade"],
        "categoria": reanalise["categoria"],
        "horario": reanalise["horario"],
        "resposta_cliente": reanalise["resposta_cliente"],
        "pontuacao_risco_inicial": reanalise["pontuacao_risco_inicial"],
        "pontuacao_risco_final": reanalise["pontuacao_risco_final"],
        "classificacao_inicial": reanalise["classificacao_inicial"],
        "classificacao": reanalise["classificacao_final"],
        "acao_recomendada": reanalise["acao_final"],
        "motivos": reanalise["motivos"],
    }
    return registrar_relatorio_fraude(relatorio)


def processar_resposta_cliente(indice_confirmacao: int, resposta_cliente: str):
    confirmacao = banco_de_dados["confirmacoes_cliente"][indice_confirmacao]
    if confirmacao.get("status") != "aguardando confirmação":
        return confirmacao.get("reanalise")

    confirmacao["status"] = "finalizada"
    confirmacao["resposta_cliente"] = resposta_cliente

    transacao = confirmacao.get("transacao", {})
    reanalise = recalcular_risco_apos_resposta(
        confirmacao["cliente"],
        float(transacao.get("valor", 0.0)),
        transacao.get("cidade", ""),
        transacao.get("categoria", ""),
        transacao.get("horario", ""),
        resposta_cliente,
    )
    relatorio_final = gerar_relatorio_final_reanalise(reanalise)

    confirmacao["reanalise"] = reanalise
    confirmacao["relatorio_final"] = relatorio_final
    registrar_mensagem(
        confirmacao["cliente"],
        "Banco",
        confirmacao["cliente"],
        f"Resposta da confirmação: {resposta_cliente}.",
    )
    registrar_evento(
        confirmacao["cliente"],
        "Agente",
        (
            f"Resposta recebida ({resposta_cliente}); risco recalculado de "
            f"{reanalise['pontuacao_risco_inicial']} para {reanalise['pontuacao_risco_final']}."
        ),
    )
    registrar_evento("Agente", "Banco", f"Relatório final de reanálise gerado para {confirmacao['cliente']}.")

    ultima_analise = st.session_state.get("ultima_analise_banco")
    if ultima_analise:
        mesma_transacao = (
            ultima_analise.get("cliente") == confirmacao["cliente"]
            and float(ultima_analise.get("valor", 0.0)) == float(transacao.get("valor", 0.0))
            and ultima_analise.get("cidade") == transacao.get("cidade", "")
            and ultima_analise.get("categoria") == transacao.get("categoria", "")
            and ultima_analise.get("horario") == transacao.get("horario", "")
        )
        if mesma_transacao:
            ultima_analise["decisao_operacional"] = reanalise["acao_final"]
            ultima_analise["status_fluxo"] = "finalizada"
            ultima_analise["reanalise"] = reanalise

    return reanalise


# ============================================================
# Função auxiliar de classificação
# ============================================================

def classificar_transacao(cliente, valor, cidade, categoria, horario):
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

    if cidade != perfil["cidade_padrao"]:
        risco += 30
        motivos.append(
            f"Cidade diferente do padrão. Esperado: {perfil['cidade_padrao']}, recebido: {cidade}."
        )

    if valor > perfil["valor_medio"] * 3:
        risco += 35
        motivos.append(
            f"Valor muito acima da média. Média: R$ {perfil['valor_medio']:.2f}, recebido: R$ {valor:.2f}."
        )

    if categoria not in perfil["categorias_comuns"]:
        risco += 20
        motivos.append(
            f"Categoria incomum para o cliente. Categoria recebida: {categoria}."
        )

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
# Ferramentas do agente
# ============================================================

@tool
def verificar_transacao(cliente: str, valor: float, cidade: str, categoria: str, horario: str) -> str:
    """
    Verifica se uma transação bancária, PIX ou compra online é suspeita de fraude.
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
            "Ela deve ser tratada como suspeita. O banco deve decidir se bloqueia, autoriza ou pede confirmação ao cliente."
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
    """

    resultado = classificar_transacao(cliente, valor, cidade, categoria, horario)

    relatorio = {
        "tipo": "analise de transacao",
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

    registrar_relatorio_fraude(relatorio)
    registrar_evento("Agente", "Banco", f"Relatório gerado para {cliente}: {resultado['status'].upper()} ({resultado['pontuacao_risco']}/100).")

    texto = "Relatório enviado ao banco:\n"
    texto += f"Cliente: {cliente}\n"
    texto += f"Classificação: {resultado['status'].upper()}\n"
    texto += f"Pontuação de risco: {resultado['pontuacao_risco']}/100\n"
    texto += "Fatores considerados pela IA:\n"

    for motivo in resultado["motivos"]:
        texto += f"- {motivo}\n"

    return texto


@tool
def solicitar_confirmacao_cliente(
    cliente: str,
    descricao_transacao: str,
    valor: float = 0.0,
    cidade: str = "",
    categoria: str = "",
    horario: str = "",
) -> str:
    """
    Solicita ao cliente uma confirmação de que a transação suspeita foi realmente feita por ele.
    """

    mensagem = {
        "cliente": cliente,
        "descricao_transacao": descricao_transacao,
        "status": "aguardando confirmação",
        "transacao": {
            "valor": valor,
            "cidade": cidade,
            "categoria": categoria,
            "horario": horario,
        },
    }

    banco_de_dados["confirmacoes_cliente"].append(mensagem)
    registrar_evento("Agente", cliente, f"Solicitação de confirmação enviada: {descricao_transacao}")

    return (
        f"Confirmação enviada ao cliente {cliente}. "
        f"Mensagem: Identificamos uma transação suspeita: {descricao_transacao}. "
        f"Confirme se essa compra foi realizada por você."
    )


@tool
def contestar_transacao_cliente(
    cliente: str,
    valor: float,
    cidade: str,
    categoria: str,
    horario: str,
    resposta_cliente: str,
) -> str:
    """
    Reanalisa uma transação depois da resposta do cliente.

    Use quando o cliente confirmar ou negar a transação. A ferramenta recalcula o risco,
    atualiza o histórico em caso de falso positivo e gera um relatório final.
    """

    reanalise = recalcular_risco_apos_resposta(
        cliente,
        valor,
        cidade,
        categoria,
        horario,
        resposta_cliente,
    )
    gerar_relatorio_final_reanalise(reanalise)
    registrar_evento(
        "Agente",
        "Banco",
        f"Contestação reanalisada para {cliente}: {reanalise['classificacao_final']} ({reanalise['pontuacao_risco_final']}/100).",
    )

    texto = "Reanálise após resposta do cliente:\n"
    texto += f"Cliente: {cliente}\n"
    texto += f"Resposta do cliente: {resposta_cliente}\n"
    texto += f"Risco inicial: {reanalise['pontuacao_risco_inicial']}/100\n"
    texto += f"Risco final: {reanalise['pontuacao_risco_final']}/100\n"
    texto += f"Classificação final: {reanalise['classificacao_final'].upper()}\n"
    texto += f"Ação recomendada: {reanalise['acao_final']}\n"
    texto += "Motivos:\n"
    for motivo in reanalise["motivos"]:
        texto += f"- {motivo}\n"

    return texto


@tool
def consultar_politicas_antifraude(pergunta: str) -> str:
    """
    Consulta o documento de politicas antifraude do banco usando RAG.

    Use para buscar limites, criterios de risco, definicoes de fraude,
    procedimentos de verificacao e regras operacionais do banco.
    """

    return consultar_documento_rag(
        pergunta,
        politicas_retriever,
        "politicas antifraude",
        POLITICAS_ANTIFRAUDE_PATH,
    )


@tool
def consultar_relatorios_fraude(pergunta: str) -> str:
    """
    Consulta o documento de relatorios historicos de fraude usando RAG.

    Use para buscar casos historicos, padroes detectados, tipos de fraude,
    resultados de investigacoes e licoes aprendidas.
    """

    return consultar_documento_rag(
        pergunta,
        relatorios_retriever,
        "relatorios historicos de fraude",
        RELATORIOS_FRAUDE_PATH,
        incluir_busca_textual_atualizada=True,
    )


@tool
def consultar_documentos_pdf(pergunta: str) -> str:
    """
    Consulta até 5 documentos PDF carregados na interface usando RAG.

    Use quando o usuário perguntar sobre documentos, PDFs, manuais, relatórios externos
    ou quando pedir para discutir respostas baseadas nos PDFs enviados.
    """

    return consultar_pdfs_rag(pergunta)


@tool
def consultar_banco_de_dados() -> str:
    """
    Consulta o banco de dados simulado, incluindo perfis de clientes, relatórios e confirmações pendentes.
    """

    texto = "Banco de dados simulado:\n\n"

    texto += "Perfis de contas da base:\n"
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

    texto += "\nHistórico atualizado por falsos positivos:\n"
    if banco_de_dados["historico_cliente"]:
        for i, evento in enumerate(banco_de_dados["historico_cliente"], start=1):
            texto += f"{i}. {evento}\n"
    else:
        texto += "Nenhum falso positivo confirmado.\n"

    texto += "\nBase de dados e modelos externos:\n"
    texto += f"Disponível: {'sim' if MODELOS_EXTERNOS_DISPONIVEIS else 'não'}\n"
    texto += f"Contas na base: {len(banco_de_dados['clientes_modelos'])}\n"
    texto += f"Transações consultáveis: {len(banco_de_dados['transacoes_modelos'])}\n"

    return texto


@tool
def consultar_base_de_modelos(pergunta: str) -> str:
    """
    Consulta a base de dados carregada de dados/bank_transactions_data_2.csv ou transacoes_clusterizadas.csv.

    Use para responder perguntas sobre AccountID, TransactionID, canais, locais,
    tipos de transação e disponibilidade dos modelos.
    """

    if not MODELOS_EXTERNOS_DISPONIVEIS:
        return f"Base/modelos indisponíveis: {ERRO_MODELOS_EXTERNOS or 'arquivos ou dependências não encontrados'}."
    return consultar_base_modelos(pergunta)


@tool
def verificar_transacao_modelo(
    account_id: str,
    transaction_amount: float,
    transaction_type: str,
    location: str,
    channel: str,
    customer_age: float,
    customer_occupation: str,
    transaction_duration: float,
    login_attempts: int,
    account_balance: float,
    device_id: str = "UNKNOWN_DEVICE",
    ip_address: str = "0.0.0.0",
    merchant_id: str = "UNKNOWN_MERCHANT",
) -> str:
    """
    Verifica uma nova transação usando a base em dados/ e os modelos em modelos/.

    Use quando a transação tiver campos do dataset bancário, como AccountID,
    TransactionAmount, TransactionType, Location, Channel e AccountBalance.
    """

    if not MODELOS_EXTERNOS_DISPONIVEIS:
        return f"Modelos indisponíveis: {ERRO_MODELOS_EXTERNOS or 'base/modelos não carregados'}."

    df_nova, resultado = analisar_nova_transacao(
        account_id=account_id,
        transaction_amount=transaction_amount,
        transaction_type=transaction_type,
        location=location,
        channel=channel,
        customer_age=customer_age,
        customer_occupation=customer_occupation,
        transaction_duration=transaction_duration,
        login_attempts=login_attempts,
        account_balance=account_balance,
        device_id=device_id,
        ip_address=ip_address,
        merchant_id=merchant_id,
    )
    return formatar_resultado_modelos(df_nova, resultado, origem="nova transação")


@tool
def verificar_transacao_modelo_por_id(transaction_id: str) -> str:
    """
    Verifica uma transação existente da base consultável pelo TransactionID usando os modelos.
    """

    if not MODELOS_EXTERNOS_DISPONIVEIS:
        return f"Modelos indisponíveis: {ERRO_MODELOS_EXTERNOS or 'base/modelos não carregados'}."

    df_nova, resultado = analisar_transacao_por_id(transaction_id)
    return formatar_resultado_modelos(df_nova, resultado, origem=f"TransactionID {transaction_id}")


tools = [
    verificar_transacao,
    gerar_relatorio_banco,
    solicitar_confirmacao_cliente,
    contestar_transacao_cliente,
    consultar_banco_de_dados,
    consultar_base_de_modelos,
    verificar_transacao_modelo,
    verificar_transacao_modelo_por_id,
    consultar_politicas_antifraude,
    consultar_relatorios_fraude,
    consultar_documentos_pdf,
]


# ============================================================
# Modelo Ollama
# ============================================================

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)

llm_com_tools = llm.bind_tools(tools)


# ============================================================
# Nó do agente
# ============================================================

def agente(state: MessagesState):
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
            "7. Em caso de suspeita, a IA deve retornar a classificação, a pontuação de risco e uma recomendação ao banco.\n"
            "8. O banco decide depois se autoriza, bloqueia ou pede confirmação ao cliente.\n"
            "9. Quando o cliente confirmar ou negar, a resposta deve voltar ao agente para contestação/reanálise.\n"
            "10. Após a reanálise, recalcule o risco, atualize histórico em falso positivo e gere relatório final.\n\n"

            "Ferramentas disponíveis:\n"
            "- verificar_transacao: use sempre que houver dados de uma compra/transação para análise.\n"
            "- gerar_relatorio_banco: use quando precisar registrar a justificativa para o banco.\n"
            "- solicitar_confirmacao_cliente: use somente quando o banco pedir explicitamente confirmação ao cliente. Informe também valor, cidade, categoria e horário.\n"
            "- contestar_transacao_cliente: use quando o cliente responder confirmação ou negação da transação.\n"
            "- consultar_banco_de_dados: use quando o usuário pedir para visualizar os dados internos.\n"
            "- consultar_base_de_modelos: use para consultar a base em dados/ e a disponibilidade dos modelos.\n"
            "- verificar_transacao_modelo: use para analisar novas transações com campos do dataset bancário e modelos treinados.\n"
            "- verificar_transacao_modelo_por_id: use para analisar uma transação existente pelo TransactionID.\n"
            "- consultar_politicas_antifraude: use para buscar politicas, limites, criterios e procedimentos antifraude.\n"
            "- consultar_relatorios_fraude: use para buscar casos historicos e padroes de fraude documentados.\n\n"
            "- consultar_documentos_pdf: use para consultar até 5 PDFs enviados na interface e responder com base nesses documentos.\n\n"

            "Ao analisar transacoes, use as politicas antifraude e os relatorios historicos quando eles ajudarem a justificar a decisao.\n"
            "Quando o usuário pedir respostas baseadas em documentos ou PDFs, consulte consultar_documentos_pdf antes de sintetizar.\n"
            "Quando o usuário informar campos como AccountID, TransactionID, Channel, CustomerOccupation, AccountBalance ou LoginAttempts, "
            "prefira as ferramentas de modelos/dados para usar a base carregada e os modelos treinados.\n"
            "Quando uma transação for suspeita, verifique a transação, gere o relatório para o banco e informe se é SUSPEITA ou NORMAL. "
            "Não envie solicitação ao cliente nessa etapa; aguarde uma decisão explícita do banco.\n"
            "Quando receber resposta do cliente, chame contestar_transacao_cliente e apresente o relatório final da reanálise.\n\n"

            "Responda em português brasileiro de forma clara, organizada e objetiva."
        )
    )

    resposta = llm_com_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [resposta]}


def retorno_cliente(state: MessagesState):
    return state


# ============================================================
# Construção do grafo
# ============================================================

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agente", agente)
graph_builder.add_node("ferramentas", ToolNode(tools))
graph_builder.add_node("retorno_cliente", retorno_cliente)

graph_builder.add_edge(START, "agente")

graph_builder.add_conditional_edges(
    "agente",
    tools_condition,
    {
        "tools": "ferramentas",
        "__end__": "__end__"
    }
)

graph_builder.add_edge("ferramentas", "retorno_cliente")
graph_builder.add_edge("retorno_cliente", "agente")

graph = graph_builder.compile()


# ============================================================
# Interface Streamlit
# ============================================================

st.set_page_config(
    page_title="Sentinela",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sentinela")
st.subheader("Sistema colaborativo de detecção de fraude bancária com papéis identificados")

st.markdown(
    """
    Este protótipo separa a colaboração entre **Banco**, **Conta/Cliente** e **Agente de IA**.
    O banco envia transações para análise, a conta acompanha confirmações pendentes e o agente usa modelos, RAG e histórico para apoiar a decisão.
    """
)


# ============================================================
# Barra lateral
# ============================================================

with st.sidebar:
    st.header("Visão geral")
    st.metric("Contas", len(banco_de_dados["clientes"]))
    st.metric("Relatórios", len(banco_de_dados["relatorios_fraude"]))
    st.metric("Contas na base de modelos", len(banco_de_dados["clientes_modelos"]))
    st.metric("Transações da base", len(banco_de_dados["transacoes_modelos"]))
    pendentes = sum(
        1
        for confirmacao in banco_de_dados["confirmacoes_cliente"]
        if confirmacao["status"] == "aguardando confirmação"
    )
    st.metric("Confirmações pendentes", pendentes)
    st.metric("Mensagens humanas", len(st.session_state.mensagens_participantes))

    st.header("RAG")
    st.write(f"Políticas: {'ativo' if politicas_retriever else 'busca textual'}")
    st.write(f"Relatórios: {'ativo' if relatorios_retriever else 'busca textual'}")

    st.header("Modelos")
    st.write(f"Base/modelos: {'ativos' if MODELOS_EXTERNOS_DISPONIVEIS else 'indisponíveis'}")
    st.write(f"Isolation Forest: {'ativo' if MODELO_ISOLATION is not None else 'indisponível'}")
    st.write(f"K-Means: {'ativo' if MODELO_CLUSTER is not None else 'indisponível'}")
    if ERRO_MODELOS_EXTERNOS:
        st.caption(ERRO_MODELOS_EXTERNOS)

    if st.button("Limpar conversa"):
        st.session_state.historico = []
        st.rerun()

    if st.button("Limpar dados gerados"):
        st.session_state.relatorios_fraude = []
        st.session_state.confirmacoes_cliente = []
        st.session_state.historico_cliente = []
        st.session_state.eventos_colaborativos = []
        st.session_state.mensagens_participantes = []
        st.session_state.ultima_analise_banco = None
        banco_de_dados["relatorios_fraude"] = st.session_state.relatorios_fraude
        banco_de_dados["confirmacoes_cliente"] = st.session_state.confirmacoes_cliente
        banco_de_dados["historico_cliente"] = st.session_state.historico_cliente
        st.rerun()


aba_banco, aba_cliente, aba_modelos, aba_agente = st.tabs(["Banco", "Cliente", "Modelos e Dados", "Agente"])


# ============================================================
# Papel: Banco
# ============================================================

with aba_banco:
    st.header("Banco")
    st.write("Envie uma transação para o agente avaliar e registre a decisão operacional.")

    banco_col1, banco_col2 = st.columns(2)

    with banco_col1:
        cliente = st.selectbox("Conta / AccountID", contas_disponiveis(), key="banco_cliente")
        perfil = perfil_conta(cliente)
        valor_padrao = float(perfil.get("valor_medio", 100.0) or 100.0)
        cidade_padrao = str(perfil.get("cidade_padrao", "desconhecido"))
        ocupacao_padrao = str(perfil.get("ocupacao_mais_comum", "desconhecido"))
        canal_padrao = str(perfil.get("canal_mais_comum", "Online"))

        valor = st.number_input("Valor da transação", min_value=0.0, value=valor_padrao, step=10.0, key="banco_valor")
        cidades = valores_unicos_base("Location", [cidade_padrao])
        cidade = st.text_input(
            "Localização",
            value=valor_texto_padrao("banco_cidade", cidade_padrao),
            key="banco_cidade",
            placeholder="Digite qualquer cidade",
        ).strip()
        if not cidade:
            cidade = cidade_padrao
        st.caption(f"Sugestões da base: {', '.join(cidades[:8])}. Você pode informar uma cidade nova.")
        saldo = st.number_input("Saldo após transação", min_value=0.0, value=max(valor_padrao * 10, 1000.0), step=100.0, key="banco_saldo")

    with banco_col2:
        categorias = valores_unicos_base("TransactionType", perfil.get("categorias_comuns", ["Debit", "Credit"]))
        categoria_padrao = perfil.get("categorias_comuns", [categorias[0]])[0] if categorias else "Debit"
        categoria_index = categorias.index(categoria_padrao) if categoria_padrao in categorias else 0
        categoria = st.selectbox("Tipo de transação", categorias, index=categoria_index, key="banco_categoria")
        canais = valores_unicos_base("Channel", [canal_padrao])
        canal_index = canais.index(canal_padrao) if canal_padrao in canais else 0
        canal = st.selectbox("Canal", canais, index=canal_index, key="banco_canal")
        horario = st.selectbox("Horário", ["diurno", "noturno"], key="banco_horario")

    banco_col3, banco_col4, banco_col5 = st.columns(3)
    with banco_col3:
        ocupacoes = valores_unicos_base("CustomerOccupation", [ocupacao_padrao])
        ocupacao_index = ocupacoes.index(ocupacao_padrao) if ocupacao_padrao in ocupacoes else 0
        ocupacao = st.selectbox("Ocupação", ocupacoes, index=ocupacao_index, key="banco_ocupacao")
        idade = st.number_input("Idade do cliente", min_value=0, max_value=120, value=40, step=1, key="banco_idade")
    with banco_col4:
        duracao = st.number_input("Duração da transação (s)", min_value=0.0, value=90.0, step=10.0, key="banco_duracao")
        tentativas_login = st.number_input("Tentativas de login", min_value=0, value=1, step=1, key="banco_login")
    with banco_col5:
        dispositivo = st.text_input("DeviceID", value="UNKNOWN_DEVICE", key="banco_device")
        ip = st.text_input("IP Address", value="0.0.0.0", key="banco_ip")
        comerciante = st.text_input("MerchantID", value="UNKNOWN_MERCHANT", key="banco_merchant")

    if perfil:
        st.caption(
            f"Perfil da conta: valor médio R$ {perfil.get('valor_medio', 0):.2f}; "
            f"local padrão {perfil.get('cidade_padrao', 'N/A')}; "
            f"canal comum {perfil.get('canal_mais_comum', 'N/A')}; "
            f"tipos comuns {', '.join(perfil.get('categorias_comuns', []))}."
        )

    if st.button("Enviar transação ao agente", key="banco_verificar"):
        analise_local = analisar_transacao_banco_com_modelos(
            cliente,
            valor,
            cidade,
            categoria,
            horario,
            canal,
            idade,
            ocupacao,
            duracao,
            tentativas_login,
            saldo,
            dispositivo,
            ip,
            comerciante,
        )
        registrar_relatorio_fraude({
            "tipo": "analise de transacao por modelos",
            "cliente": cliente,
            "valor": valor,
            "cidade": cidade,
            "categoria": categoria,
            "horario": horario,
            "acao_predita": analise_local["acao_predita"],
            "pontuacao_risco": analise_local["pontuacao_risco"],
            "classificacao": analise_local["status"],
            "motivos": analise_local["motivos"],
        })
        st.session_state.ultima_analise_banco = {
            "cliente": cliente,
            "valor": valor,
            "cidade": cidade,
            "categoria": categoria,
            "horario": horario,
            "canal": canal,
            "saldo": saldo,
            "ocupacao": ocupacao,
            "idade": idade,
            "duracao": duracao,
            "tentativas_login": tentativas_login,
            "cluster_predito": analise_local.get("resultado_modelo", {}).get("cluster_predito"),
            "perfil_cluster": analise_local.get("resultado_modelo", {}).get("perfil_cluster"),
            "status": analise_local["status"],
            "pontuacao_risco": analise_local["pontuacao_risco"],
            "acao_predita": analise_local["acao_predita"],
            "motivos": analise_local["motivos"],
            "status_fluxo": "aguardando decisão do banco",
            "decisao_operacional": None,
        }
        mensagem = (
            f"O banco deseja verificar uma transação da conta {cliente} "
            f"no valor de {valor} reais, na cidade de {cidade}, "
            f"tipo {categoria}, canal {canal}, no horário {horario}, "
            f"saldo posterior {saldo}, ocupação {ocupacao}, tentativas de login {tentativas_login}. "
            "Retorne objetivamente se a transação é SUSPEITA ou NORMAL, com pontuação de risco e justificativa. "
            "Não envie mensagem ao cliente ainda; o banco decidirá a ação operacional depois."
        )

        st.session_state.historico.append(HumanMessage(content=mensagem))

        with st.spinner("Agente analisando a transação..."):
            resultado = graph.invoke({"messages": st.session_state.historico})
            st.session_state.historico = resultado["messages"]

        st.success("Análise concluída.")
        st.rerun()

    st.divider()
    st.subheader("Decisão operacional do banco")

    ultima_analise = st.session_state.ultima_analise_banco
    if ultima_analise:
        st.write(
            f"IA classificou a transação como **{ultima_analise['status'].upper()}** "
            f"com risco **{ultima_analise['pontuacao_risco']}/100**."
        )
        st.write(f"Ação predita: **{ultima_analise['acao_predita']}**")
        st.write(f"Fluxo: **{ultima_analise.get('status_fluxo', 'aguardando decisão do banco')}**")
        if ultima_analise.get("decisao_operacional"):
            st.write(f"Decisão operacional: **{ultima_analise['decisao_operacional']}**")
        if ultima_analise.get("cluster_predito") is not None:
            st.write(f"Cluster predito: **{ultima_analise['cluster_predito']}**")
        if ultima_analise.get("perfil_cluster"):
            with st.expander("Perfil do cluster usado na análise"):
                st.json(ultima_analise["perfil_cluster"])
        for motivo in ultima_analise["motivos"]:
            st.write(f"- {motivo}")

        if ultima_analise.get("status_fluxo") == "aguardando decisão do banco":
            decisao_col1, decisao_col2, decisao_col3 = st.columns(3)
            with decisao_col1:
                if st.button("Autorizar compra", key="banco_autorizar_compra"):
                    ultima_analise["status_fluxo"] = "finalizada"
                    ultima_analise["decisao_operacional"] = "compra autorizada pelo banco"
                    registrar_mensagem(
                        "Banco",
                        ultima_analise["cliente"],
                        ultima_analise["cliente"],
                        (
                            f"Sua transação de R$ {ultima_analise['valor']:.2f} em "
                            f"{ultima_analise['cidade']} foi autorizada."
                        ),
                    )
                    registrar_evento("Banco", "Agente", "Banco autorizou a transação após análise da IA.")
                    st.success("Cliente informado sobre autorização.")
                    st.rerun()
            with decisao_col2:
                if st.button("Não autorizar", key="banco_negar_compra"):
                    ultima_analise["status_fluxo"] = "finalizada"
                    ultima_analise["decisao_operacional"] = "compra não autorizada pelo banco"
                    registrar_mensagem(
                        "Banco",
                        ultima_analise["cliente"],
                        ultima_analise["cliente"],
                        (
                            f"Sua transação de R$ {ultima_analise['valor']:.2f} em "
                            f"{ultima_analise['cidade']} não foi autorizada por segurança."
                        ),
                    )
                    registrar_evento("Banco", "Agente", "Banco não autorizou a transação após análise da IA.")
                    st.success("Cliente informado sobre não autorização.")
                    st.rerun()
            with decisao_col3:
                if st.button("Pedir confirmação", key="banco_pedir_confirmacao"):
                    registrar_confirmacao_pedida_pelo_banco(
                        ultima_analise["cliente"],
                        ultima_analise["valor"],
                        ultima_analise["cidade"],
                        ultima_analise["categoria"],
                        ultima_analise["horario"],
                    )
                    ultima_analise["status_fluxo"] = "aguardando resposta do cliente"
                    ultima_analise["decisao_operacional"] = "confirmação solicitada ao cliente"
                    st.success("Pedido de confirmação enviado ao cliente.")
                    st.rerun()
        elif ultima_analise.get("status_fluxo") == "aguardando resposta do cliente":
            st.info("Aguardando resposta do cliente. As ações do banco ficam bloqueadas para esta transação.")
        else:
            st.success("Fluxo desta transação finalizado. Envie uma nova transação para iniciar outra análise.")
    else:
        st.write("Nenhuma análise recente disponível. Envie uma transação ao agente primeiro.")

    st.divider()
    st.subheader("Discutir documentos com o agente")

    st.write("Consulte políticas e relatórios por RAG ou envie a pergunta ao agente para discutir os documentos.")
    banco_doc_col1, banco_doc_col2 = st.columns([2, 1])
    with banco_doc_col1:
        pergunta_banco_documentos = st.text_area(
            "Pergunta do banco sobre os documentos",
            value="Com base nas políticas e nos relatórios históricos, quais fatores justificam pedir confirmação ao cliente?",
            key="banco_pergunta_documentos",
            height=90,
        )
    with banco_doc_col2:
        fonte_documental_banco = st.radio(
            "Fonte",
            ["Políticas e relatórios", "Políticas", "Relatórios"],
            key="banco_fonte_documental",
        )

    if st.button("Consultar documentos RAG", key="banco_consultar_documentos_rag"):
        respostas_documentais = []
        with st.spinner("Consultando documentos..."):
            if fonte_documental_banco in ["Políticas e relatórios", "Políticas"]:
                respostas_documentais.append(
                    consultar_documento_rag(
                        pergunta_banco_documentos,
                        politicas_retriever,
                        "políticas antifraude",
                        POLITICAS_ANTIFRAUDE_PATH,
                    )
                )
            if fonte_documental_banco in ["Políticas e relatórios", "Relatórios"]:
                respostas_documentais.append(
                    consultar_documento_rag(
                        pergunta_banco_documentos,
                        relatorios_retriever,
                        "relatórios históricos de fraude",
                        RELATORIOS_FRAUDE_PATH,
                        incluir_busca_textual_atualizada=True,
                    )
                )
        st.session_state.resposta_banco_documentos = "\n\n".join(respostas_documentais)

    if st.button("Conversar com agente sobre documentos", key="banco_conversar_agente_documentos"):
        contexto_documental = []
        with st.spinner("Recuperando documentos e acionando o agente..."):
            if fonte_documental_banco in ["Políticas e relatórios", "Políticas"]:
                contexto_documental.append(
                    consultar_documento_rag(
                        pergunta_banco_documentos,
                        politicas_retriever,
                        "políticas antifraude",
                        POLITICAS_ANTIFRAUDE_PATH,
                    )
                )
            if fonte_documental_banco in ["Políticas e relatórios", "Relatórios"]:
                contexto_documental.append(
                    consultar_documento_rag(
                        pergunta_banco_documentos,
                        relatorios_retriever,
                        "relatórios históricos de fraude",
                        RELATORIOS_FRAUDE_PATH,
                        incluir_busca_textual_atualizada=True,
                    )
                )
            prompt_documental = (
                "O banco quer discutir os documentos antifraude com o agente.\n\n"
                f"Pergunta do banco: {pergunta_banco_documentos}\n\n"
                "Contexto RAG recuperado:\n"
                f"{chr(10).join(contexto_documental)}\n\n"
                "Responda como apoio ao banco: sintetize os trechos relevantes, explique a implicação operacional "
                "e indique uma recomendação objetiva."
            )
            st.session_state.historico.append(HumanMessage(content=prompt_documental))
            resultado = graph.invoke({"messages": st.session_state.historico})
            st.session_state.historico = resultado["messages"]
            registrar_evento("Banco", "Agente", f"Banco discutiu documentos RAG: {pergunta_banco_documentos}")
        st.rerun()

    if "resposta_banco_documentos" in st.session_state:
        st.info(st.session_state.resposta_banco_documentos)

    st.divider()
    st.subheader("Comunicação com o cliente")

    msg_cliente = st.selectbox(
        "Conta destinatária",
        contas_disponiveis(),
        key="banco_msg_cliente",
    )
    msg_banco = st.text_area(
        "Mensagem do banco ao cliente",
        value="Precisamos confirmar uma transação recente. Você reconhece essa compra?",
        key="banco_msg_texto",
    )

    ultima_analise_msg = st.session_state.ultima_analise_banco
    mensagem_transacional_bloqueada = bool(
        ultima_analise_msg
        and ultima_analise_msg.get("cliente") == msg_cliente
        and ultima_analise_msg.get("status_fluxo") in [
            "aguardando resposta do cliente",
            "finalizada",
        ]
    )

    if mensagem_transacional_bloqueada:
        st.info(
            "Existe uma decisão transacional em andamento ou finalizada para este cliente. "
            "Use uma nova análise para iniciar outro fluxo de mensagens."
        )

    if st.button("Enviar mensagem ao cliente", key="banco_enviar_msg", disabled=mensagem_transacional_bloqueada):
        registrar_mensagem("Banco", msg_cliente, msg_cliente, msg_banco)
        st.success("Mensagem enviada ao cliente.")
        st.rerun()

    mensagens_banco = [
        mensagem
        for mensagem in st.session_state.mensagens_participantes
        if mensagem["destino"] == "Banco" or mensagem["origem"] == "Banco"
    ]

    if mensagens_banco:
        st.write("Mensagens mediadas:")
        for mensagem in mensagens_banco:
            st.write(f"**{mensagem['origem']} → {mensagem['destino']}** ({mensagem['cliente']}): {mensagem['mensagem']}")
    else:
        st.write("Nenhuma mensagem mediada registrada para o banco.")

    st.divider()
    st.subheader("Relatórios para o banco")

    if banco_de_dados["relatorios_fraude"]:
        for indice, relatorio in enumerate(banco_de_dados["relatorios_fraude"], start=1):
            cliente_relatorio = relatorio.get("cliente", relatorio.get("account_id", "N/A"))
            classificacao_relatorio = str(relatorio.get("classificacao", "indisponivel")).upper()
            with st.expander(f"Relatório {indice} - {cliente_relatorio} - {classificacao_relatorio}"):
                st.json(relatorio)
    else:
        st.write("Nenhum relatório registrado ainda.")

    st.subheader("Amostra de perfis da base")
    perfis_amostra = list(banco_de_dados["clientes"].items())[:10]
    if perfis_amostra:
        for nome_cliente, perfil in perfis_amostra:
            st.write(
                f"**{nome_cliente}**: local padrão {perfil.get('cidade_padrao', 'N/A')}, "
                f"valor médio R$ {perfil.get('valor_medio', 0):.2f}, "
                f"tipos {', '.join(perfil.get('categorias_comuns', []))}, "
                f"canal {perfil.get('canal_mais_comum', 'N/A')}, "
                f"transações {perfil.get('quantidade_transacoes', 0)}."
            )
    else:
        st.write("Nenhum perfil de conta carregado da base.")


# ============================================================
# Papel: Cliente
# ============================================================

with aba_cliente:
    st.header("Cliente")
    st.write("Acompanhe solicitações de confirmação enviadas pelo banco e registre a resposta do cliente.")

    cliente_visualizacao = st.selectbox(
        "Visualizar como conta",
        contas_disponiveis(),
        key="cliente_visualizacao",
    )

    perfil_visualizacao = perfil_conta(cliente_visualizacao)
    if perfil_visualizacao:
        perfil_col1, perfil_col2, perfil_col3, perfil_col4 = st.columns(4)
        perfil_col1.metric("Valor médio", f"R$ {perfil_visualizacao.get('valor_medio', 0):.2f}")
        perfil_col2.metric("Transações", perfil_visualizacao.get("quantidade_transacoes", 0))
        perfil_col3.metric("Local padrão", perfil_visualizacao.get("cidade_padrao", "N/A"))
        perfil_col4.metric("Canal comum", perfil_visualizacao.get("canal_mais_comum", "N/A"))
        st.caption(
            f"Tipos comuns: {', '.join(perfil_visualizacao.get('categorias_comuns', []))}. "
            f"Ocupação comum: {perfil_visualizacao.get('ocupacao_mais_comum', 'N/A')}."
        )

    confirmacoes_cliente = [
        (indice, confirmacao)
        for indice, confirmacao in enumerate(banco_de_dados["confirmacoes_cliente"])
        if confirmacao["cliente"] == cliente_visualizacao
    ]

    mensagens_cliente = [
        mensagem
        for mensagem in st.session_state.mensagens_participantes
        if mensagem["destino"] == cliente_visualizacao or mensagem["origem"] == cliente_visualizacao
    ]

    st.subheader("Mensagens mediadas pelo sistema")
    if mensagens_cliente:
        for mensagem in mensagens_cliente:
            st.write(f"**{mensagem['origem']} → {mensagem['destino']}**: {mensagem['mensagem']}")
    else:
        st.write("Nenhuma mensagem mediada para este cliente.")

    st.divider()

    if confirmacoes_cliente:
        for indice, confirmacao in confirmacoes_cliente:
            st.subheader(f"Solicitação {indice + 1}")
            st.write(confirmacao["descricao_transacao"])
            st.write(f"Status: **{confirmacao['status']}**")
            if confirmacao.get("resposta_cliente"):
                st.write(f"Resposta registrada: **{confirmacao['resposta_cliente']}**")

            if confirmacao["status"] == "aguardando confirmação":
                cliente_col1, cliente_col2 = st.columns(2)
                with cliente_col1:
                    if st.button("Confirmar compra", key=f"confirmar_{indice}"):
                        processar_resposta_cliente(indice, "confirmada pelo cliente")
                        st.success("Resposta registrada. A transação foi reanalisada e o fluxo foi finalizado.")
                        st.rerun()
                with cliente_col2:
                    if st.button("Não reconheço", key=f"negar_{indice}"):
                        processar_resposta_cliente(indice, "negada pelo cliente")
                        st.success("Resposta registrada. A transação foi reanalisada e o fluxo foi finalizado.")
                        st.rerun()
            else:
                st.info("Esta solicitação já foi respondida e está fechada.")
            if "reanalise" in confirmacao:
                st.write("Reanálise:")
                st.json(confirmacao["reanalise"])
            st.divider()
    else:
        st.write("Nenhuma confirmação enviada para este cliente.")

    st.subheader("Histórico geral de confirmações")
    if banco_de_dados["confirmacoes_cliente"]:
        st.json(banco_de_dados["confirmacoes_cliente"])
    else:
        st.write("Nenhuma confirmação registrada ainda.")

    st.subheader("Histórico atualizado por falsos positivos")
    if banco_de_dados["historico_cliente"]:
        st.json(banco_de_dados["historico_cliente"])
    else:
        st.write("Nenhum falso positivo confirmado ainda.")


# ============================================================
# Papel: Modelos e Dados
# ============================================================

with aba_modelos:
    st.header("Modelos e Dados")
    st.write("Consulte a base em `dados/` e analise transações usando os modelos em `modelos/`.")

    if not MODELOS_EXTERNOS_DISPONIVEIS:
        st.warning(f"Base/modelos indisponíveis: {ERRO_MODELOS_EXTERNOS or 'verifique dados/ e modelos/.'}")
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Transações carregadas", 0 if DF_TRANSACOES is None else len(DF_TRANSACOES))
        col_m2.metric("Contas com perfil", len(PERFIS_CONTAS))
        col_m3.metric("Transações consultáveis", len(TRANSACOES_CONSULTA))

        st.divider()
        st.subheader("Análise tabular da base")

        tabela_aba1, tabela_aba2, tabela_aba3, tabela_aba4 = st.tabs([
            "Resumo",
            "Clusters",
            "Segmentos",
            "Transações",
        ])

        with tabela_aba1:
            if DF_TRANSACOES is not None:
                resumo_geral = {
                    "métrica": [
                        "transações",
                        "contas únicas",
                        "valor médio",
                        "valor máximo",
                        "saldo médio",
                        "tentativas médias de login",
                        "duração média",
                    ],
                    "valor": [
                        len(DF_TRANSACOES),
                        DF_TRANSACOES["AccountID"].nunique(),
                        round(float(DF_TRANSACOES["TransactionAmount"].mean()), 2),
                        round(float(DF_TRANSACOES["TransactionAmount"].max()), 2),
                        round(float(DF_TRANSACOES["AccountBalance"].mean()), 2),
                        round(float(DF_TRANSACOES["LoginAttempts"].mean()), 2),
                        round(float(DF_TRANSACOES["TransactionDuration"].mean()), 2),
                    ],
                }
                st.dataframe(resumo_geral, use_container_width=True, hide_index=True)

        with tabela_aba2:
            if PERFIS_CLUSTERS is not None and not getattr(PERFIS_CLUSTERS, "empty", True):
                st.dataframe(PERFIS_CLUSTERS, use_container_width=True, height=320)
            elif DF_TRANSACOES is not None and "cluster" in DF_TRANSACOES.columns:
                resumo_cluster = (
                    DF_TRANSACOES.groupby("cluster")
                    .agg(
                        quantidade=("TransactionID", "count"),
                        valor_medio=("TransactionAmount", "mean"),
                        valor_maximo=("TransactionAmount", "max"),
                        saldo_medio=("AccountBalance", "mean"),
                        tentativas_login_media=("LoginAttempts", "mean"),
                    )
                    .reset_index()
                )
                st.dataframe(resumo_cluster, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum perfil de cluster encontrado.")

        with tabela_aba3:
            if DF_TRANSACOES is not None:
                seg_col1, seg_col2 = st.columns(2)
                with seg_col1:
                    por_canal = (
                        DF_TRANSACOES.groupby("Channel")
                        .agg(
                            quantidade=("TransactionID", "count"),
                            valor_medio=("TransactionAmount", "mean"),
                            tentativas_login_media=("LoginAttempts", "mean"),
                        )
                        .sort_values("quantidade", ascending=False)
                        .reset_index()
                    )
                    st.write("Por canal")
                    st.dataframe(por_canal, use_container_width=True, hide_index=True)
                with seg_col2:
                    por_tipo = (
                        DF_TRANSACOES.groupby("TransactionType")
                        .agg(
                            quantidade=("TransactionID", "count"),
                            valor_medio=("TransactionAmount", "mean"),
                            valor_maximo=("TransactionAmount", "max"),
                        )
                        .sort_values("quantidade", ascending=False)
                        .reset_index()
                    )
                    st.write("Por tipo de transação")
                    st.dataframe(por_tipo, use_container_width=True, hide_index=True)

                por_local = (
                    DF_TRANSACOES.groupby("Location")
                    .agg(
                        quantidade=("TransactionID", "count"),
                        valor_medio=("TransactionAmount", "mean"),
                        tentativas_login_media=("LoginAttempts", "mean"),
                    )
                    .sort_values("quantidade", ascending=False)
                    .head(25)
                    .reset_index()
                )
                st.write("Top locais")
                st.dataframe(por_local, use_container_width=True, hide_index=True)

        with tabela_aba4:
            if DF_TRANSACOES is not None:
                filtro_col1, filtro_col2, filtro_col3 = st.columns(3)
                with filtro_col1:
                    filtro_conta = st.selectbox("Conta", ["Todas"] + valores_unicos_base("AccountID", []), key="filtro_modelos_conta")
                with filtro_col2:
                    filtro_canal = st.selectbox("Canal", ["Todos"] + valores_unicos_base("Channel", []), key="filtro_modelos_canal")
                with filtro_col3:
                    filtro_tipo = st.selectbox("Tipo", ["Todos"] + valores_unicos_base("TransactionType", []), key="filtro_modelos_tipo")

                df_tabela = DF_TRANSACOES.copy()
                if filtro_conta != "Todas":
                    df_tabela = df_tabela[df_tabela["AccountID"].astype(str) == filtro_conta]
                if filtro_canal != "Todos":
                    df_tabela = df_tabela[df_tabela["Channel"].astype(str) == filtro_canal]
                if filtro_tipo != "Todos":
                    df_tabela = df_tabela[df_tabela["TransactionType"].astype(str) == filtro_tipo]

                colunas_tabela = [
                    coluna for coluna in [
                        "TransactionID",
                        "AccountID",
                        "TransactionAmount",
                        "TransactionType",
                        "Location",
                        "Channel",
                        "CustomerAge",
                        "CustomerOccupation",
                        "TransactionDuration",
                        "LoginAttempts",
                        "AccountBalance",
                        "cluster",
                        "anomalia_isolation_forest",
                    ]
                    if coluna in df_tabela.columns
                ]
                st.write(f"Registros filtrados: {len(df_tabela)}")
                st.dataframe(df_tabela[colunas_tabela].head(500), use_container_width=True, height=360)

        st.divider()
        st.subheader("Consulta da base")

        pergunta_base = st.text_input(
            "Buscar na base",
            value="transações online suspeitas",
            key="consulta_base_modelos",
        )
        if st.button("Consultar base de modelos", key="btn_consultar_base_modelos"):
            st.session_state.resposta_base_modelos = consultar_base_modelos(pergunta_base)

        if "resposta_base_modelos" in st.session_state:
            st.info(st.session_state.resposta_base_modelos)

        st.divider()
        st.subheader("Analisar transação existente")

        if DF_TRANSACOES is not None and "TransactionID" in DF_TRANSACOES.columns:
            ids_transacoes = DF_TRANSACOES["TransactionID"].astype(str).head(1000).tolist()
            transaction_id_modelo = st.selectbox(
                "TransactionID",
                ids_transacoes,
                key="modelo_transaction_id",
            )

            if st.button("Analisar TransactionID", key="btn_analisar_transaction_id"):
                df_nova, resultado_modelo = analisar_transacao_por_id(transaction_id_modelo)
                st.session_state.ultimo_resultado_modelos = formatar_resultado_modelos(
                    df_nova,
                    resultado_modelo,
                    origem=f"TransactionID {transaction_id_modelo}",
                )
                if df_nova is not None:
                    linha = df_nova.iloc[0]
                    registrar_relatorio_fraude({
                        "tipo": "analise por modelo",
                        "cliente": str(linha["AccountID"]),
                        "valor": float(linha["TransactionAmount"]),
                        "cidade": str(linha["Location"]),
                        "categoria": str(linha["TransactionType"]),
                        "horario": "diurno" if 6 <= int(linha["TransactionHour"]) < 18 else "noturno",
                        "acao_predita": resultado_modelo.get("acao_predita"),
                        "pontuacao_risco": resultado_modelo.get("pontuacao_risco", 0),
                        "classificacao": resultado_modelo.get("status"),
                        "motivos": resultado_modelo.get("motivos", []),
                    })
                st.rerun()

        st.divider()
        st.subheader("Analisar nova transação pelos modelos")

        contas_modelo = list(PERFIS_CONTAS.keys())[:1000] or ["UNKNOWN_ACCOUNT"]
        tipos_modelo = sorted(DF_TRANSACOES["TransactionType"].dropna().astype(str).unique()) if DF_TRANSACOES is not None else ["Debit", "Credit"]
        locais_modelo = sorted(DF_TRANSACOES["Location"].dropna().astype(str).unique()) if DF_TRANSACOES is not None else ["Unknown"]
        canais_modelo = sorted(DF_TRANSACOES["Channel"].dropna().astype(str).unique()) if DF_TRANSACOES is not None else ["Online", "ATM", "Branch"]
        ocupacoes_modelo = sorted(DF_TRANSACOES["CustomerOccupation"].dropna().astype(str).unique()) if DF_TRANSACOES is not None else ["Unknown"]

        with st.form("form_modelos_nova_transacao"):
            form_col1, form_col2, form_col3 = st.columns(3)

            with form_col1:
                account_id_modelo = st.selectbox("AccountID", contas_modelo)
                amount_modelo = st.number_input("TransactionAmount", min_value=0.0, value=100.0, step=50.0)
                type_modelo = st.selectbox("TransactionType", tipos_modelo)

            with form_col2:
                location_modelo = st.text_input(
                    "Location",
                    value=locais_modelo[0],
                    placeholder="Digite qualquer cidade/local",
                ).strip()
                if not location_modelo:
                    location_modelo = locais_modelo[0]
                st.caption(f"Sugestões: {', '.join(locais_modelo[:6])}.")
                channel_modelo = st.selectbox("Channel", canais_modelo)
                balance_modelo = st.number_input("AccountBalance", min_value=0.0, value=1000.0, step=100.0)

            with form_col3:
                age_modelo = st.number_input("CustomerAge", min_value=0, max_value=120, value=30, step=1)
                occupation_modelo = st.selectbox("CustomerOccupation", ocupacoes_modelo)
                duration_modelo = st.number_input("TransactionDuration", min_value=0.0, value=60.0, step=10.0)
                login_modelo = st.number_input("LoginAttempts", min_value=0, value=1, step=1)

            device_modelo = st.text_input("DeviceID", value="UNKNOWN_DEVICE")
            ip_modelo = st.text_input("IP Address", value="0.0.0.0")
            merchant_modelo = st.text_input("MerchantID", value="UNKNOWN_MERCHANT")

            enviar_modelo = st.form_submit_button("Analisar nova transação")

        if enviar_modelo:
            df_nova, resultado_modelo = analisar_nova_transacao(
                account_id=account_id_modelo,
                transaction_amount=amount_modelo,
                transaction_type=type_modelo,
                location=location_modelo,
                channel=channel_modelo,
                customer_age=age_modelo,
                customer_occupation=occupation_modelo,
                transaction_duration=duration_modelo,
                login_attempts=login_modelo,
                account_balance=balance_modelo,
                device_id=device_modelo,
                ip_address=ip_modelo,
                merchant_id=merchant_modelo,
            )
            st.session_state.ultimo_resultado_modelos = formatar_resultado_modelos(
                df_nova,
                resultado_modelo,
                origem="nova transação",
            )
            registrar_relatorio_fraude({
                "tipo": "analise por modelo",
                "cliente": account_id_modelo,
                "valor": amount_modelo,
                "cidade": location_modelo,
                "categoria": type_modelo,
                "horario": "modelo",
                "acao_predita": resultado_modelo.get("acao_predita"),
                "pontuacao_risco": resultado_modelo.get("pontuacao_risco", 0),
                "classificacao": resultado_modelo.get("status"),
                "motivos": resultado_modelo.get("motivos", []),
            })
            st.rerun()

        if st.session_state.ultimo_resultado_modelos:
            st.divider()
            st.subheader("Resultado dos modelos")
            st.info(st.session_state.ultimo_resultado_modelos)

        st.divider()
        st.subheader("Amostra da base")
        if DF_TRANSACOES is not None:
            colunas_modelos = [
                coluna for coluna in [
                    "TransactionID",
                    "AccountID",
                    "TransactionAmount",
                    "TransactionType",
                    "Location",
                    "Channel",
                    "CustomerAge",
                    "CustomerOccupation",
                    "TransactionDuration",
                    "LoginAttempts",
                    "AccountBalance",
                    "cluster",
                    "anomalia_isolation_forest",
                ]
                if coluna in DF_TRANSACOES.columns
            ]
            st.dataframe(DF_TRANSACOES[colunas_modelos].head(200), use_container_width=True, height=360)


# ============================================================
# Papel: Agente
# ============================================================

with aba_agente:
    st.header("Agente de IA")
    st.write("Consulte a base RAG e converse com o agente sobre transações, políticas e padrões históricos.")

    st.subheader("Mediação colaborativa")

    if st.session_state.eventos_colaborativos:
        for indice, evento in enumerate(reversed(st.session_state.eventos_colaborativos[-8:]), start=1):
            st.write(f"{indice}. **{evento['origem']} → {evento['destino']}**: {evento['descricao']}")
    else:
        st.write("Nenhum evento colaborativo registrado ainda.")

    if st.button("Pedir resumo de mediação ao agente", key="resumo_mediacao"):
        eventos = "\n".join(
            f"- {evento['origem']} -> {evento['destino']}: {evento['descricao']}"
            for evento in st.session_state.eventos_colaborativos
        ) or "Nenhum evento registrado."
        mensagens = "\n".join(
            f"- {mensagem['origem']} -> {mensagem['destino']} ({mensagem['cliente']}): {mensagem['mensagem']}"
            for mensagem in st.session_state.mensagens_participantes
        ) or "Nenhuma mensagem humana registrada."
        prompt = (
            "Atue como mediador do sistema colaborativo. Resuma o estado da interação entre Banco e Cliente, "
            "aponte pendências e sugira o próximo passo operacional.\n\n"
            f"Eventos:\n{eventos}\n\nMensagens:\n{mensagens}"
        )
        st.session_state.historico.append(HumanMessage(content=prompt))
        with st.spinner("Agente mediando a interação..."):
            resultado = graph.invoke({"messages": st.session_state.historico})
            st.session_state.historico = resultado["messages"]
        st.rerun()

    st.divider()

    st.subheader("Base de conhecimento RAG")
    st.write("Use os documentos internos ou envie até 5 PDFs para consulta e discussão com o agente.")

    pdfs_enviados = st.file_uploader(
        "Documentos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdfs_rag_upload",
    )
    if pdfs_enviados and len(pdfs_enviados) > MAX_PDF_DOCUMENTS:
        st.warning(f"Somente os primeiros {MAX_PDF_DOCUMENTS} PDFs serão usados.")

    if st.button("Salvar PDFs para RAG", key="salvar_pdfs_rag"):
        caminhos_pdf = salvar_pdfs_enviados(pdfs_enviados)
        st.session_state.pdfs_rag_paths = caminhos_pdf
        build_pdf_retriever.clear()
        st.success(f"{len(caminhos_pdf)} PDF(s) disponível(is) para consulta.")
        st.rerun()

    caminhos_pdf_atuais = st.session_state.get("pdfs_rag_paths") or listar_pdfs_disponiveis()
    if caminhos_pdf_atuais:
        st.caption("PDFs disponíveis: " + ", ".join(os.path.basename(caminho) for caminho in caminhos_pdf_atuais))

    pergunta_pdfs = st.text_input(
        "Pergunta sobre PDFs",
        value="Quais pontos dos PDFs ajudam na decisão do banco?",
        key="pergunta_pdfs",
    )
    pdf_col1, pdf_col2 = st.columns(2)
    with pdf_col1:
        if st.button("Consultar PDFs", key="consultar_pdfs"):
            with st.spinner("Consultando PDFs..."):
                st.session_state.resposta_pdfs = consultar_pdfs_rag(pergunta_pdfs, caminhos_pdf_atuais)
    with pdf_col2:
        if st.button("Discutir PDFs com agente", key="discutir_pdfs_agente"):
            contexto_pdf = consultar_pdfs_rag(pergunta_pdfs, caminhos_pdf_atuais)
            prompt_pdf = (
                "Discuta os PDFs carregados com base no contexto RAG abaixo.\n\n"
                f"Pergunta: {pergunta_pdfs}\n\n"
                f"Contexto:\n{contexto_pdf}\n\n"
                "Gere uma resposta objetiva para apoiar a decisão colaborativa do banco."
            )
            st.session_state.historico.append(HumanMessage(content=prompt_pdf))
            with st.spinner("Agente discutindo PDFs..."):
                resultado = graph.invoke({"messages": st.session_state.historico})
                st.session_state.historico = resultado["messages"]
            registrar_evento("Banco", "Agente", f"Banco discutiu PDFs: {pergunta_pdfs}")
            st.rerun()

    if "resposta_pdfs" in st.session_state:
        st.info(st.session_state.resposta_pdfs)

    st.divider()
    rag_col1, rag_col2 = st.columns(2)

    with rag_col1:
        pergunta_politicas = st.text_input(
            "Pergunta sobre políticas",
            value="Quais indicadores aumentam o risco de fraude?",
            key="pergunta_politicas",
        )

        if st.button("Consultar políticas", key="consultar_politicas"):
            with st.spinner("Consultando políticas antifraude..."):
                st.session_state.resposta_politicas = consultar_documento_rag(
                    pergunta_politicas,
                    politicas_retriever,
                    "políticas antifraude",
                    POLITICAS_ANTIFRAUDE_PATH,
                )

        if "resposta_politicas" in st.session_state:
            st.info(st.session_state.resposta_politicas)

    with rag_col2:
        pergunta_relatorios = st.text_input(
            "Pergunta sobre relatórios",
            value="Quais fraudes envolveram mudança de localização?",
            key="pergunta_relatorios",
        )

        if st.button("Consultar relatórios", key="consultar_relatorios"):
            with st.spinner("Consultando relatórios de fraude..."):
                st.session_state.resposta_relatorios = consultar_documento_rag(
                    pergunta_relatorios,
                    relatorios_retriever,
                    "relatórios históricos de fraude",
                    RELATORIOS_FRAUDE_PATH,
                    incluir_busca_textual_atualizada=True,
                )

        if "resposta_relatorios" in st.session_state:
            st.info(st.session_state.resposta_relatorios)

    st.divider()
    st.subheader("Chat com o agente")

    for mensagem in st.session_state.historico:
        tipo = mensagem.type

        if tipo == "human":
            with st.chat_message("user"):
                st.write(mensagem.content)

        elif tipo == "ai":
            if mensagem.content:
                with st.chat_message("assistant"):
                    st.write(mensagem.content)

        elif tipo == "tool":
            with st.chat_message("assistant"):
                st.info(mensagem.content)

    entrada = st.chat_input("Digite uma mensagem para o agente...")

    if entrada:
        st.session_state.historico.append(HumanMessage(content=entrada))

        with st.spinner("Agente pensando..."):
            resultado = graph.invoke({"messages": st.session_state.historico})
            st.session_state.historico = resultado["messages"]

        st.rerun()
