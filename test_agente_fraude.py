import copy
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

MODULE_PATH = Path(__file__).resolve().parent / "app.py"
MODULE_NAME = "agente_fraude_module"


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return getattr(sys.modules["streamlit"], name)


def _prepare_dummy_imports():
    streamlit_mod = types.ModuleType("streamlit")
    streamlit_mod.session_state = SessionState()
    streamlit_mod.sidebar = DummyContext()

    def cache_resource(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def no_op(*args, **kwargs):
        return None

    def button(*args, **kwargs):
        return False

    def selectbox(label, options, index=0, *args, **kwargs):
        return list(options)[index] if options else None

    def radio(label, options, *args, **kwargs):
        return list(options)[0] if options else None

    def text_input(label, value="", *args, **kwargs):
        return value

    def text_area(label, value="", *args, **kwargs):
        return value

    def number_input(label, value=0, *args, **kwargs):
        return value

    def columns(spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [DummyContext() for _ in range(count)]

    def tabs(labels):
        return [DummyContext() for _ in labels]

    def chat_input(*args, **kwargs):
        return None

    for name in [
        "set_page_config",
        "title",
        "subheader",
        "markdown",
        "header",
        "metric",
        "write",
        "caption",
        "divider",
        "success",
        "info",
        "warning",
        "json",
        "dataframe",
        "rerun",
    ]:
        setattr(streamlit_mod, name, no_op)

    streamlit_mod.cache_resource = cache_resource
    streamlit_mod.button = button
    streamlit_mod.selectbox = selectbox
    streamlit_mod.radio = radio
    streamlit_mod.text_input = text_input
    streamlit_mod.text_area = text_area
    streamlit_mod.number_input = number_input
    streamlit_mod.columns = columns
    streamlit_mod.tabs = tabs
    streamlit_mod.chat_input = chat_input
    streamlit_mod.file_uploader = lambda *args, **kwargs: []
    streamlit_mod.spinner = lambda *args, **kwargs: DummyContext()
    streamlit_mod.expander = lambda *args, **kwargs: DummyContext()
    streamlit_mod.form = lambda *args, **kwargs: DummyContext()
    streamlit_mod.chat_message = lambda *args, **kwargs: DummyContext()
    streamlit_mod.form_submit_button = button
    sys.modules["streamlit"] = streamlit_mod

    langchain_core = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = langchain_core

    messages_mod = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content):
            self.content = content

    class HumanMessage(BaseMessage):
        type = "human"

    class SystemMessage(BaseMessage):
        type = "system"

    messages_mod.HumanMessage = HumanMessage
    messages_mod.SystemMessage = SystemMessage
    sys.modules["langchain_core.messages"] = messages_mod

    tools_mod = types.ModuleType("langchain_core.tools")
    tools_mod.tool = lambda func: func
    sys.modules["langchain_core.tools"] = tools_mod

    documents_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_mod.Document = Document
    sys.modules["langchain_core.documents"] = documents_mod

    ollama_mod = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return HumanMessage(content="mock response")

    class OllamaEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

    ollama_mod.ChatOllama = ChatOllama
    ollama_mod.OllamaEmbeddings = OllamaEmbeddings
    sys.modules["langchain_ollama"] = ollama_mod

    graph_pkg = types.ModuleType("langgraph")
    sys.modules["langgraph"] = graph_pkg

    graph_mod = types.ModuleType("langgraph.graph")

    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def compile(self):
            return self

        def invoke(self, state):
            return {"messages": state["messages"]}

    graph_mod.StateGraph = StateGraph
    graph_mod.START = object()
    graph_mod.MessagesState = dict
    sys.modules["langgraph.graph"] = graph_mod

    prebuilt_mod = types.ModuleType("langgraph.prebuilt")

    class ToolNode:
        def __init__(self, tools):
            self.tools = tools

    prebuilt_mod.ToolNode = ToolNode
    prebuilt_mod.tools_condition = lambda *args, **kwargs: True
    sys.modules["langgraph.prebuilt"] = prebuilt_mod

    splitters_mod = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):
            return documents

    splitters_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = splitters_mod

    chroma_mod = types.ModuleType("langchain_chroma")

    class Chroma:
        @classmethod
        def from_documents(cls, *args, **kwargs):
            raise RuntimeError("vector store desabilitado no teste")

    chroma_mod.Chroma = Chroma
    sys.modules["langchain_chroma"] = chroma_mod


_prepare_dummy_imports()
_spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
_agente_fraude_module = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = _agente_fraude_module
_spec.loader.exec_module(_agente_fraude_module)


@pytest.fixture(scope="session")
def agente_fraude():
    return _agente_fraude_module


@pytest.fixture
def banco_de_dados_limpo():
    return copy.deepcopy({
        "clientes": {
            "Joao": {
                "cidade_padrao": "Sao Carlos",
                "valor_medio": 120.0,
                "categorias_comuns": ["mercado", "farmacia", "restaurante"],
                "horario_comum": "diurno",
            },
            "Ana": {
                "cidade_padrao": "Campinas",
                "valor_medio": 250.0,
                "categorias_comuns": ["roupas", "mercado", "transporte"],
                "horario_comum": "diurno",
            },
        },
        "relatorios_fraude": [],
        "confirmacoes_cliente": [],
        "historico_cliente": [],
        "clientes_modelos": {},
        "transacoes_modelos": [],
    })


@pytest.mark.parametrize(
    "cliente, valor, cidade, categoria, horario, status, risco, acao",
    [
        ("Joao", 100.0, "Sao Carlos", "mercado", "diurno", "normal", 0, "compra legítima"),
        ("Joao", 100.0, "Campinas", "viagem", "diurno", "suspeita", 50, "fraude"),
        ("Marcos", 1000.0, "Sao Paulo", "mercado", "diurno", "suspeita", 80, "fraude"),
    ],
)
def test_classificar_transacao_cenarios_atuais(
    agente_fraude, banco_de_dados_limpo, cliente, valor, cidade, categoria, horario, status, risco, acao
):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo):
        resultado = agente_fraude.classificar_transacao(cliente, valor, cidade, categoria, horario)

    assert resultado["status"] == status
    assert resultado["pontuacao_risco"] == risco
    assert resultado["acao_predita"] == acao


def test_gerar_relatorio_banco_registra_em_memoria_sem_duplicar(agente_fraude, banco_de_dados_limpo):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo), patch.object(
        agente_fraude, "persistir_relatorio_fraude", return_value="rel-1"
    ):
        texto = agente_fraude.gerar_relatorio_banco("Joao", 100.0, "Campinas", "viagem", "diurno")
        agente_fraude.gerar_relatorio_banco("Joao", 100.0, "Campinas", "viagem", "diurno")

    assert "Relatório enviado ao banco" in texto
    assert len(banco_de_dados_limpo["relatorios_fraude"]) == 1
    assert banco_de_dados_limpo["relatorios_fraude"][0]["id_relatorio"] == "rel-1"
    assert banco_de_dados_limpo["relatorios_fraude"][0]["classificacao"] == "suspeita"


def test_solicitar_confirmacao_cliente_guarda_dados_estruturados(agente_fraude, banco_de_dados_limpo):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo), patch.object(
        agente_fraude, "registrar_evento"
    ):
        texto = agente_fraude.solicitar_confirmacao_cliente(
            cliente="Joao",
            descricao_transacao="Compra em cidade nova",
            valor=300.0,
            cidade="Curitiba",
            categoria="viagem",
            horario="noturno",
        )

    assert "Confirmação enviada ao cliente Joao" in texto
    confirmacao = banco_de_dados_limpo["confirmacoes_cliente"][0]
    assert confirmacao["status"] == "aguardando confirmação"
    assert confirmacao["transacao"]["cidade"] == "Curitiba"


def test_recalcular_risco_cliente_confirma_reduz_risco_e_atualiza_historico(agente_fraude, banco_de_dados_limpo):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo):
        reanalise = agente_fraude.recalcular_risco_apos_resposta(
            "Joao",
            100.0,
            "Campinas",
            "viagem",
            "diurno",
            "confirmada pelo cliente",
        )

    assert reanalise["pontuacao_risco_inicial"] == 50
    assert reanalise["pontuacao_risco_final"] == 5
    assert reanalise["classificacao_final"] == "normal"
    assert banco_de_dados_limpo["historico_cliente"][0]["cidade"] == "Campinas"


def test_recalcular_risco_cliente_nega_confirma_fraude(agente_fraude, banco_de_dados_limpo):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo):
        reanalise = agente_fraude.recalcular_risco_apos_resposta(
            "Joao",
            100.0,
            "Campinas",
            "viagem",
            "diurno",
            "negada pelo cliente",
        )

    assert reanalise["pontuacao_risco_final"] == 85
    assert reanalise["classificacao_final"] == "fraude confirmada"
    assert reanalise["acao_final"] == "bloquear transação e abrir contestação"


def test_registrar_confirmacao_pedida_pelo_banco_eh_idempotente(agente_fraude, banco_de_dados_limpo):
    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo), patch.object(
        agente_fraude, "registrar_mensagem"
    ), patch.object(agente_fraude, "registrar_evento"):
        primeira = agente_fraude.registrar_confirmacao_pedida_pelo_banco(
            "Joao", 100.0, "Campinas", "viagem", "diurno"
        )
        segunda = agente_fraude.registrar_confirmacao_pedida_pelo_banco(
            "Joao", 100.0, "Campinas", "viagem", "diurno"
        )

    assert primeira is segunda
    assert len(banco_de_dados_limpo["confirmacoes_cliente"]) == 1


def test_consultar_documento_rag_usa_retriever_quando_disponivel(agente_fraude, tmp_path):
    doc = MagicMock()
    doc.page_content = "Trecho relevante sobre cidade nova e confirmação."
    retriever = MagicMock(invoke=MagicMock(return_value=[doc]))

    resposta = agente_fraude.consultar_documento_rag(
        "cidade nova",
        retriever,
        "documento teste",
        str(tmp_path / "nao_usado.txt"),
    )

    assert "Contexto recuperado de documento teste" in resposta
    assert "Trecho relevante sobre cidade nova" in resposta


def test_consultar_documento_rag_fallback_textual_quando_retriever_indisponivel(agente_fraude, tmp_path):
    arquivo = tmp_path / "politicas.txt"
    arquivo.write_text(
        "Bloco inicial sem alvo.\n\nCidade nova exige confirmação do banco.\n\nOutro bloco.",
        encoding="utf-8",
    )

    resposta = agente_fraude.consultar_documento_rag(
        "cidade confirmação",
        None,
        "politicas teste",
        str(arquivo),
    )

    assert "Contexto recuperado de politicas teste por busca textual" in resposta
    assert "Cidade nova exige confirmação" in resposta


def test_consultar_pdfs_rag_sem_arquivos_retorna_mensagem(agente_fraude):
    with patch.object(agente_fraude, "listar_pdfs_disponiveis", return_value=[]):
        resposta = agente_fraude.consultar_pdfs_rag("pergunta")

    assert "Nenhum PDF disponível" in resposta


def test_consultar_pdfs_rag_com_retriever_retorna_contexto(agente_fraude, tmp_path):
    caminho_pdf = tmp_path / "manual.pdf"
    caminho_pdf.write_bytes(b"%PDF-1.4")
    documento = MagicMock()
    documento.page_content = "Conteúdo do PDF sobre decisão coletiva."
    documento.metadata = {"source": str(caminho_pdf), "page": 0}
    retriever = MagicMock(invoke=MagicMock(return_value=[documento]))

    with patch.object(agente_fraude, "build_pdf_retriever", return_value=retriever):
        resposta = agente_fraude.consultar_pdfs_rag("decisão", [str(caminho_pdf)])

    assert "Contexto recuperado dos PDFs" in resposta
    assert "manual.pdf, página 1" in resposta
    assert "Conteúdo do PDF sobre decisão coletiva" in resposta


def test_consultar_banco_de_dados_exibe_chaves_atuais(agente_fraude, banco_de_dados_limpo):
    banco_de_dados_limpo["relatorios_fraude"].append({"cliente": "Joao", "classificacao": "normal"})

    with patch.object(agente_fraude, "banco_de_dados", banco_de_dados_limpo), patch.object(
        agente_fraude, "MODELOS_EXTERNOS_DISPONIVEIS", False
    ):
        texto = agente_fraude.consultar_banco_de_dados()

    assert "Banco de dados simulado:" in texto
    assert "Perfis de contas da base:" in texto
    assert "Relatórios de fraude registrados:" in texto
    assert "Base de dados e modelos externos:" in texto


def test_valor_texto_padrao_inicializa_session_state(agente_fraude):
    agente_fraude.st.session_state.pop("cidade_teste", None)

    assert agente_fraude.valor_texto_padrao("cidade_teste", "Curitiba") == "Curitiba"
    agente_fraude.st.session_state["cidade_teste"] = "Recife"
    assert agente_fraude.valor_texto_padrao("cidade_teste", "Curitiba") == "Recife"


@patch(f"{MODULE_NAME}.llm_com_tools")
def test_agente_monta_prompt_atual(mock_llm_com_tools, agente_fraude):
    mock_llm_com_tools.invoke.return_value = agente_fraude.HumanMessage(content="resposta")
    state = {"messages": [agente_fraude.HumanMessage(content="Consulte políticas de fraude.")]}

    resultado = agente_fraude.agente(state)

    mock_llm_com_tools.invoke.assert_called_once()
    chamadas = mock_llm_com_tools.invoke.call_args.args[0]
    assert isinstance(chamadas[0], agente_fraude.SystemMessage)
    assert "consultar_politicas_antifraude" in chamadas[0].content
    assert chamadas[1].content == "Consulte políticas de fraude."
    assert resultado["messages"][0].content == "resposta"
