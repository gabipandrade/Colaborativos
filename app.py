import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# Banco de dados simulado GLOBAL
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


if "historico" not in st.session_state:
    st.session_state.historico = []


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


tools = [
    verificar_transacao,
    gerar_relatorio_banco,
    solicitar_confirmacao_cliente,
    consultar_banco_de_dados
]


# ============================================================
# Modelo Ollama
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
            "- consultar_banco_de_dados: use quando o usuário pedir para visualizar os dados internos.\n\n"

            "Quando uma transação for suspeita, além de verificar a transação, também gere o relatório para o banco "
            "e solicite confirmação ao cliente.\n\n"

            "Responda em português brasileiro de forma clara, organizada e objetiva."
        )
    )

    resposta = llm_com_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [resposta]}


# ============================================================
# Construção do grafo
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
# Interface Streamlit
# ============================================================

st.set_page_config(
    page_title="Colaborativo 1.0",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Colaborativo 1.0")
st.subheader("Sistema colaborativo de detecção de fraude bancária com React Agent")

st.markdown(
    """
    Este protótipo simula um sistema colaborativo entre **Banco**, **Cliente** e **IA preditiva**.

    O agente pode conversar com o usuário e também utilizar ferramentas para:
    - verificar transações;
    - gerar relatórios para o banco;
    - solicitar confirmação ao cliente;
    - consultar o banco de dados simulado.
    """
)


# ============================================================
# Barra lateral
# ============================================================

with st.sidebar:
    st.header("Clientes cadastrados")

    for cliente_sidebar, perfil in banco_de_dados["clientes"].items():
        st.markdown(f"### {cliente_sidebar}")
        st.write(f"Cidade padrão: {perfil['cidade_padrao']}")
        st.write(f"Valor médio: R$ {perfil['valor_medio']:.2f}")
        st.write(f"Categorias comuns: {', '.join(perfil['categorias_comuns'])}")
        st.write(f"Horário comum: {perfil['horario_comum']}")
        st.divider()

    if st.button("Limpar conversa"):
        st.session_state.historico = []
        st.rerun()


st.divider()


# ============================================================
# Formulário de análise rápida
# ============================================================

st.header("Análise rápida de transação")

col1, col2 = st.columns(2)

with col1:
    cliente = st.selectbox("Cliente", ["Joao", "Ana", "Carlos", "Outro"])

    if cliente == "Outro":
        cliente = st.text_input("Nome do cliente")

    valor = st.number_input("Valor da transação", min_value=0.0, value=100.0, step=10.0)
    cidade = st.text_input("Cidade", value="Sao Carlos")

with col2:
    categoria = st.selectbox(
        "Categoria",
        ["mercado", "farmacia", "restaurante", "roupas", "transporte", "eletronicos", "viagem", "outro"]
    )

    if categoria == "outro":
        categoria = st.text_input("Digite a categoria")

    horario = st.selectbox("Horário", ["diurno", "noturno"])


if st.button("Verificar transação"):
    mensagem = (
        f"O banco deseja verificar uma compra do cliente {cliente} "
        f"no valor de {valor} reais, na cidade de {cidade}, "
        f"categoria {categoria}, no horário {horario}."
    )

    st.session_state.historico.append(HumanMessage(content=mensagem))

    with st.spinner("Agente analisando a transação..."):
        resultado = graph.invoke({"messages": st.session_state.historico})
        st.session_state.historico = resultado["messages"]

    st.success("Análise concluída.")
    st.rerun()


st.divider()


# ============================================================
# Chat com o agente
# ============================================================

st.header("Chat com o agente")

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


st.divider()


# ============================================================
# Visualização do banco de dados simulado
# ============================================================

st.header("Banco de dados simulado")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Relatórios gerados")
    if banco_de_dados["relatorios_fraude"]:
        st.json(banco_de_dados["relatorios_fraude"])
    else:
        st.write("Nenhum relatório registrado ainda.")

with col_b:
    st.subheader("Confirmações enviadas")
    if banco_de_dados["confirmacoes_cliente"]:
        st.json(banco_de_dados["confirmacoes_cliente"])
    else:
        st.write("Nenhuma confirmação enviada ainda.")