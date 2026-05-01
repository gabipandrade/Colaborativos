from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


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


tools = [
    verificar_transacao,
    gerar_relatorio_banco,
    solicitar_confirmacao_cliente,
    consultar_banco_de_dados
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
            "- consultar_banco_de_dados: use quando o usuário pedir para visualizar os dados internos.\n\n"

            "Importante:\n"
            "Quando uma transação for claramente suspeita, use as ferramentas necessárias para verificar, "
            "gerar relatório e solicitar confirmação ao cliente.\n"
            "Quando a transação parecer normal, apenas explique que ela coincide com o padrão esperado.\n"
            "Responda em português brasileiro de forma clara, organizada e objetiva."
        )
    )

    resposta = llm_com_tools.invoke([system_prompt] + state["messages"])

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