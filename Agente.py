import os
import joblib
import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest


# ============================================================
# Caminhos
# ============================================================

CAMINHO_TRANSACOES = "dados/bank_transactions_data_2.csv"

CAMINHO_MODELO_CLUSTER = "modelos/modelo_cluster.pkl"
CAMINHO_MODELO_ISOLATION = "modelos/modelo_isolation_forest.pkl"
CAMINHO_PERFIS_CLUSTERS = "dados/perfis_clusters.csv"


# ============================================================
# Colunas do dataset
# ============================================================

COLUNAS_OBRIGATORIAS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "AccountBalance",
    "PreviousTransactionDate",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts"
]

COLUNAS_NUMERICAS = [
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "TransactionHour",
    "TransactionDayOfWeek",
    "TransactionMonth",
    "HoursSincePreviousTransaction"
]

COLUNAS_CATEGORICAS = [
    "AccountID",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerOccupation"
]

COLUNAS_MODELO = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS


# ============================================================
# Carregamento e preparação do dataset
# ============================================================

def carregar_dataset(caminho_csv: str = CAMINHO_TRANSACOES) -> pd.DataFrame:
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv)

    for coluna in COLUNAS_OBRIGATORIAS:
        if coluna not in df.columns:
            raise ValueError(f"A coluna obrigatória '{coluna}' não existe no dataset.")

    return df


def criar_atributos_derivados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")

    df["TransactionHour"] = df["TransactionDate"].dt.hour
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df["TransactionMonth"] = df["TransactionDate"].dt.month

    df["HoursSincePreviousTransaction"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds() / 3600

    df["TransactionHour"] = df["TransactionHour"].fillna(-1)
    df["TransactionDayOfWeek"] = df["TransactionDayOfWeek"].fillna(-1)
    df["TransactionMonth"] = df["TransactionMonth"].fillna(-1)
    df["HoursSincePreviousTransaction"] = df["HoursSincePreviousTransaction"].fillna(-1)

    for coluna in [
        "TransactionAmount",
        "AccountBalance",
        "CustomerAge",
        "TransactionDuration",
        "LoginAttempts"
    ]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0)

    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = df[coluna].astype(str).fillna("desconhecido")

    return df


def transformar_transacoes_para_lista(df: pd.DataFrame) -> list:
    """
    Cria uma lista de transações para a interface.
    Mantém os nomes originais do dataset e também alguns apelidos antigos
    para evitar quebrar partes da interface já feitas.
    """

    transacoes = []

    for _, linha in df.iterrows():
        hora = int(linha["TransactionHour"]) if linha["TransactionHour"] != -1 else 12
        periodo = "diurno" if 6 <= hora < 18 else "noturno"

        transacao = {
            # nomes originais
            "TransactionID": str(linha["TransactionID"]),
            "AccountID": str(linha["AccountID"]),
            "TransactionAmount": float(linha["TransactionAmount"]),
            "TransactionDate": str(linha["TransactionDate"]),
            "TransactionType": str(linha["TransactionType"]),
            "Location": str(linha["Location"]),
            "DeviceID": str(linha["DeviceID"]),
            "IP Address": str(linha["IP Address"]),
            "MerchantID": str(linha["MerchantID"]),
            "AccountBalance": float(linha["AccountBalance"]),
            "PreviousTransactionDate": str(linha["PreviousTransactionDate"]),
            "Channel": str(linha["Channel"]),
            "CustomerAge": float(linha["CustomerAge"]),
            "CustomerOccupation": str(linha["CustomerOccupation"]),
            "TransactionDuration": float(linha["TransactionDuration"]),
            "LoginAttempts": int(linha["LoginAttempts"]),

            # apelidos para compatibilidade com a interface antiga
            "id_transacao": str(linha["TransactionID"]),
            "cliente": str(linha["AccountID"]),
            "valor": float(linha["TransactionAmount"]),
            "cidade": str(linha["Location"]),
            "categoria": str(linha["TransactionType"]),
            "horario": periodo,
            "rotulo_real": "desconhecido"
        }

        transacoes.append(transacao)

    return transacoes


def gerar_perfis_contas(df: pd.DataFrame) -> dict:
    """
    Gera perfis por AccountID a partir do próprio dataset.
    Isso substitui a necessidade de um clientes.csv manual.
    """

    perfis = {}

    for account_id, grupo in df.groupby("AccountID"):
        local_mais_comum = grupo["Location"].mode()
        tipo_mais_comum = grupo["TransactionType"].mode()
        canal_mais_comum = grupo["Channel"].mode()
        ocupacao_mais_comum = grupo["CustomerOccupation"].mode()

        hora_media = grupo["TransactionHour"].mean()
        horario_comum = "diurno" if 6 <= hora_media < 18 else "noturno"

        perfis[str(account_id)] = {
            "cidade_padrao": str(local_mais_comum.iloc[0]) if not local_mais_comum.empty else "desconhecido",
            "valor_medio": round(float(grupo["TransactionAmount"].mean()), 2),
            "categorias_comuns": list(grupo["TransactionType"].value_counts().head(3).index.astype(str)),
            "canal_mais_comum": str(canal_mais_comum.iloc[0]) if not canal_mais_comum.empty else "desconhecido",
            "ocupacao_mais_comum": str(ocupacao_mais_comum.iloc[0]) if not ocupacao_mais_comum.empty else "desconhecido",
            "horario_comum": horario_comum,
            "quantidade_transacoes": len(grupo)
        }

    return perfis


# ============================================================
# Pré-processamento para modelos
# ============================================================

def criar_preprocessador():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUNAS_CATEGORICAS)
        ]
    )


def carregar_modelo_cluster():
    if not os.path.exists(CAMINHO_MODELO_CLUSTER):
        return None

    return joblib.load(CAMINHO_MODELO_CLUSTER)


def carregar_perfis_clusters():
    if not os.path.exists(CAMINHO_PERFIS_CLUSTERS):
        return pd.DataFrame()

    return pd.read_csv(CAMINHO_PERFIS_CLUSTERS)


def treinar_ou_carregar_isolation_forest(df: pd.DataFrame):
    """
    Treina ou carrega o Isolation Forest.

    O Isolation Forest é usado para detectar anomalias.
    Ele não aprende rótulos de fraude. Ele aprende o padrão geral da base
    e marca pontos muito diferentes como anômalos.
    """

    os.makedirs("modelos", exist_ok=True)

    if os.path.exists(CAMINHO_MODELO_ISOLATION):
        return joblib.load(CAMINHO_MODELO_ISOLATION)

    preprocessador = criar_preprocessador()

    modelo = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("detector_anomalia", IsolationForest(
                n_estimators=200,
                contamination=0.08,
                random_state=42
            ))
        ]
    )

    X = df[COLUNAS_MODELO]
    modelo.fit(X)

    joblib.dump(modelo, CAMINHO_MODELO_ISOLATION)

    return modelo


# ============================================================
# Inicialização dos dados e modelos
# ============================================================

df_transacoes = carregar_dataset()
df_transacoes = criar_atributos_derivados(df_transacoes)

modelo_cluster = carregar_modelo_cluster()
perfis_clusters = carregar_perfis_clusters()
modelo_isolation = treinar_ou_carregar_isolation_forest(df_transacoes)

banco_de_dados = {
    "clientes": gerar_perfis_contas(df_transacoes),
    "transacoes": transformar_transacoes_para_lista(df_transacoes),
    "relatorios_fraude": [],
    "confirmacoes_cliente": []
}


# ============================================================
# Funções auxiliares de análise
# ============================================================

def montar_dataframe_transacao(
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
    transaction_date: str = "",
    previous_transaction_date: str = ""
) -> pd.DataFrame:

    if not transaction_date:
        transaction_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    if not previous_transaction_date:
        previous_transaction_date = (
            pd.Timestamp.now() - pd.Timedelta(hours=24)
        ).strftime("%Y-%m-%d %H:%M:%S")

    nova = pd.DataFrame([{
        "TransactionID": "NOVA_TRANSACAO",
        "AccountID": account_id,
        "TransactionAmount": transaction_amount,
        "TransactionDate": transaction_date,
        "TransactionType": transaction_type,
        "Location": location,
        "DeviceID": device_id,
        "IP Address": ip_address,
        "MerchantID": merchant_id,
        "AccountBalance": account_balance,
        "PreviousTransactionDate": previous_transaction_date,
        "Channel": channel,
        "CustomerAge": customer_age,
        "CustomerOccupation": customer_occupation,
        "TransactionDuration": transaction_duration,
        "LoginAttempts": login_attempts
    }])

    nova = criar_atributos_derivados(nova)

    return nova


def calcular_analise_transacao(df_nova: pd.DataFrame) -> dict:
    """
    Executa análise combinando:
    1. Isolation Forest;
    2. K-Means, se disponível;
    3. regras interpretáveis de risco.

    A decisão final não depende apenas do Isolation Forest.
    """

    X_novo = df_nova[COLUNAS_MODELO]
    linha = df_nova.iloc[0]

    # ========================================================
    # 1. Isolation Forest
    # ========================================================

    pred_anomalia = int(modelo_isolation.predict(X_novo)[0])
    decision_score = float(modelo_isolation.decision_function(X_novo)[0])

    if pred_anomalia == -1:
        classificacao_anomalia = "anomalia"
    else:
        classificacao_anomalia = "normal"

    # Quanto menor o score, mais anômalo.
    # Este valor sozinho não será usado como decisão final.
    risco_isolation = int(max(0, min(100, 50 - decision_score * 100)))

    # ========================================================
    # 2. K-Means
    # ========================================================

    cluster_predito = None
    perfil_cluster = None
    risco_cluster = 0

    if modelo_cluster is not None:
        try:
            cluster_predito = int(modelo_cluster.predict(X_novo)[0])

            if not perfis_clusters.empty and "cluster" in perfis_clusters.columns:
                perfil = perfis_clusters[perfis_clusters["cluster"] == cluster_predito]

                if not perfil.empty:
                    perfil_cluster = perfil.iloc[0].to_dict()

                    if "pontuacao_risco_cluster" in perfil_cluster:
                        risco_cluster = int(perfil_cluster["pontuacao_risco_cluster"])

        except Exception:
            cluster_predito = None
            perfil_cluster = None
            risco_cluster = 0

    # ========================================================
    # 3. Regras interpretáveis de risco
    # ========================================================

    risco_regras = 0
    motivos_regras = []

    media_valor_geral = df_transacoes["TransactionAmount"].mean()
    desvio_valor_geral = df_transacoes["TransactionAmount"].std()

    media_duracao = df_transacoes["TransactionDuration"].mean()
    media_login = df_transacoes["LoginAttempts"].mean()

    # Perfil da própria conta, se existir
    account_id = str(linha["AccountID"])

    if account_id in banco_de_dados["clientes"]:
        perfil_conta = banco_de_dados["clientes"][account_id]
        valor_medio_conta = perfil_conta["valor_medio"]
        cidade_padrao = perfil_conta["cidade_padrao"]
        canal_padrao = perfil_conta["canal_mais_comum"]
        ocupacao_padrao = perfil_conta["ocupacao_mais_comum"]
        tipos_comuns = perfil_conta["categorias_comuns"]
    else:
        perfil_conta = None
        valor_medio_conta = media_valor_geral
        cidade_padrao = None
        canal_padrao = None
        ocupacao_padrao = None
        tipos_comuns = []

        risco_regras += 15
        motivos_regras.append("A conta não possui perfil histórico suficiente na base.")

    # Valor muito acima da média da própria conta
    if valor_medio_conta > 0 and linha["TransactionAmount"] > valor_medio_conta * 5:
        risco_regras += 35
        motivos_regras.append(
            f"Valor muito acima da média da conta. "
            f"Média da conta: R$ {valor_medio_conta:.2f}, "
            f"valor analisado: R$ {linha['TransactionAmount']:.2f}."
        )

    # Valor muito acima da média geral
    if linha["TransactionAmount"] > media_valor_geral + 3 * desvio_valor_geral:
        risco_regras += 30
        motivos_regras.append(
            f"Valor muito acima do padrão geral da base. "
            f"Média geral: R$ {media_valor_geral:.2f}, "
            f"valor analisado: R$ {linha['TransactionAmount']:.2f}."
        )

    # Muitas tentativas de login
    if linha["LoginAttempts"] >= 5:
        risco_regras += 30
        motivos_regras.append(
            f"Número muito elevado de tentativas de login: {int(linha['LoginAttempts'])}."
        )
    elif linha["LoginAttempts"] >= 3:
        risco_regras += 20
        motivos_regras.append(
            f"Número elevado de tentativas de login: {int(linha['LoginAttempts'])}."
        )

    # Duração anormalmente alta
    if linha["TransactionDuration"] > media_duracao * 3:
        risco_regras += 20
        motivos_regras.append(
            f"Duração muito acima da média. "
            f"Média: {media_duracao:.2f}s, duração analisada: {linha['TransactionDuration']:.2f}s."
        )
    elif linha["TransactionDuration"] > media_duracao * 1.5:
        risco_regras += 10
        motivos_regras.append(
            f"Duração acima da média. "
            f"Média: {media_duracao:.2f}s, duração analisada: {linha['TransactionDuration']:.2f}s."
        )

    # Cidade/localização diferente do padrão da conta
    if cidade_padrao and str(linha["Location"]) != cidade_padrao:
        risco_regras += 15
        motivos_regras.append(
            f"Localização diferente do padrão da conta. "
            f"Local comum: {cidade_padrao}, local analisado: {linha['Location']}."
        )

    # Canal diferente do padrão da conta
    if canal_padrao and str(linha["Channel"]) != canal_padrao:
        risco_regras += 10
        motivos_regras.append(
            f"Canal diferente do padrão da conta. "
            f"Canal comum: {canal_padrao}, canal analisado: {linha['Channel']}."
        )

    # Tipo de transação incomum
    if tipos_comuns and str(linha["TransactionType"]) not in tipos_comuns:
        risco_regras += 10
        motivos_regras.append(
            f"Tipo de transação incomum para a conta: {linha['TransactionType']}."
        )

    # Ocupação diferente do perfil histórico da conta
    if ocupacao_padrao and str(linha["CustomerOccupation"]) != ocupacao_padrao:
        risco_regras += 10
        motivos_regras.append(
            f"Ocupação informada difere do perfil histórico da conta. "
            f"Ocupação comum: {ocupacao_padrao}, ocupação analisada: {linha['CustomerOccupation']}."
        )

    # Saldo baixo após transação de valor alto
    if linha["TransactionAmount"] > media_valor_geral * 5 and linha["AccountBalance"] < media_valor_geral:
        risco_regras += 15
        motivos_regras.append(
            "Transação de valor muito alto associada a saldo restante baixo."
        )

    # ========================================================
    # 4. Decisão final combinada
    # ========================================================

    pontuacao_risco = max(
        risco_isolation,
        risco_cluster,
        risco_regras
    )

    pontuacao_risco = int(max(0, min(100, pontuacao_risco)))

    if classificacao_anomalia == "anomalia":
        pontuacao_risco = max(pontuacao_risco, 65)

    if pontuacao_risco >= 60:
        status = "suspeita"
        acao_predita = "possível fraude"
    elif pontuacao_risco >= 40:
        status = "atenção"
        acao_predita = "transação de risco intermediário"
    else:
        status = "normal"
        acao_predita = "transação aparentemente legítima"

    return {
        "status": status,
        "acao_predita": acao_predita,
        "classificacao_anomalia": classificacao_anomalia,
        "pontuacao_risco": pontuacao_risco,
        "decision_score": decision_score,
        "risco_isolation": risco_isolation,
        "risco_cluster": risco_cluster,
        "risco_regras": risco_regras,
        "motivos_regras": motivos_regras,
        "cluster_predito": cluster_predito,
        "perfil_cluster": perfil_cluster
    }

def gerar_justificativa(df_nova: pd.DataFrame, resultado: dict) -> list:
    linha = df_nova.iloc[0]

    motivos = []

    if resultado["classificacao_anomalia"] == "anomalia":
        motivos.append(
            "O Isolation Forest classificou a transação como anômala em relação ao padrão geral da base."
        )
    else:
        motivos.append(
            "O Isolation Forest classificou a transação como normal, porém a decisão final também considera regras interpretáveis de risco."
        )

    motivos.append(
        f"Risco pelo Isolation Forest: {resultado['risco_isolation']}/100."
    )

    motivos.append(
        f"Risco pelas regras interpretáveis: {resultado['risco_regras']}/100."
    )

    if resultado["risco_cluster"] > 0:
        motivos.append(
            f"Risco associado ao cluster: {resultado['risco_cluster']}/100."
        )

    for motivo in resultado["motivos_regras"]:
        motivos.append(motivo)

    if resultado["cluster_predito"] is not None:
        motivos.append(
            f"A transação foi associada ao cluster {resultado['cluster_predito']} aprendido pelo K-Means."
        )

    if resultado["perfil_cluster"] is not None:
        perfil = resultado["perfil_cluster"]

        if "nivel_risco_cluster" in perfil:
            motivos.append(
                f"O cluster associado possui nível de risco '{perfil['nivel_risco_cluster']}'."
            )

        if "motivos_risco_cluster" in perfil:
            motivos.append(
                f"Motivos associados ao cluster: {perfil['motivos_risco_cluster']}."
            )

    if not motivos:
        motivos.append("A transação não apresentou fatores relevantes de risco.")

    return motivos

def formatar_resultado(df_nova: pd.DataFrame, resultado: dict, origem: str = "manual") -> str:
    linha = df_nova.iloc[0]
    motivos = gerar_justificativa(df_nova, resultado)

    texto = "Resultado da análise preditiva e não supervisionada:\n\n"

    texto += f"Origem da análise: {origem}\n"
    texto += f"Conta/Cliente: {linha['AccountID']}\n"
    texto += f"Valor: R$ {linha['TransactionAmount']:.2f}\n"
    texto += f"Tipo de transação: {linha['TransactionType']}\n"
    texto += f"Localização: {linha['Location']}\n"
    texto += f"Canal: {linha['Channel']}\n"
    texto += f"Dispositivo: {linha['DeviceID']}\n"
    texto += f"IP: {linha['IP Address']}\n"
    texto += f"Comerciante: {linha['MerchantID']}\n"
    texto += f"Idade do cliente: {linha['CustomerAge']}\n"
    texto += f"Ocupação: {linha['CustomerOccupation']}\n"
    texto += f"Duração da transação: {linha['TransactionDuration']} segundos\n"
    texto += f"Tentativas de login: {int(linha['LoginAttempts'])}\n"
    texto += f"Saldo após transação: R$ {linha['AccountBalance']:.2f}\n\n"

    texto += f"Ação predita pela IA: {resultado['acao_predita']}\n"
    texto += f"Classificação por anomalia: {resultado['classificacao_anomalia'].upper()}\n"
    texto += f"Pontuação de risco estimada: {resultado['pontuacao_risco']}/100\n"
    texto += f"Score interno do Isolation Forest: {resultado['decision_score']:.4f}\n"

    if resultado["cluster_predito"] is not None:
        texto += f"Cluster atribuído pelo K-Means: {resultado['cluster_predito']}\n"
    else:
        texto += "Cluster atribuído pelo K-Means: modelo de cluster não carregado.\n"

    texto += "\nJustificativa:\n"
    for motivo in motivos:
        texto += f"- {motivo}\n"

    if resultado["status"] == "suspeita":
        texto += (
            "\nConclusão: a transação deve ser tratada como SUSPEITA. "
            "Mesmo que algum modelo isolado não a classifique como anômala, "
            "os fatores combinados indicam risco elevado. "
            "Recomenda-se solicitar confirmação ao cliente e registrar relatório para o banco."
        )
    elif resultado["status"] == "atenção":
        texto += (
            "\nConclusão: a transação apresenta risco intermediário. "
            "Ela não deve ser bloqueada automaticamente, mas recomenda-se verificação adicional."
        )
    else:
        texto += (
            "\nConclusão: a transação não apresentou indícios fortes de anomalia. "
            "Ela parece compatível com os padrões aprendidos na base."
        )
    return texto


# ============================================================
# Ferramentas do agente
# ============================================================

@tool
def verificar_transacao(
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
    transaction_date: str = "",
    previous_transaction_date: str = ""
) -> str:
    """
    Verifica uma nova transação usando K-Means, quando disponível, e Isolation Forest.

    Use esta ferramenta quando o usuário fornecer os dados de uma nova transação.
    """

    df_nova = montar_dataframe_transacao(
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
        transaction_date=transaction_date,
        previous_transaction_date=previous_transaction_date
    )

    resultado = calcular_analise_transacao(df_nova)

    return formatar_resultado(df_nova, resultado, origem="nova transação informada pelo usuário")


@tool
def verificar_transacao_por_id(transaction_id: str) -> str:
    """
    Verifica uma transação existente no dataset usando TransactionID.
    """

    df_filtrado = df_transacoes[
        df_transacoes["TransactionID"].astype(str) == str(transaction_id)
    ]

    if df_filtrado.empty:
        return f"Nenhuma transação encontrada com TransactionID = {transaction_id}."

    df_nova = df_filtrado.iloc[[0]].copy()

    resultado = calcular_analise_transacao(df_nova)

    texto = f"Transação carregada da base pelo TransactionID {transaction_id}.\n\n"
    texto += formatar_resultado(df_nova, resultado, origem="transação carregada do dataset")

    return texto


@tool
def gerar_relatorio_banco(
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
    merchant_id: str = "UNKNOWN_MERCHANT"
) -> str:
    """
    Gera um relatório para o banco a partir dos dados de uma transação analisada.
    """

    df_nova = montar_dataframe_transacao(
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
        merchant_id=merchant_id
    )

    resultado = calcular_analise_transacao(df_nova)
    motivos = gerar_justificativa(df_nova, resultado)

    relatorio = {
        "account_id": account_id,
        "transaction_amount": transaction_amount,
        "transaction_type": transaction_type,
        "location": location,
        "channel": channel,
        "customer_age": customer_age,
        "customer_occupation": customer_occupation,
        "transaction_duration": transaction_duration,
        "login_attempts": login_attempts,
        "account_balance": account_balance,
        "device_id": device_id,
        "ip_address": ip_address,
        "merchant_id": merchant_id,
        "acao_predita": resultado["acao_predita"],
        "classificacao_anomalia": resultado["classificacao_anomalia"],
        "pontuacao_risco": resultado["pontuacao_risco"],
        "cluster_predito": resultado["cluster_predito"],
        "motivos": motivos
    }

    banco_de_dados["relatorios_fraude"].append(relatorio)

    texto = "Relatório registrado para o banco:\n\n"
    texto += f"Conta/Cliente: {account_id}\n"
    texto += f"Classificação: {resultado['status'].upper()}\n"
    texto += f"Pontuação de risco: {resultado['pontuacao_risco']}/100\n"
    texto += f"Classificação por anomalia: {resultado['classificacao_anomalia']}\n"
    texto += f"Cluster: {resultado['cluster_predito']}\n\n"

    texto += "Fatores considerados:\n"
    for motivo in motivos:
        texto += f"- {motivo}\n"

    return texto


@tool
def solicitar_confirmacao_cliente(account_id: str, descricao_transacao: str) -> str:
    """
    Solicita ao cliente/conta uma confirmação de que a transação suspeita foi realmente feita por ele.
    """

    mensagem = {
        "account_id": account_id,
        "descricao_transacao": descricao_transacao,
        "status": "aguardando confirmação"
    }

    banco_de_dados["confirmacoes_cliente"].append(mensagem)

    return (
        f"Confirmação enviada para a conta {account_id}. "
        f"Mensagem: Identificamos uma transação suspeita: {descricao_transacao}. "
        f"Confirme se essa transação foi realizada por você."
    )


@tool
def consultar_banco_de_dados() -> str:
    """
    Consulta o banco de dados carregado, incluindo transações, perfis de contas,
    relatórios e confirmações pendentes.
    """

    texto = "Banco de dados carregado do dataset bancário:\n\n"

    texto += f"Total de contas carregadas: {len(banco_de_dados['clientes'])}\n"
    texto += f"Total de transações carregadas: {len(banco_de_dados['transacoes'])}\n\n"

    texto += "Primeiras 5 contas/perfis gerados automaticamente:\n"
    for i, (conta, perfil) in enumerate(banco_de_dados["clientes"].items()):
        if i >= 5:
            break
        texto += f"- {conta}: {perfil}\n"

    texto += "\nPrimeiras 5 transações da base:\n"
    for transacao in banco_de_dados["transacoes"][:5]:
        texto += f"- TransactionID: {transacao['TransactionID']}, AccountID: {transacao['AccountID']}, "
        texto += f"Valor: R$ {transacao['TransactionAmount']:.2f}, Local: {transacao['Location']}, "
        texto += f"Canal: {transacao['Channel']}\n"

    texto += "\nModelos carregados:\n"
    texto += "- Isolation Forest: carregado e disponível.\n"
    texto += "- K-Means: carregado.\n" if modelo_cluster is not None else "- K-Means: não encontrado. Execute Treinamento.py para gerar modelo_cluster.pkl.\n"

    texto += "\nRelatórios registrados:\n"
    if banco_de_dados["relatorios_fraude"]:
        for i, relatorio in enumerate(banco_de_dados["relatorios_fraude"], start=1):
            texto += f"{i}. {relatorio}\n"
    else:
        texto += "Nenhum relatório registrado.\n"

    texto += "\nConfirmações enviadas:\n"
    if banco_de_dados["confirmacoes_cliente"]:
        for i, confirmacao in enumerate(banco_de_dados["confirmacoes_cliente"], start=1):
            texto += f"{i}. {confirmacao}\n"
    else:
        texto += "Nenhuma confirmação enviada.\n"

    return texto


tools = [
    verificar_transacao,
    verificar_transacao_por_id,
    gerar_relatorio_banco,
    solicitar_confirmacao_cliente,
    consultar_banco_de_dados
]


# ============================================================
# Configuração do Ollama
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

llm_com_tools = llm.bind_tools(tools)


# ============================================================
# Comportamento do agente
# ============================================================

def agente(state: MessagesState):
    system_prompt = SystemMessage(
        content=(
            "Você é um agente colaborativo de detecção de fraude bancária.\n\n"

            "O sistema possui dois participantes principais: Banco e Cliente.\n"
            "O banco envia transações para análise e o agente usa IA não supervisionada "
            "para identificar padrões e possíveis anomalias.\n\n"

            "A base de dados é o arquivo dados/bank_transactions_data_2.csv.\n"
            "As principais colunas são TransactionID, AccountID, TransactionAmount, "
            "TransactionDate, TransactionType, Location, DeviceID, IP Address, MerchantID, "
            "AccountBalance, PreviousTransactionDate, Channel, CustomerAge, "
            "CustomerOccupation, TransactionDuration e LoginAttempts.\n\n"

            "O sistema usa dois métodos de aprendizado não supervisionado:\n"
            "1. K-Means, quando o modelo estiver disponível, para associar a transação "
            "a um cluster de comportamento.\n"
            "2. Isolation Forest para detectar se a transação é anômala em relação "
            "ao padrão geral da base.\n\n"

            "Ferramentas disponíveis:\n"
            "- verificar_transacao: use quando o usuário fornecer uma nova transação manualmente.\n"
            "- verificar_transacao_por_id: use quando o usuário pedir para verificar uma transação pelo TransactionID.\n"
            "- gerar_relatorio_banco: use quando for necessário registrar a justificativa para o banco.\n"
            "- solicitar_confirmacao_cliente: use quando a transação for suspeita ou anômala.\n"
            "- consultar_banco_de_dados: use quando o usuário pedir para visualizar os dados internos.\n\n"

            "Quando uma transação for classificada como suspeita ou anômala, explique os motivos "
            "e recomende a confirmação do cliente e o registro de relatório para o banco.\n\n"

            "Responda sempre em português brasileiro, com clareza e objetividade."
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
# Funções auxiliares para a interface
# ============================================================

def criar_mensagem_usuario(texto: str) -> HumanMessage:
    return HumanMessage(content=texto)


def executar_agente(historico):
    resultado = graph.invoke({"messages": historico})
    return resultado["messages"]


def limpar_banco_de_dados():
    banco_de_dados["relatorios_fraude"].clear()
    banco_de_dados["confirmacoes_cliente"].clear()