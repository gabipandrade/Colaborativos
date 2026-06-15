import os

try:
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    MODELOS_DADOS_DISPONIVEIS = True
except ImportError:
    joblib = None
    pd = None
    ColumnTransformer = None
    IsolationForest = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    MODELOS_DADOS_DISPONIVEIS = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, "dados")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")

CAMINHO_TRANSACOES = os.path.join(DADOS_DIR, "bank_transactions_data_2.csv")
CAMINHO_TRANSACOES_CLUSTERIZADAS = os.path.join(DADOS_DIR, "transacoes_clusterizadas.csv")
CAMINHO_PERFIS_CLUSTERS = os.path.join(DADOS_DIR, "perfis_clusters.csv")
CAMINHO_MODELO_CLUSTER = os.path.join(MODELOS_DIR, "modelo_cluster.pkl")
CAMINHO_MODELO_ISOLATION = os.path.join(MODELOS_DIR, "modelo_isolation_forest.pkl")

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
    "LoginAttempts",
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
    "HoursSincePreviousTransaction",
]

COLUNAS_CATEGORICAS = [
    "AccountID",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerOccupation",
]

COLUNAS_MODELO = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS


def carregar_dataset():
    if not MODELOS_DADOS_DISPONIVEIS:
        return None

    caminho = CAMINHO_TRANSACOES_CLUSTERIZADAS
    if not os.path.exists(caminho):
        caminho = CAMINHO_TRANSACOES
    if not os.path.exists(caminho):
        return None

    df = pd.read_csv(caminho)
    for coluna in COLUNAS_OBRIGATORIAS:
        if coluna not in df.columns:
            return None

    return criar_atributos_derivados(df)


def criar_atributos_derivados(df):
    df = df.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")

    df["TransactionHour"] = df["TransactionDate"].dt.hour
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["HoursSincePreviousTransaction"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds() / 3600

    for coluna in COLUNAS_NUMERICAS:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(-1)

    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = df[coluna].astype(str).fillna("desconhecido")

    return df


def criar_preprocessador():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        ]
    )


def carregar_modelo_cluster():
    if not MODELOS_DADOS_DISPONIVEIS or not os.path.exists(CAMINHO_MODELO_CLUSTER):
        return None
    return joblib.load(CAMINHO_MODELO_CLUSTER)


def carregar_perfis_clusters():
    if not MODELOS_DADOS_DISPONIVEIS or not os.path.exists(CAMINHO_PERFIS_CLUSTERS):
        return pd.DataFrame() if pd is not None else None
    return pd.read_csv(CAMINHO_PERFIS_CLUSTERS)


def perfis_clusters_por_id():
    if PERFIS_CLUSTERS is None or getattr(PERFIS_CLUSTERS, "empty", True):
        return {}

    perfis = {}
    for _, linha in PERFIS_CLUSTERS.iterrows():
        if "cluster" in linha:
            perfis[int(linha["cluster"])] = linha.to_dict()
    return perfis


def carregar_ou_treinar_isolation(df):
    if not MODELOS_DADOS_DISPONIVEIS or df is None:
        return None
    if os.path.exists(CAMINHO_MODELO_ISOLATION):
        return joblib.load(CAMINHO_MODELO_ISOLATION)

    os.makedirs(MODELOS_DIR, exist_ok=True)
    modelo = Pipeline(
        steps=[
            ("preprocessamento", criar_preprocessador()),
            (
                "detector_anomalia",
                IsolationForest(n_estimators=200, contamination=0.08, random_state=42),
            ),
        ]
    )
    modelo.fit(df[COLUNAS_MODELO])
    joblib.dump(modelo, CAMINHO_MODELO_ISOLATION)
    return modelo


def gerar_perfis_contas(df):
    if df is None:
        return {}

    perfis = {}
    for account_id, grupo in df.groupby("AccountID"):
        local = grupo["Location"].mode()
        canal = grupo["Channel"].mode()
        ocupacao = grupo["CustomerOccupation"].mode()
        hora_media = grupo["TransactionHour"].mean()
        horario_comum = "diurno" if 6 <= hora_media < 18 else "noturno"

        perfis[str(account_id)] = {
            "cidade_padrao": str(local.iloc[0]) if not local.empty else "desconhecido",
            "valor_medio": round(float(grupo["TransactionAmount"].mean()), 2),
            "categorias_comuns": list(grupo["TransactionType"].value_counts().head(3).index.astype(str)),
            "canal_mais_comum": str(canal.iloc[0]) if not canal.empty else "desconhecido",
            "ocupacao_mais_comum": str(ocupacao.iloc[0]) if not ocupacao.empty else "desconhecido",
            "horario_comum": horario_comum,
            "quantidade_transacoes": int(len(grupo)),
        }
    return perfis


def transformar_transacoes_para_lista(df, limite=500):
    if df is None:
        return []

    transacoes = []
    for _, linha in df.head(limite).iterrows():
        hora = int(linha["TransactionHour"]) if linha["TransactionHour"] != -1 else 12
        periodo = "diurno" if 6 <= hora < 18 else "noturno"
        transacoes.append(
            {
                "TransactionID": str(linha["TransactionID"]),
                "AccountID": str(linha["AccountID"]),
                "TransactionAmount": float(linha["TransactionAmount"]),
                "TransactionType": str(linha["TransactionType"]),
                "Location": str(linha["Location"]),
                "Channel": str(linha["Channel"]),
                "CustomerAge": float(linha["CustomerAge"]),
                "CustomerOccupation": str(linha["CustomerOccupation"]),
                "TransactionDuration": float(linha["TransactionDuration"]),
                "LoginAttempts": int(linha["LoginAttempts"]),
                "AccountBalance": float(linha["AccountBalance"]),
                "cliente": str(linha["AccountID"]),
                "valor": float(linha["TransactionAmount"]),
                "cidade": str(linha["Location"]),
                "categoria": str(linha["TransactionType"]),
                "horario": periodo,
            }
        )
    return transacoes


DF_TRANSACOES = carregar_dataset()
MODELO_CLUSTER = carregar_modelo_cluster()
PERFIS_CLUSTERS = carregar_perfis_clusters()
MODELO_ISOLATION = carregar_ou_treinar_isolation(DF_TRANSACOES)
PERFIS_CONTAS = gerar_perfis_contas(DF_TRANSACOES)
TRANSACOES_CONSULTA = transformar_transacoes_para_lista(DF_TRANSACOES)


def base_disponivel():
    return MODELOS_DADOS_DISPONIVEIS and DF_TRANSACOES is not None and MODELO_ISOLATION is not None


def montar_dataframe_transacao(
    account_id,
    transaction_amount,
    transaction_type,
    location,
    channel,
    customer_age,
    customer_occupation,
    transaction_duration,
    login_attempts,
    account_balance,
    device_id="UNKNOWN_DEVICE",
    ip_address="0.0.0.0",
    merchant_id="UNKNOWN_MERCHANT",
    transaction_date="",
    previous_transaction_date="",
):
    if not transaction_date:
        transaction_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    if not previous_transaction_date:
        previous_transaction_date = (pd.Timestamp.now() - pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    nova = pd.DataFrame(
        [
            {
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
                "LoginAttempts": login_attempts,
            }
        ]
    )
    return criar_atributos_derivados(nova)


def analisar_dataframe_transacao(df_nova):
    if not base_disponivel():
        return {
            "status": "indisponivel",
            "acao_predita": "modelo indisponível",
            "pontuacao_risco": 0,
            "classificacao_anomalia": "indisponivel",
            "decision_score": 0.0,
            "risco_isolation": 0,
            "risco_cluster": 0,
            "risco_regras": 0,
            "motivos": ["Base de dados ou modelos não estão disponíveis."],
            "cluster_predito": None,
            "perfil_cluster": None,
        }

    x_novo = df_nova[COLUNAS_MODELO]
    linha = df_nova.iloc[0]

    pred_anomalia = int(MODELO_ISOLATION.predict(x_novo)[0])
    decision_score = float(MODELO_ISOLATION.decision_function(x_novo)[0])
    classificacao_anomalia = "anomalia" if pred_anomalia == -1 else "normal"
    risco_isolation = int(max(0, min(100, 50 - decision_score * 100)))

    cluster_predito = None
    perfil_cluster = None
    risco_cluster = 0
    if MODELO_CLUSTER is not None:
        try:
            cluster_predito = int(MODELO_CLUSTER.predict(x_novo)[0])
            perfis_cluster = perfis_clusters_por_id()
            perfil_cluster = perfis_cluster.get(cluster_predito)
            if perfil_cluster and "pontuacao_risco_cluster" in perfil_cluster:
                risco_cluster = int(perfil_cluster["pontuacao_risco_cluster"])
        except Exception:
            cluster_predito = None
            perfil_cluster = None
            risco_cluster = 0

    risco_regras = 0
    motivos = []
    media_valor = DF_TRANSACOES["TransactionAmount"].mean()
    desvio_valor = DF_TRANSACOES["TransactionAmount"].std()
    media_duracao = DF_TRANSACOES["TransactionDuration"].mean()

    account_id = str(linha["AccountID"])
    perfil_conta = PERFIS_CONTAS.get(account_id)
    if perfil_conta:
        valor_medio_conta = perfil_conta["valor_medio"]
        cidade_padrao = perfil_conta["cidade_padrao"]
        canal_padrao = perfil_conta["canal_mais_comum"]
        ocupacao_padrao = perfil_conta["ocupacao_mais_comum"]
        tipos_comuns = perfil_conta["categorias_comuns"]
    else:
        valor_medio_conta = media_valor
        cidade_padrao = None
        canal_padrao = None
        ocupacao_padrao = None
        tipos_comuns = []
        risco_regras += 15
        motivos.append("A conta não possui perfil histórico suficiente na base.")

    if valor_medio_conta > 0 and linha["TransactionAmount"] > valor_medio_conta * 5:
        risco_regras += 35
        motivos.append(f"Valor muito acima da média da conta: R$ {valor_medio_conta:.2f}.")
    if linha["TransactionAmount"] > media_valor + 3 * desvio_valor:
        risco_regras += 30
        motivos.append(f"Valor muito acima do padrão geral da base: média R$ {media_valor:.2f}.")
    if linha["LoginAttempts"] >= 5:
        risco_regras += 30
        motivos.append(f"Número muito elevado de tentativas de login: {int(linha['LoginAttempts'])}.")
    elif linha["LoginAttempts"] >= 3:
        risco_regras += 20
        motivos.append(f"Número elevado de tentativas de login: {int(linha['LoginAttempts'])}.")
    if linha["TransactionDuration"] > media_duracao * 3:
        risco_regras += 20
        motivos.append(f"Duração muito acima da média: {linha['TransactionDuration']:.2f}s.")
    if cidade_padrao and str(linha["Location"]) != cidade_padrao:
        risco_regras += 15
        motivos.append(f"Localização diferente do padrão da conta: {cidade_padrao}.")
    if canal_padrao and str(linha["Channel"]) != canal_padrao:
        risco_regras += 10
        motivos.append(f"Canal diferente do padrão da conta: {canal_padrao}.")
    if tipos_comuns and str(linha["TransactionType"]) not in tipos_comuns:
        risco_regras += 10
        motivos.append(f"Tipo de transação incomum para a conta: {linha['TransactionType']}.")
    if ocupacao_padrao and str(linha["CustomerOccupation"]) != ocupacao_padrao:
        risco_regras += 10
        motivos.append(f"Ocupação difere do perfil histórico: {ocupacao_padrao}.")

    pontuacao_risco = int(max(0, min(100, max(risco_isolation, risco_cluster, risco_regras))))
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

    motivos_base = [
        f"Isolation Forest classificou como {classificacao_anomalia}.",
        f"Risco pelo Isolation Forest: {risco_isolation}/100.",
        f"Risco por regras interpretáveis: {risco_regras}/100.",
    ]
    if cluster_predito is not None:
        motivos_base.append(f"K-Means associou a transação ao cluster {cluster_predito}.")
    if risco_cluster:
        motivos_base.append(f"Risco associado ao cluster: {risco_cluster}/100.")

    return {
        "status": status,
        "acao_predita": acao_predita,
        "pontuacao_risco": pontuacao_risco,
        "classificacao_anomalia": classificacao_anomalia,
        "decision_score": decision_score,
        "risco_isolation": risco_isolation,
        "risco_cluster": risco_cluster,
        "risco_regras": risco_regras,
        "motivos": motivos_base + motivos,
        "cluster_predito": cluster_predito,
        "perfil_cluster": perfil_cluster,
    }

def analisar_nova_transacao(**kwargs):
    df_nova = montar_dataframe_transacao(**kwargs)
    resultado = analisar_dataframe_transacao(df_nova)
    return df_nova, resultado


def analisar_transacao_por_id(transaction_id):
    if DF_TRANSACOES is None:
        return None, {"status": "indisponivel", "motivos": ["Base de transações não carregada."]}
    df_filtrado = DF_TRANSACOES[DF_TRANSACOES["TransactionID"].astype(str) == str(transaction_id)]
    if df_filtrado.empty:
        return None, {"status": "nao_encontrada", "motivos": [f"TransactionID {transaction_id} não encontrado."]}
    df_nova = df_filtrado.iloc[[0]].copy()
    return df_nova, analisar_dataframe_transacao(df_nova)


def formatar_resultado(df_nova, resultado, origem="modelo"):
    if df_nova is None:
        return "\n".join(resultado.get("motivos", ["Transação não encontrada."]))

    linha = df_nova.iloc[0]
    texto = "Resultado da análise com base de dados e modelos:\n\n"
    texto += f"Origem: {origem}\n"
    texto += f"Conta/Cliente: {linha['AccountID']}\n"
    texto += f"Valor: R$ {linha['TransactionAmount']:.2f}\n"
    texto += f"Tipo: {linha['TransactionType']}\n"
    texto += f"Localização: {linha['Location']}\n"
    texto += f"Canal: {linha['Channel']}\n"
    texto += f"Saldo após transação: R$ {linha['AccountBalance']:.2f}\n\n"
    texto += f"Ação predita: {resultado.get('acao_predita', 'indisponivel')}\n"
    texto += f"Classificação: {resultado.get('status', 'indisponivel').upper()}\n"
    texto += f"Pontuação de risco: {resultado.get('pontuacao_risco', 0)}/100\n"
    texto += f"Anomalia: {resultado.get('classificacao_anomalia', 'indisponivel')}\n"
    texto += f"Cluster: {resultado.get('cluster_predito', 'indisponivel')}\n\n"
    texto += "Justificativa:\n"
    for motivo in resultado.get("motivos", []):
        texto += f"- {motivo}\n"
    return texto


def consultar_base_modelos(pergunta):
    if DF_TRANSACOES is None:
        return "Base de dados não disponível."

    termos = [termo.lower() for termo in pergunta.replace(",", " ").replace(".", " ").split() if len(termo) > 2]
    df = DF_TRANSACOES.copy()
    partes = [
        "Base de dados de transações carregada para consulta:",
        f"- Total de transações: {len(df)}",
        f"- Total de contas: {df['AccountID'].nunique()}",
        f"- Modelos: Isolation Forest {'ativo' if MODELO_ISOLATION is not None else 'indisponível'}; K-Means {'ativo' if MODELO_CLUSTER is not None else 'indisponível'}.",
    ]

    if termos:
        mascara = pd.Series(False, index=df.index)
        for coluna in ["TransactionID", "AccountID", "TransactionType", "Location", "Channel", "CustomerOccupation"]:
            mascara = mascara | df[coluna].astype(str).str.lower().apply(lambda valor: any(termo in valor for termo in termos))
        df = df[mascara]

    partes.append(f"- Registros encontrados pela busca textual: {len(df)}")
    for _, linha in df.head(5).iterrows():
        partes.append(
            f"TransactionID {linha['TransactionID']} | Conta {linha['AccountID']} | "
            f"R$ {linha['TransactionAmount']:.2f} | {linha['TransactionType']} | "
            f"{linha['Location']} | {linha['Channel']}"
        )

    return "\n".join(partes)
