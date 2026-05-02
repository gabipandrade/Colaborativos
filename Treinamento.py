import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Caminhos
# ============================================================

CAMINHO_TRANSACOES = "dados/bank_transactions_data_2.csv"

PASTA_MODELOS = "modelos"
CAMINHO_MODELO = "modelos/modelo_cluster.pkl"

CAMINHO_PERFIS_CLUSTERS = "dados/perfis_clusters.csv"
CAMINHO_TRANSACOES_CLUSTERIZADAS = "dados/transacoes_clusterizadas.csv"


# ============================================================
# Carregamento do dataset
# ============================================================

def carregar_dataset():
    """
    Carrega o dataset de transações bancárias.

    Colunas esperadas:
    TransactionID, AccountID, TransactionAmount, TransactionDate,
    TransactionType, Location, DeviceID, IP Address, MerchantID,
    AccountBalance, PreviousTransactionDate, Channel, CustomerAge,
    CustomerOccupation, TransactionDuration, LoginAttempts
    """

    if not os.path.exists(CAMINHO_TRANSACOES):
        raise FileNotFoundError(f"Arquivo não encontrado: {CAMINHO_TRANSACOES}")

    df = pd.read_csv(CAMINHO_TRANSACOES)

    colunas_obrigatorias = [
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

    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            raise ValueError(f"A coluna obrigatória '{coluna}' não existe no dataset.")

    return df


# ============================================================
# Engenharia de atributos
# ============================================================

def criar_atributos_derivados(df):
    """
    Cria atributos úteis para o modelo não supervisionado.

    Como o K-Means trabalha com números e categorias transformadas,
    datas precisam ser convertidas em atributos numéricos.
    """

    df = df.copy()

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")

    # Hora da transação
    df["TransactionHour"] = df["TransactionDate"].dt.hour

    # Dia da semana: segunda=0, domingo=6
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek

    # Mês da transação
    df["TransactionMonth"] = df["TransactionDate"].dt.month

    # Tempo desde a transação anterior, em horas
    df["HoursSincePreviousTransaction"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds() / 3600

    # Tratamento de valores ausentes gerados por datas inválidas
    df["TransactionHour"] = df["TransactionHour"].fillna(-1)
    df["TransactionDayOfWeek"] = df["TransactionDayOfWeek"].fillna(-1)
    df["TransactionMonth"] = df["TransactionMonth"].fillna(-1)
    df["HoursSincePreviousTransaction"] = df["HoursSincePreviousTransaction"].fillna(-1)

    # Garantir tipos numéricos
    df["TransactionAmount"] = pd.to_numeric(df["TransactionAmount"], errors="coerce").fillna(0)
    df["AccountBalance"] = pd.to_numeric(df["AccountBalance"], errors="coerce").fillna(0)
    df["CustomerAge"] = pd.to_numeric(df["CustomerAge"], errors="coerce").fillna(0)
    df["TransactionDuration"] = pd.to_numeric(df["TransactionDuration"], errors="coerce").fillna(0)
    df["LoginAttempts"] = pd.to_numeric(df["LoginAttempts"], errors="coerce").fillna(0)

    # Garantir texto nas categóricas
    colunas_texto = [
        "AccountID",
        "TransactionType",
        "Location",
        "DeviceID",
        "IP Address",
        "MerchantID",
        "Channel",
        "CustomerOccupation"
    ]

    for coluna in colunas_texto:
        df[coluna] = df[coluna].astype(str).fillna("desconhecido")

    return df


# ============================================================
# Pré-processamento
# ============================================================

def preparar_dados():
    """
    Define quais colunas serão usadas no treinamento.

    Importante:
    TransactionID não entra no treinamento, pois é apenas identificador.
    TransactionDate e PreviousTransactionDate também não entram diretamente,
    pois foram convertidas em atributos derivados.
    """

    colunas_numericas = [
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

    colunas_categoricas = [
        "AccountID",
        "TransactionType",
        "Location",
        "DeviceID",
        "IP Address",
        "MerchantID",
        "Channel",
        "CustomerOccupation"
    ]

    pre_processador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas)
        ]
    )

    return pre_processador, colunas_numericas, colunas_categoricas


# ============================================================
# Treinamento
# ============================================================

def treinar_modelo(df, n_clusters=5):
    """
    Treina um modelo K-Means para agrupar transações semelhantes.

    Como é aprendizado não supervisionado, o modelo não usa rótulo de fraude.
    Ele apenas aprende agrupamentos de comportamento.
    """

    pre_processador, colunas_numericas, colunas_categoricas = preparar_dados()

    modelo = Pipeline(
        steps=[
            ("preprocessamento", pre_processador),
            ("clusterizador", KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            ))
        ]
    )

    X = df[colunas_numericas + colunas_categoricas]

    modelo.fit(X)

    clusters = modelo.predict(X)
    df["cluster"] = clusters

    X_transformado = modelo.named_steps["preprocessamento"].transform(X)

    score = silhouette_score(X_transformado, clusters)

    return modelo, df, score


# ============================================================
# Perfil dos clusters
# ============================================================

def top_valores(serie, n=3):
    """
    Retorna os valores mais frequentes de uma coluna em formato texto.
    """

    if serie.empty:
        return "N/A"

    return ", ".join(serie.value_counts().head(n).index.astype(str))


def gerar_perfis_clusters(df):
    """
    Gera uma tabela explicativa sobre os clusters encontrados.

    Essa tabela será usada pelo agente para justificar futuras análises.
    """

    perfis = []

    for cluster_id in sorted(df["cluster"].unique()):
        grupo = df[df["cluster"] == cluster_id]

        perfil = {
            "cluster": cluster_id,
            "quantidade_transacoes": len(grupo),

            "valor_medio_transacao": round(grupo["TransactionAmount"].mean(), 2),
            "valor_minimo_transacao": round(grupo["TransactionAmount"].min(), 2),
            "valor_maximo_transacao": round(grupo["TransactionAmount"].max(), 2),

            "saldo_medio_conta": round(grupo["AccountBalance"].mean(), 2),
            "idade_media_cliente": round(grupo["CustomerAge"].mean(), 2),
            "duracao_media_transacao": round(grupo["TransactionDuration"].mean(), 2),
            "media_tentativas_login": round(grupo["LoginAttempts"].mean(), 2),

            "hora_media_transacao": round(grupo["TransactionHour"].mean(), 2),
            "tempo_medio_desde_transacao_anterior_horas": round(
                grupo["HoursSincePreviousTransaction"].mean(),
                2
            ),

            "contas_mais_comuns": top_valores(grupo["AccountID"]),
            "tipos_transacao_mais_comuns": top_valores(grupo["TransactionType"]),
            "locais_mais_comuns": top_valores(grupo["Location"]),
            "dispositivos_mais_comuns": top_valores(grupo["DeviceID"]),
            "ips_mais_comuns": top_valores(grupo["IP Address"]),
            "comerciantes_mais_comuns": top_valores(grupo["MerchantID"]),
            "canais_mais_comuns": top_valores(grupo["Channel"]),
            "ocupacoes_mais_comuns": top_valores(grupo["CustomerOccupation"]),
        }

        # Interpretação simples do risco do cluster
        risco = 0
        motivos_risco = []

        if perfil["valor_medio_transacao"] > df["TransactionAmount"].mean() * 1.5:
            risco += 25
            motivos_risco.append("valor médio acima da média geral")

        if perfil["media_tentativas_login"] > df["LoginAttempts"].mean() * 1.5:
            risco += 30
            motivos_risco.append("média elevada de tentativas de login")

        if perfil["tempo_medio_desde_transacao_anterior_horas"] < df["HoursSincePreviousTransaction"].median():
            risco += 15
            motivos_risco.append("frequência de transações mais alta que o padrão")

        if perfil["duracao_media_transacao"] > df["TransactionDuration"].mean() * 1.5:
            risco += 10
            motivos_risco.append("duração média elevada das transações")

        if risco >= 50:
            nivel_risco = "alto"
        elif risco >= 25:
            nivel_risco = "medio"
        else:
            nivel_risco = "baixo"

        perfil["pontuacao_risco_cluster"] = risco
        perfil["nivel_risco_cluster"] = nivel_risco
        perfil["motivos_risco_cluster"] = (
            "; ".join(motivos_risco)
            if motivos_risco
            else "cluster próximo de padrões comuns da base"
        )

        perfis.append(perfil)

    return pd.DataFrame(perfis)


# ============================================================
# Execução principal
# ============================================================

def main():
    os.makedirs(PASTA_MODELOS, exist_ok=True)

    print("Carregando dataset...")
    df = carregar_dataset()

    print(f"Total de transações carregadas: {len(df)}")

    print("Criando atributos derivados...")
    df = criar_atributos_derivados(df)

    print("Treinando modelo não supervisionado com KMeans...")
    modelo, df_com_clusters, score = treinar_modelo(df, n_clusters=5)

    print(f"Silhouette Score: {score:.4f}")

    print("Gerando perfis dos clusters...")
    perfis_clusters = gerar_perfis_clusters(df_com_clusters)

    print("Salvando modelo treinado...")
    joblib.dump(modelo, CAMINHO_MODELO)

    print("Salvando perfis dos clusters...")
    perfis_clusters.to_csv(CAMINHO_PERFIS_CLUSTERS, index=False)

    print("Salvando transações com cluster atribuído...")
    df_com_clusters.to_csv(CAMINHO_TRANSACOES_CLUSTERIZADAS, index=False)

    print("\nTreinamento finalizado com sucesso.")
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Perfis dos clusters salvos em: {CAMINHO_PERFIS_CLUSTERS}")
    print(f"Transações clusterizadas salvas em: {CAMINHO_TRANSACOES_CLUSTERIZADAS}")

    print("\nResumo dos clusters:")
    print(perfis_clusters)


if __name__ == "__main__":
    main()