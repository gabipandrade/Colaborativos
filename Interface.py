import os
import pandas as pd
import plotly.express as px
import streamlit as st

from Agente import (
    banco_de_dados,
    criar_mensagem_usuario,
    executar_agente,
    limpar_banco_de_dados
)


# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="Sentinel 1.0",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# Estado da aplicação
# ============================================================

if "historico" not in st.session_state:
    st.session_state.historico = []

if "ultimo_resultado" not in st.session_state:
    st.session_state.ultimo_resultado = None

if "ultima_transacao_prompt" not in st.session_state:
    st.session_state.ultima_transacao_prompt = None

if "ultima_transacao_dict" not in st.session_state:
    st.session_state.ultima_transacao_dict = None


# ============================================================
# Funções auxiliares da interface
# ============================================================

def carregar_dataframe_transacoes():
    """
    Carrega as transações para exibição na interface.

    Prioridade:
    1. dados/transacoes_clusterizadas.csv, se existir;
    2. banco_de_dados["transacoes"], carregado pelo Agente.py.
    """

    caminho_clusterizado = "dados/transacoes_clusterizadas.csv"

    if os.path.exists(caminho_clusterizado):
        df = pd.read_csv(caminho_clusterizado)
    else:
        df = pd.DataFrame(banco_de_dados["transacoes"])

    return df


def enviar_mensagem_para_agente(mensagem: str):
    """
    Envia uma mensagem ao agente e retorna a última resposta textual.
    """

    st.session_state.historico.append(criar_mensagem_usuario(mensagem))

    with st.spinner("Agente processando..."):
        st.session_state.historico = executar_agente(st.session_state.historico)

    resposta = ""

    for msg in reversed(st.session_state.historico):
        if msg.type == "ai" and msg.content:
            resposta = msg.content
            break

    return resposta


def montar_prompt_transacao(
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
    device_id,
    ip_address,
    merchant_id
):
    return (
        f"O banco recebeu uma transação da conta {account_id} "
        f"no valor de {transaction_amount}, tipo {transaction_type}, "
        f"localização {location}, canal {channel}, "
        f"cliente de {customer_age} anos, ocupação {customer_occupation}, "
        f"duração de {transaction_duration} segundos, "
        f"{login_attempts} tentativas de login, "
        f"saldo após transação de {account_balance}, "
        f"dispositivo {device_id}, IP {ip_address}, comerciante {merchant_id}. "
        f"Analise a transação, diga se é normal, atenção ou suspeita, "
        f"explique a justificativa e indique a ação recomendada."
    )

def exibir_card_resultado(texto_resultado: str):
    """
    Exibe o resultado da análise em um card visual com bom contraste.
    """

    texto_minusculo = texto_resultado.lower()

    if (
        "suspeita" in texto_minusculo
        or "possível fraude" in texto_minusculo
        or "possivel fraude" in texto_minusculo
    ):
        titulo = "🚨 Transação Suspeita"
        cor_fundo = "#fff1f2"      # vermelho bem claro
        cor_borda = "#dc2626"      # vermelho forte
        cor_titulo = "#991b1b"     # vermelho escuro
        cor_texto = "#111827"      # quase preto

    elif (
        "atenção" in texto_minusculo
        or "atencao" in texto_minusculo
        or "risco intermediário" in texto_minusculo
        or "risco intermediario" in texto_minusculo
    ):
        titulo = "⚠️ Transação em Atenção"
        cor_fundo = "#fffbeb"      # amarelo claro
        cor_borda = "#f59e0b"      # amarelo forte
        cor_titulo = "#92400e"     # marrom escuro
        cor_texto = "#111827"

    else:
        titulo = "✅ Transação Normal"
        cor_fundo = "#ecfdf5"      # verde claro
        cor_borda = "#16a34a"      # verde forte
        cor_titulo = "#166534"     # verde escuro
        cor_texto = "#111827"

    st.markdown(
        f"""
        <div style="
            padding: 22px;
            border-radius: 14px;
            background-color: {cor_fundo};
            border-left: 8px solid {cor_borda};
            border-top: 1px solid {cor_borda};
            border-right: 1px solid {cor_borda};
            border-bottom: 1px solid {cor_borda};
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        ">
            <h3 style="
                color: {cor_titulo};
                margin-top: 0;
                margin-bottom: 12px;
                font-weight: 700;
            ">
                {titulo}
            </h3>
            <div style="
                color: {cor_texto};
                font-size: 16px;
                line-height: 1.6;
                white-space: pre-wrap;
            ">
                {texto_resultado}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mostrar_metricas_gerais(df):
    total_transacoes = len(df)
    total_contas = len(banco_de_dados["clientes"])
    total_relatorios = len(banco_de_dados["relatorios_fraude"])
    total_confirmacoes = len(banco_de_dados["confirmacoes_cliente"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de transações", total_transacoes)
    col2.metric("Total de contas", total_contas)
    col3.metric("Alertas/relatórios", total_relatorios)
    col4.metric("Confirmações pendentes", total_confirmacoes)


def detectar_coluna_cluster(df):
    for nome in ["cluster", "Cluster", "kmeans_cluster"]:
        if nome in df.columns:
            return nome
    return None


def detectar_coluna_anomalia(df):
    for nome in ["anomalia_isolation_forest", "is_anomaly", "anomalia", "anomaly"]:
        if nome in df.columns:
            return nome
    return None


# ============================================================
# Dados principais
# ============================================================

df_transacoes = carregar_dataframe_transacoes()

coluna_cluster = detectar_coluna_cluster(df_transacoes)
coluna_anomalia = detectar_coluna_anomalia(df_transacoes)


# ============================================================
# Cabeçalho
# ============================================================

st.title("🛡️ Sentinel 1.0")
st.subheader("Sistema colaborativo de detecção de fraude bancária com React Agent")

st.markdown(
    """
    O Sentinel 1.0 simula um sistema colaborativo entre **Banco**, **Cliente** e **IA preditiva**.
    O agente utiliza **K-Means** para agrupamento comportamental e **Isolation Forest** para detecção de anomalias.
    """
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Controles")

    st.write("Modelos:")
    st.success("Isolation Forest disponível")
    st.success("K-Means disponível")

    st.divider()

    if st.button("Limpar conversa"):
        st.session_state.historico = []
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_transacao_prompt = None
        st.session_state.ultima_transacao_dict = None
        st.rerun()

    if st.button("Limpar relatórios e confirmações"):
        limpar_banco_de_dados()
        st.rerun()

    st.divider()

    st.header("📌 Amostra de contas")

    for i, (conta, perfil) in enumerate(banco_de_dados["clientes"].items()):
        if i >= 5:
            break

        with st.expander(conta):
            st.write(f"Cidade padrão: {perfil.get('cidade_padrao', 'N/A')}")
            st.write(f"Valor médio: R$ {perfil.get('valor_medio', 0):.2f}")
            st.write(f"Tipos comuns: {', '.join(perfil.get('categorias_comuns', []))}")
            st.write(f"Canal comum: {perfil.get('canal_mais_comum', 'N/A')}")
            st.write(f"Ocupação comum: {perfil.get('ocupacao_mais_comum', 'N/A')}")
            st.write(f"Quantidade de transações: {perfil.get('quantidade_transacoes', 0)}")


# ============================================================
# Abas principais
# ============================================================

aba_visao, aba_analise, aba_base, aba_chat, aba_relatorios = st.tabs(
    [
        "📊 Visão Geral",
        "🔎 Analisar Transação",
        "🧾 Base de Transações",
        "💬 Chat com Agente",
        "📁 Relatórios"
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with aba_visao:
    st.header("📊 Visão Geral")

    mostrar_metricas_gerais(df_transacoes)

    st.divider()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Transações por canal")

        if "Channel" in df_transacoes.columns:
            df_canal = df_transacoes["Channel"].value_counts().reset_index()
            df_canal.columns = ["Channel", "Quantidade"]

            fig_canal = px.bar(
                df_canal,
                x="Channel",
                y="Quantidade",
                title="Quantidade de transações por canal",
                text="Quantidade"
            )

            st.plotly_chart(fig_canal, use_container_width=True)
        else:
            st.warning("Coluna Channel não encontrada no dataset.")

    with col_g2:
        st.subheader("Distribuição dos valores")

        if "TransactionAmount" in df_transacoes.columns:
            fig_valores = px.histogram(
                df_transacoes,
                x="TransactionAmount",
                nbins=50,
                title="Distribuição dos valores das transações"
            )

            st.plotly_chart(fig_valores, use_container_width=True)
        else:
            st.warning("Coluna TransactionAmount não encontrada no dataset.")

    st.divider()

    st.header("Visualizações dos clusters")

    if coluna_cluster:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Quantidade por cluster")

            df_cluster_count = df_transacoes[coluna_cluster].value_counts().reset_index()
            df_cluster_count.columns = ["Cluster", "Quantidade"]

            fig_cluster = px.bar(
                df_cluster_count,
                x="Cluster",
                y="Quantidade",
                title="Distribuição das transações por cluster",
                text="Quantidade"
            )

            st.plotly_chart(fig_cluster, use_container_width=True)

        with col_c2:
            st.subheader("Valor por cluster")

            if "TransactionAmount" in df_transacoes.columns:
                fig_box_cluster = px.box(
                    df_transacoes,
                    x=coluna_cluster,
                    y="TransactionAmount",
                    title="Distribuição de valores por cluster"
                )

                st.plotly_chart(fig_box_cluster, use_container_width=True)

        st.subheader("Clusters por valor e saldo")

        if "TransactionAmount" in df_transacoes.columns and "AccountBalance" in df_transacoes.columns:
            fig_scatter_cluster = px.scatter(
                df_transacoes,
                x="TransactionAmount",
                y="AccountBalance",
                color=coluna_cluster,
                hover_data=[
                    "TransactionID",
                    "AccountID",
                    "Channel",
                    "Location",
                    "TransactionType"
                ],
                title="Relação entre valor da transação e saldo da conta por cluster"
            )

            st.plotly_chart(fig_scatter_cluster, use_container_width=True)

    else:
        st.info(
            "Nenhuma coluna de cluster encontrada. "
            "Execute o treinamento para gerar dados/transacoes_clusterizadas.csv."
        )

    st.divider()

    st.header("Visualização de anomalias")

    if coluna_anomalia:
        df_anomalia = df_transacoes[coluna_anomalia].value_counts().reset_index()
        df_anomalia.columns = ["Classe", "Quantidade"]

        fig_anomalia = px.pie(
            df_anomalia,
            names="Classe",
            values="Quantidade",
            title="Distribuição de anomalias pelo Isolation Forest"
        )

        st.plotly_chart(fig_anomalia, use_container_width=True)

    else:
        st.info(
            "Nenhuma coluna de anomalia encontrada na base clusterizada. "
            "O agente ainda consegue usar o modelo Isolation Forest em tempo de análise."
        )


# ============================================================
# ABA 2 — ANALISAR TRANSAÇÃO
# ============================================================

with aba_analise:
    st.header("🔎 Analisar Transação")

    st.markdown(
        """
        Preencha os dados abaixo para simular uma nova transação bancária.
        O agente irá analisar a transação usando os modelos disponíveis e regras interpretáveis de risco.
        """
    )

    contas_disponiveis = list(banco_de_dados["clientes"].keys())
    ocupacoes_disponiveis = sorted(df_transacoes["CustomerOccupation"].dropna().astype(str).unique()) if "CustomerOccupation" in df_transacoes.columns else []
    canais_disponiveis = sorted(df_transacoes["Channel"].dropna().astype(str).unique()) if "Channel" in df_transacoes.columns else ["Online", "ATM", "Branch"]
    locais_disponiveis = sorted(df_transacoes["Location"].dropna().astype(str).unique()) if "Location" in df_transacoes.columns else []
    tipos_disponiveis = sorted(df_transacoes["TransactionType"].dropna().astype(str).unique()) if "TransactionType" in df_transacoes.columns else ["Debit", "Credit"]

    with st.form("form_analise_transacao"):
        col1, col2, col3 = st.columns(3)

        with col1:
            account_id = st.selectbox("Conta / AccountID", contas_disponiveis)
            transaction_amount = st.number_input(
                "Valor da transação",
                min_value=0.0,
                value=100.0,
                step=50.0
            )
            transaction_type = st.selectbox("Tipo de transação", tipos_disponiveis)

        with col2:
            location = st.selectbox("Localização", locais_disponiveis)
            channel = st.selectbox("Canal", canais_disponiveis)
            account_balance = st.number_input(
                "Saldo após transação",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )

        with col3:
            customer_age = st.number_input(
                "Idade do cliente",
                min_value=0,
                max_value=120,
                value=30,
                step=1
            )
            customer_occupation = st.selectbox("Ocupação", ocupacoes_disponiveis)
            transaction_duration = st.number_input(
                "Duração da transação em segundos",
                min_value=0.0,
                value=60.0,
                step=10.0
            )
            login_attempts = st.number_input(
                "Tentativas de login",
                min_value=0,
                value=1,
                step=1
            )

        st.divider()

        col4, col5, col6 = st.columns(3)

        with col4:
            device_id = st.text_input("DeviceID", value="UNKNOWN_DEVICE")

        with col5:
            ip_address = st.text_input("IP Address", value="0.0.0.0")

        with col6:
            merchant_id = st.text_input("MerchantID", value="UNKNOWN_MERCHANT")

        enviar = st.form_submit_button("Analisar transação")

    if enviar:
        prompt = montar_prompt_transacao(
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

        st.session_state.ultima_transacao_prompt = prompt

        st.session_state.ultima_transacao_dict = {
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
            "merchant_id": merchant_id
        }

        resposta = enviar_mensagem_para_agente(prompt)

        st.session_state.ultimo_resultado = resposta
        st.rerun()

    if st.session_state.ultimo_resultado:
        st.divider()
        st.subheader("Resultado da análise")

        exibir_card_resultado(st.session_state.ultimo_resultado)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("Gerar relatório para o banco"):
                if st.session_state.ultima_transacao_prompt:
                    resposta_relatorio = enviar_mensagem_para_agente(
                        "Gere um relatório para o banco com base na última transação analisada. "
                        + st.session_state.ultima_transacao_prompt
                    )

                    st.session_state.ultimo_resultado = resposta_relatorio
                    st.success("Relatório solicitado ao agente.")
                    st.rerun()

        with col_b2:
            if st.button("Solicitar confirmação ao cliente"):
                if st.session_state.ultima_transacao_dict:
                    dados = st.session_state.ultima_transacao_dict

                    mensagem_confirmacao = (
                        f"Solicite confirmação ao cliente da conta {dados['account_id']} "
                        f"sobre a transação de valor {dados['transaction_amount']}, "
                        f"tipo {dados['transaction_type']}, localização {dados['location']}, "
                        f"canal {dados['channel']}."
                    )

                    resposta_confirmacao = enviar_mensagem_para_agente(mensagem_confirmacao)

                    st.session_state.ultimo_resultado = resposta_confirmacao
                    st.success("Confirmação solicitada ao agente.")
                    st.rerun()


# ============================================================
# ABA 3 — BASE DE TRANSAÇÕES
# ============================================================

with aba_base:
    st.header("🧾 Base de Transações")

    st.markdown("Use os filtros abaixo para explorar a base e selecionar uma transação para análise.")

    df_filtrado = df_transacoes.copy()

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        contas_filtro = sorted(df_filtrado["AccountID"].dropna().astype(str).unique()) if "AccountID" in df_filtrado.columns else []
        conta_escolhida = st.selectbox("Filtrar por AccountID", ["Todos"] + contas_filtro)

    with col_f2:
        canais_filtro = sorted(df_filtrado["Channel"].dropna().astype(str).unique()) if "Channel" in df_filtrado.columns else []
        canal_escolhido = st.selectbox("Filtrar por Channel", ["Todos"] + canais_filtro)

    with col_f3:
        locais_filtro = sorted(df_filtrado["Location"].dropna().astype(str).unique()) if "Location" in df_filtrado.columns else []
        local_escolhido = st.selectbox("Filtrar por Location", ["Todos"] + locais_filtro)

    with col_f4:
        tipos_filtro = sorted(df_filtrado["TransactionType"].dropna().astype(str).unique()) if "TransactionType" in df_filtrado.columns else []
        tipo_escolhido = st.selectbox("Filtrar por TransactionType", ["Todos"] + tipos_filtro)

    if conta_escolhida != "Todos":
        df_filtrado = df_filtrado[df_filtrado["AccountID"].astype(str) == conta_escolhida]

    if canal_escolhido != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Channel"].astype(str) == canal_escolhido]

    if local_escolhido != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Location"].astype(str) == local_escolhido]

    if tipo_escolhido != "Todos":
        df_filtrado = df_filtrado[df_filtrado["TransactionType"].astype(str) == tipo_escolhido]

    st.write(f"Transações após filtros: {len(df_filtrado)}")

    colunas_exibicao = [
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
            coluna_cluster,
            coluna_anomalia
        ]
        if coluna and coluna in df_filtrado.columns
    ]

    st.dataframe(
        df_filtrado[colunas_exibicao],
        use_container_width=True,
        height=420
    )

    st.subheader("Analisar transação selecionada")

    if "TransactionID" in df_filtrado.columns and not df_filtrado.empty:
        id_para_analise = st.selectbox(
            "Selecione o TransactionID para análise",
            df_filtrado["TransactionID"].astype(str).tolist()
        )

        transacao_selecionada = df_filtrado[
            df_filtrado["TransactionID"].astype(str) == str(id_para_analise)
        ].iloc[0].to_dict()

        with st.expander("Prévia da transação selecionada"):
            st.json(transacao_selecionada)

        if st.button("Analisar transação selecionada"):
            resposta = enviar_mensagem_para_agente(
                f"Verifique a transação com TransactionID {id_para_analise}."
            )

            st.session_state.ultimo_resultado = resposta
            st.success("Transação enviada ao agente para análise.")
            st.rerun()
    else:
        st.warning("Nenhuma transação disponível para análise com os filtros atuais.")


# ============================================================
# ABA 4 — CHAT COM AGENTE
# ============================================================

with aba_chat:
    st.header("💬 Chat com Agente")

    st.markdown("Sugestões de prompts:")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        if st.button("Consultar banco de dados"):
            resposta = enviar_mensagem_para_agente("Consulte o banco de dados carregado.")
            st.session_state.ultimo_resultado = resposta
            st.rerun()

    with col_p2:
        if st.button("Explicar modelos"):
            resposta = enviar_mensagem_para_agente(
                "Explique como o sistema usa K-Means e Isolation Forest para apoiar a detecção de fraude."
            )
            st.session_state.ultimo_resultado = resposta
            st.rerun()

    with col_p3:
        if st.button("Mostrar relatórios"):
            resposta = enviar_mensagem_para_agente(
                "Mostre os relatórios e confirmações registrados até agora."
            )
            st.session_state.ultimo_resultado = resposta
            st.rerun()

    st.divider()

    for mensagem in st.session_state.historico:
        tipo = mensagem.type

        if tipo == "human":
            with st.chat_message("user"):
                st.write(mensagem.content)

        elif tipo == "ai":
            if mensagem.content:
                with st.chat_message("assistant"):
                    st.write(mensagem.content)

        # Saída bruta das ferramentas fica oculta para deixar a interface mais limpa.
        # Para depuração, descomente o bloco abaixo.
        #
        # elif tipo == "tool":
        #     with st.chat_message("assistant"):
        #         st.info(mensagem.content)

    entrada = st.chat_input("Digite uma mensagem para o agente...")

    if entrada:
        resposta = enviar_mensagem_para_agente(entrada)
        st.session_state.ultimo_resultado = resposta
        st.rerun()


# ============================================================
# ABA 5 — RELATÓRIOS
# ============================================================

with aba_relatorios:
    st.header("📁 Relatórios e Confirmações")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("Relatórios gerados")

        if banco_de_dados["relatorios_fraude"]:
            st.json(banco_de_dados["relatorios_fraude"])
        else:
            st.info("Nenhum relatório registrado ainda.")

    with col_r2:
        st.subheader("Confirmações enviadas")

        if banco_de_dados["confirmacoes_cliente"]:
            st.json(banco_de_dados["confirmacoes_cliente"])
        else:
            st.info("Nenhuma confirmação enviada ainda.")

    st.divider()

    st.subheader("Último resultado do agente")

    if st.session_state.ultimo_resultado:
        exibir_card_resultado(st.session_state.ultimo_resultado)
    else:
        st.info("Nenhuma análise foi realizada ainda.")