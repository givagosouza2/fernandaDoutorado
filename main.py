import streamlit as st
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

st.set_page_config(page_title="MANOVA - Equilíbrio", layout="wide")

st.title("MANOVA - Controle postural")
st.write("Envie separadamente as bases de olhos abertos e olhos fechados.")

# -----------------------------
# Funções auxiliares
# -----------------------------
def make_unique_columns(columns):
    seen = {}
    new_columns = []

    for col in columns:
        col = str(col).replace("\ufeff", "").strip()

        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    return new_columns


def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Formato não suportado. Use CSV ou Excel.")
        return None

    df.columns = make_unique_columns(df.columns)
    return df


# -----------------------------
# Upload dos dois arquivos
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    file_oe = st.file_uploader(
        "Arquivo olhos abertos - OE",
        type=["csv", "xlsx", "xls"],
        key="oe"
    )

with col2:
    file_ce = st.file_uploader(
        "Arquivo olhos fechados - CE",
        type=["csv", "xlsx", "xls"],
        key="ce"
    )


if file_oe is not None and file_ce is not None:

    df_oe = read_file(file_oe)
    df_ce = read_file(file_ce)

    if df_oe is not None and df_ce is not None:

        df_oe["Condition"] = "OE"
        df_ce["Condition"] = "CE"

        df = pd.concat([df_oe, df_ce], ignore_index=True)

        st.subheader("Pré-visualização da base combinada")
        st.dataframe(df.head(20), use_container_width=True)

        # Detectar coluna de grupo
        possible_group_cols = ["Groups", "Group", "Grupo", "grupo"]

        group_col = None
        for col in possible_group_cols:
            if col in df.columns:
                group_col = col
                break

        if group_col is None:
            st.error("Não encontrei a coluna de grupos. Verifique se existe uma coluna chamada 'Groups'.")
            st.stop()

        st.success(f"Coluna de grupo detectada: {group_col}")

        # Variáveis numéricas disponíveis
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        st.subheader("Seleção das variáveis dependentes")

        selected_vars = st.multiselect(
            "Escolha as variáveis para incluir na MANOVA",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_vars) < 2:
            st.warning("Selecione pelo menos duas variáveis numéricas para rodar a MANOVA.")
            st.stop()

        # Remover linhas com valores ausentes
        analysis_df = df[[group_col, "Condition"] + selected_vars].dropna()

        st.subheader("Base usada na análise")
        st.dataframe(analysis_df, use_container_width=True)

        # Renomear variáveis para nomes seguros na fórmula
        safe_names = {
            col: (
                col.replace(" ", "_")
                   .replace("-", "_")
                   .replace("/", "_")
                   .replace("(", "")
                   .replace(")", "")
                   .replace(".", "_")
            )
            for col in selected_vars
        }

        analysis_df = analysis_df.rename(columns=safe_names)

        dependent_vars = list(safe_names.values())

        formula = " + ".join(dependent_vars) + f" ~ {group_col} * Condition"

        st.subheader("Modelo MANOVA")
        st.code(formula)

        if st.button("Rodar MANOVA"):

            try:
                maov = MANOVA.from_formula(formula, data=analysis_df)
                result = maov.mv_test()

                st.subheader("Resultado da MANOVA")
                st.text(result)

            except Exception as e:
                st.error("Erro ao rodar a MANOVA.")
                st.exception(e)

else:
    st.info("Envie os dois arquivos para iniciar a análise.")
