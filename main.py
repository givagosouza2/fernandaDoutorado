# app_manova_equilibrio.py
# Streamlit app para MANOVA em dados de controle postural
# Autor: modelo adaptável para dados LONG com grupo, sujeito e condição visual

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.multivariate.manova import MANOVA
from scipy import stats

st.set_page_config(page_title="MANOVA - Controle Postural", layout="wide")

st.title("MANOVA para avaliação multidimensional do controle postural")
st.markdown(
    """
Este aplicativo realiza MANOVA em dados de equilíbrio postural, especialmente para comparar grupos
por múltiplas variáveis estabilométricas. Ele foi pensado para dados no formato **LONG**.

Estrutura esperada:

| sujeito | grupo | condicao | variável_1 | variável_2 | variável_3 |
|---|---|---|---|---|---|
| P01 | Controle | OA | ... | ... | ... |
| P01 | Controle | OF | ... | ... | ... |
| P02 | HIV | OA | ... | ... | ... |
| P02 | HIV | OF | ... | ... | ... |
    """
)

# =====================================================
# Funções auxiliares
# =====================================================

def clean_colnames(cols):
    new_cols = []
    for c in cols:
        c = str(c).strip()
        c = re.sub(r"\s+", "_", c)
        c = c.replace("/", "_").replace("-", "_")
        c = re.sub(r"[^0-9a-zA-Z_À-ÿ]", "", c)
        new_cols.append(c)
    return new_cols


def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";", decimal=",")
    elif name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Escolha a planilha", xls.sheet_names)
        return pd.read_excel(uploaded_file, sheet_name=sheet)
    else:
        st.error("Formato não reconhecido. Use CSV, XLS ou XLSX.")
        st.stop()


def manova_from_formula(df, y_vars, factor):
    safe_df = df.copy()
    formula = " + ".join(y_vars) + f" ~ C({factor})"
    model = MANOVA.from_formula(formula, data=safe_df)
    return model.mv_test()


def result_to_text(result):
    buffer = io.StringIO()
    print(result, file=buffer)
    return buffer.getvalue()


def univariate_tests(df, y_vars, group_col):
    rows = []
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return pd.DataFrame()

    g1, g2 = groups[0], groups[1]
    for var in y_vars:
        a = pd.to_numeric(df.loc[df[group_col] == g1, var], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[group_col] == g2, var], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue

        t, p = stats.ttest_ind(a, b, equal_var=False)
        mean_diff = a.mean() - b.mean()
        pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

        rows.append({
            "Variável": var,
            f"Média_{g1}": a.mean(),
            f"Média_{g2}": b.mean(),
            "Diferença_médias": mean_diff,
            "t_Welch": t,
            "p": p,
            "Cohen_d": cohen_d,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_Bonferroni"] = np.minimum(out["p"] * len(out), 1)
    return out


# =====================================================
# Upload
# =====================================================

uploaded = st.sidebar.file_uploader("Carregue o arquivo CSV/XLSX", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Carregue um arquivo para iniciar.")
    st.stop()

raw_df = read_file(uploaded)
raw_df.columns = clean_colnames(raw_df.columns)

st.subheader("Pré-visualização dos dados")
st.dataframe(raw_df.head(20), use_container_width=True)

# =====================================================
# Configuração das colunas
# =====================================================

st.sidebar.header("Configuração")
cols = list(raw_df.columns)

subject_col = st.sidebar.selectbox("Coluna do sujeito/participante", cols)
group_col = st.sidebar.selectbox("Coluna do grupo", cols)
condition_col = st.sidebar.selectbox("Coluna da condição visual", ["Nenhuma"] + cols)

numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

# tentar converter colunas aparentemente numéricas
for c in raw_df.columns:
    if c not in numeric_cols:
        converted = pd.to_numeric(raw_df[c].astype(str).str.replace(",", "."), errors="coerce")
        if converted.notna().sum() >= max(3, int(0.5 * len(raw_df))):
            raw_df[c] = converted

numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
exclude_defaults = [subject_col, group_col]
if condition_col != "Nenhuma":
    exclude_defaults.append(condition_col)

candidate_vars = [c for c in numeric_cols if c not in exclude_defaults]

dependent_vars = st.sidebar.multiselect(
    "Variáveis dependentes para MANOVA",
    candidate_vars,
    default=candidate_vars[: min(8, len(candidate_vars))]
)

st.sidebar.markdown("---")
run = st.sidebar.button("Rodar MANOVA")

if not dependent_vars:
    st.warning("Selecione pelo menos duas variáveis dependentes numéricas.")
    st.stop()

if len(dependent_vars) < 2:
    st.warning("MANOVA exige pelo menos duas variáveis dependentes.")
    st.stop()

# =====================================================
# Limpeza básica
# =====================================================

df = raw_df[[subject_col, group_col] + ([] if condition_col == "Nenhuma" else [condition_col]) + dependent_vars].copy()
df = df.dropna()
df[group_col] = df[group_col].astype(str)
if condition_col != "Nenhuma":
    df[condition_col] = df[condition_col].astype(str)

st.subheader("Resumo da amostra")
col1, col2, col3 = st.columns(3)
col1.metric("Linhas válidas", len(df))
col2.metric("Participantes únicos", df[subject_col].nunique())
col3.metric("Grupos", df[group_col].nunique())

st.write("Distribuição por grupo:")
st.dataframe(df[group_col].value_counts().rename_axis("grupo").reset_index(name="n"), use_container_width=True)

if condition_col != "Nenhuma":
    st.write("Distribuição por grupo e condição:")
    st.dataframe(pd.crosstab(df[group_col], df[condition_col]), use_container_width=True)

# =====================================================
# MANOVA
# =====================================================

if run:
    st.header("Resultados MANOVA")

    if condition_col == "Nenhuma":
        st.markdown("### MANOVA entre grupos")
        st.markdown(
            "Testa se o conjunto multivariado das variáveis dependentes difere entre os grupos."
        )
        try:
            res = manova_from_formula(df, dependent_vars, group_col)
            text = result_to_text(res)
            st.code(text, language="text")
        except Exception as e:
            st.error(f"Erro ao rodar MANOVA: {e}")

        st.markdown("### Testes univariados exploratórios")
        uni = univariate_tests(df, dependent_vars, group_col)
        if not uni.empty:
            st.dataframe(uni, use_container_width=True)
            st.download_button(
                "Baixar testes univariados CSV",
                data=uni.to_csv(index=False).encode("utf-8"),
                file_name="testes_univariados_manova.csv",
                mime="text/csv"
            )
        else:
            st.info("Testes univariados disponíveis apenas quando há exatamente dois grupos.")

    else:
        st.markdown("### 1. MANOVA por condição visual")
        conditions = sorted(df[condition_col].dropna().unique())

        for cond in conditions:
            st.markdown(f"#### Condição: {cond}")
            df_cond = df[df[condition_col] == cond].copy()
            if df_cond[group_col].nunique() < 2:
                st.warning(f"A condição {cond} possui menos de dois grupos.")
                continue
            try:
                res = manova_from_formula(df_cond, dependent_vars, group_col)
                st.code(result_to_text(res), language="text")
            except Exception as e:
                st.error(f"Erro na condição {cond}: {e}")

        st.markdown("### 2. MANOVA das diferenças entre condições")
        st.markdown(
            "Esta análise transforma os dados LONG em WIDE e calcula a diferença entre duas condições. "
            "Ela é útil para testar se a resposta à retirada da visão difere entre os grupos."
        )

        if len(conditions) != 2:
            st.warning(
                "A análise de DELTA exige exatamente duas condições. "
                f"Foram encontradas {len(conditions)} condições: {conditions}"
            )
        else:
            cond_a = st.selectbox("Condição de referência", conditions, index=0)
            cond_b = st.selectbox("Condição de comparação", conditions, index=1)

            wide = df.pivot_table(
                index=[subject_col, group_col],
                columns=condition_col,
                values=dependent_vars,
                aggfunc="mean"
            )
            wide.columns = [f"{v}_{c}" for v, c in wide.columns]
            wide = wide.reset_index()

            delta_vars = []
            for var in dependent_vars:
                col_a = f"{var}_{cond_a}"
                col_b = f"{var}_{cond_b}"
                if col_a in wide.columns and col_b in wide.columns:
                    delta_name = f"DELTA_{var}_{cond_b}_menos_{cond_a}"
                    wide[delta_name] = wide[col_b] - wide[col_a]
                    delta_vars.append(delta_name)

            wide_delta = wide[[subject_col, group_col] + delta_vars].dropna()

            st.write("Dados de DELTA:")
            st.dataframe(wide_delta.head(20), use_container_width=True)

            if len(delta_vars) < 2:
                st.warning("Não há pelo menos duas variáveis DELTA para MANOVA.")
            elif wide_delta[group_col].nunique() < 2:
                st.warning("A análise de DELTA exige pelo menos dois grupos.")
            else:
                try:
                    res_delta = manova_from_formula(wide_delta, delta_vars, group_col)
                    st.code(result_to_text(res_delta), language="text")
                except Exception as e:
                    st.error(f"Erro ao rodar MANOVA dos deltas: {e}")

                st.markdown("### Testes univariados exploratórios dos deltas")
                uni_delta = univariate_tests(wide_delta, delta_vars, group_col)
                if not uni_delta.empty:
                    st.dataframe(uni_delta, use_container_width=True)
                    st.download_button(
                        "Baixar deltas + testes univariados CSV",
                        data=uni_delta.to_csv(index=False).encode("utf-8"),
                        file_name="testes_univariados_deltas.csv",
                        mime="text/csv"
                    )

                st.download_button(
                    "Baixar tabela WIDE com deltas",
                    data=wide_delta.to_csv(index=False).encode("utf-8"),
                    file_name="dados_delta_manova.csv",
                    mime="text/csv"
                )

    st.header("Como interpretar")
    st.markdown(
        """
- **Pillai's trace**: estatística multivariada mais robusta; é uma boa opção para reportar.
- **Wilks' lambda**: quanto menor, maior a separação multivariada entre grupos.
- **p < 0,05**: indica diferença multivariada significativa.
- A MANOVA mostra se existe diferença no **perfil global** das variáveis; depois disso, os testes univariados ajudam a identificar quais variáveis mais contribuem.

Para dados com olhos abertos e fechados, a MANOVA dos **deltas** testa se a mudança entre condições é diferente entre grupos, ou seja, se há indício de resposta diferente à privação visual.
        """
    )
