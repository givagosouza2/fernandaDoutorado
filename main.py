import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.multivariate.manova import MANOVA

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(
    page_title="MANOVA e LDA - Controle Postural",
    layout="wide"
)

st.title("MANOVA e LDA - Controle Postural")
st.write(
    "Aplicativo para análise multivariada de dados de equilíbrio em duas condições: "
    "**olhos abertos (OE)** e **olhos fechados (CE)**."
)


# ============================================================
# Funções auxiliares
# ============================================================
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
        st.error("Formato não suportado. Use CSV, XLSX ou XLS.")
        return None

    df.columns = make_unique_columns(df.columns)
    return df


def sanitize_column_name(col):
    col = str(col)
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("/", "_")
    col = col.replace("\\", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace(".", "_")
    col = col.replace(",", "_")
    col = col.replace("%", "perc")
    return col


def run_lda(df, features, target_col, title="LDA"):
    st.subheader(title)

    X = df[features].astype(float).to_numpy()
    y = df[target_col].astype(str).to_numpy()

    if len(np.unique(y)) < 2:
        st.warning("A LDA precisa de pelo menos dois grupos.")
        return

    min_n = pd.Series(y).value_counts().min()
    cv_n = min(5, min_n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()

    if cv_n < 2:
        st.warning("Não há participantes suficientes por grupo para validação cruzada.")
        return

    cv = StratifiedKFold(
        n_splits=cv_n,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        lda,
        X_scaled,
        y,
        cv=cv
    )

    st.write(f"**Acurácia média ({cv_n}-fold CV):** {scores.mean():.3f}")
    st.write(f"**Desvio-padrão da acurácia:** {scores.std():.3f}")

    lda.fit(X_scaled, y)
    X_lda = lda.transform(X_scaled)

    y_pred = lda.predict(X_scaled)
    acc_resub = accuracy_score(y, y_pred)

    st.write(f"**Acurácia aparente, sem validação cruzada:** {acc_resub:.3f}")

    # Matriz de confusão
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {label}" for label in labels],
        columns=[f"Predito {label}" for label in labels]
    )

    st.write("**Matriz de confusão aparente:**")
    st.dataframe(cm_df, use_container_width=True)

    # Gráfico LDA
    fig, ax = plt.subplots(figsize=(7, 5))

    if X_lda.shape[1] >= 2:
        for group in labels:
            idx = y == group
            ax.scatter(
                X_lda[idx, 0],
                X_lda[idx, 1],
                label=str(group),
                alpha=0.75
            )

        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title(title)

    else:
        for group in labels:
            idx = y == group
            ax.scatter(
                X_lda[idx, 0],
                np.zeros_like(X_lda[idx, 0]),
                label=str(group),
                alpha=0.75
            )

        ax.set_xlabel("LD1")
        ax.set_yticks([])
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Pesos da LDA
    coef = lda.coef_

    if coef.ndim == 2:
        importance = np.mean(np.abs(coef), axis=0)
    else:
        importance = np.abs(coef)

    importance_df = pd.DataFrame({
        "Variável": features,
        "Importância média absoluta": importance
    }).sort_values(
        by="Importância média absoluta",
        ascending=False
    )

    st.write("**Variáveis com maior contribuição para a discriminação:**")
    st.dataframe(importance_df, use_container_width=True)


# ============================================================
# Upload dos arquivos
# ============================================================
st.header("1. Upload dos arquivos")

col1, col2 = st.columns(2)

with col1:
    file_oe = st.file_uploader(
        "Arquivo de olhos abertos - OE",
        type=["csv", "xlsx", "xls"],
        key="file_oe"
    )

with col2:
    file_ce = st.file_uploader(
        "Arquivo de olhos fechados - CE",
        type=["csv", "xlsx", "xls"],
        key="file_ce"
    )


if file_oe is None or file_ce is None:
    st.info("Envie os dois arquivos para iniciar a análise.")
    st.stop()


df_oe = read_file(file_oe)
df_ce = read_file(file_ce)

if df_oe is None or df_ce is None:
    st.stop()


df_oe["Condition"] = "OE"
df_ce["Condition"] = "CE"

raw_df = pd.concat([df_oe, df_ce], ignore_index=True)
raw_df.columns = make_unique_columns(raw_df.columns)


# ============================================================
# Visualização inicial
# ============================================================
st.header("2. Pré-visualização dos dados")

st.write("Base combinada:")
st.dataframe(raw_df.head(20).astype(str), use_container_width=True)


# ============================================================
# Seleção da coluna de grupo
# ============================================================
st.header("3. Configuração da análise")

possible_group_cols = ["Groups", "Group", "Grupo", "grupo", "GROUP", "groups"]

default_group = None
for col in possible_group_cols:
    if col in raw_df.columns:
        default_group = col
        break

if default_group is None:
    default_group = raw_df.columns[0]

group_col = st.selectbox(
    "Selecione a coluna que identifica os grupos",
    raw_df.columns,
    index=list(raw_df.columns).index(default_group)
)

# Variáveis numéricas
numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 2:
    st.error("Não foram encontradas pelo menos duas variáveis numéricas para análise.")
    st.stop()

selected_vars = st.multiselect(
    "Selecione as variáveis dependentes",
    numeric_cols,
    default=numeric_cols
)

if len(selected_vars) < 2:
    st.warning("Selecione pelo menos duas variáveis numéricas.")
    st.stop()


analysis_df = raw_df[[group_col, "Condition"] + selected_vars].copy()
analysis_df = analysis_df.dropna()

# Converter grupo e condição para texto
analysis_df[group_col] = analysis_df[group_col].astype(str)
analysis_df["Condition"] = analysis_df["Condition"].astype(str)

# Converter variáveis para numérico
for var in selected_vars:
    analysis_df[var] = pd.to_numeric(analysis_df[var], errors="coerce")

analysis_df = analysis_df.dropna()

st.write("Base final usada nas análises:")
st.dataframe(analysis_df.head(30).astype(str), use_container_width=True)

st.write("Número de casos por grupo e condição:")
count_df = analysis_df.groupby([group_col, "Condition"]).size().reset_index(name="n")
st.dataframe(count_df, use_container_width=True)


# ============================================================
# Preparação para MANOVA
# ============================================================
safe_names = {col: sanitize_column_name(col) for col in selected_vars}
analysis_safe = analysis_df.rename(columns=safe_names)

dependent_vars = list(safe_names.values())

formula = " + ".join(dependent_vars) + f" ~ {group_col} * Condition"


# ============================================================
# MANOVA
# ============================================================
st.header("4. MANOVA")

st.write("Modelo:")
st.code(formula)

if st.button("Rodar MANOVA"):

    try:
        maov = MANOVA.from_formula(formula, data=analysis_safe)
        result = maov.mv_test()

        st.subheader("Resultado da MANOVA")
        st.text(result)

    except Exception as e:
        st.error("Erro ao rodar a MANOVA.")
        st.exception(e)


# ============================================================
# LDA
# ============================================================
st.header("5. LDA - Classificação dos grupos")

st.write(
    "A LDA será usada para verificar se as variáveis posturais conseguem "
    "classificar os participantes nos seus respectivos grupos."
)

lda_option = st.radio(
    "Escolha o tipo de LDA",
    [
        "LDA geral - grupos usando OE + CE juntos",
        "LDA separada por condição visual",
        "LDA classificando combinação Grupo_Condição"
    ]
)


if st.button("Rodar LDA"):

    try:

        if lda_option == "LDA geral - grupos usando OE + CE juntos":

            run_lda(
                df=analysis_safe,
                features=dependent_vars,
                target_col=group_col,
                title="LDA geral - Classificação dos grupos"
            )

        elif lda_option == "LDA separada por condição visual":

            for cond in ["OE", "CE"]:

                df_cond = analysis_safe[analysis_safe["Condition"] == cond]

                st.markdown("---")
                st.write(f"## Condição: {cond}")

                if df_cond.empty:
                    st.warning(f"Não há dados para a condição {cond}.")
                    continue

                run_lda(
                    df=df_cond,
                    features=dependent_vars,
                    target_col=group_col,
                    title=f"LDA - Classificação dos grupos na condição {cond}"
                )

        elif lda_option == "LDA classificando combinação Grupo_Condição":

            analysis_safe["Group_Condition"] = (
                analysis_safe[group_col].astype(str)
                + "_"
                + analysis_safe["Condition"].astype(str)
            )

            run_lda(
                df=analysis_safe,
                features=dependent_vars,
                target_col="Group_Condition",
                title="LDA - Classificação Grupo + Condição"
            )

    except Exception as e:
        st.error("Erro ao rodar a LDA.")
        st.exception(e)


# ============================================================
# Orientação interpretativa
# ============================================================
st.header("6. Como interpretar")

st.markdown(
    """
### MANOVA

- **Groups significativo**: os grupos diferem no conjunto das variáveis posturais.
- **Condition significativo**: olhos abertos e olhos fechados diferem globalmente.
- **Groups × Condition significativo**: o efeito da condição visual depende do grupo.

### LDA

- **Acurácia alta** indica que as variáveis posturais classificam bem os grupos.
- Compare a LDA em **OE** e **CE**:
    - Se a acurácia for maior em CE, a ausência de visão aumenta a separação entre os grupos.
    - Se a acurácia for maior em OE, a condição de olhos abertos discrimina melhor os grupos.
- A matriz de confusão mostra quais grupos foram classificados corretamente ou confundidos.
"""
)
