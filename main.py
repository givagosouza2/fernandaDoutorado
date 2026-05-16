import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)


# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="Medidas Repetidas e LDA - Controle Postural",
    layout="wide"
)

st.title("Controle Postural - Medidas Repetidas e Classificação")
st.write(
    """
Aplicativo para análise de dados de equilíbrio em duas condições visuais:
**olhos abertos (OE)** e **olhos fechados (CE)**.

A análise foi organizada em duas partes:

1. **Modelos mistos**, considerando que OE e CE são medidas repetidas do mesmo participante.
2. **Modelos classificatórios**, comparando a capacidade de separar HIV e controles usando:
   OE, CE, OE + CE e Delta CE - OE.
"""
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
    col = col.replace("\ufeff", "")
    col = col.strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("/", "_")
    col = col.replace("\\", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace(".", "_")
    col = col.replace(",", "_")
    col = col.replace("%", "perc")
    col = col.replace(":", "_")
    col = col.replace(";", "_")
    col = col.replace("+", "_")
    col = col.replace("*", "_")
    col = col.replace("=", "_")
    col = col.replace("[", "")
    col = col.replace("]", "")
    col = col.replace("{", "")
    col = col.replace("}", "")

    if col == "":
        col = "coluna"

    if col[0].isdigit():
        col = "var_" + col

    return col


def find_default_column(columns, possible_names):
    for name in possible_names:
        if name in columns:
            return name
    return columns[0]


def prepare_analysis_dataframe(raw_df, id_col, group_col, selected_vars):
    df = raw_df[[id_col, group_col, "Condition"] + selected_vars].copy()

    df[id_col] = df[id_col].astype(str)
    df[group_col] = df[group_col].astype(str)
    df["Condition"] = df["Condition"].astype(str)

    for var in selected_vars:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    df = df.dropna()

    return df


def create_safe_dataframe(df, id_col, group_col, selected_vars):
    rename_map = {}

    rename_map[id_col] = "ID_safe"
    rename_map[group_col] = "Group_safe"
    rename_map["Condition"] = "Condition_safe"

    for var in selected_vars:
        rename_map[var] = sanitize_column_name(var)

    df_safe = df.rename(columns=rename_map)

    safe_vars = [rename_map[var] for var in selected_vars]

    return df_safe, safe_vars


# ============================================================
# Modelo misto para medidas repetidas
# ============================================================

def run_mixed_models(df_safe, safe_vars):
    st.subheader("Modelos mistos para medidas repetidas")

    st.write(
        """
Cada variável postural será analisada separadamente usando o modelo:

`Variável ~ Grupo * Condição + (1 | Participante)`

O termo mais importante para sua pergunta é a interação:

`Grupo × Condição`

Ela indica se a mudança de OE para CE é diferente entre os grupos.
"""
    )

    results = []
    all_pvalues = []
    pvalue_keys = []

    for var in safe_vars:
        formula = f"{var} ~ C(Group_safe) * C(Condition_safe)"

        st.markdown("---")
        st.write(f"### Variável: `{var}`")
        st.code(formula)

        try:
            model = smf.mixedlm(
                formula=formula,
                data=df_safe,
                groups=df_safe["ID_safe"]
            )

            fit = model.fit(reml=False, method="lbfgs")

            st.text(fit.summary())

            params = fit.params
            pvalues = fit.pvalues

            for term in pvalues.index:
                if term == "Intercept":
                    continue

                pval = pvalues[term]

                results.append({
                    "Variável": var,
                    "Termo": term,
                    "Coeficiente": params.get(term, np.nan),
                    "p": pval
                })

                all_pvalues.append(pval)
                pvalue_keys.append((var, term))

        except Exception as e:
            st.error(f"Erro ao ajustar o modelo misto para a variável {var}.")
            st.exception(e)

    if len(results) > 0:
        results_df = pd.DataFrame(results)

        try:
            reject, p_fdr, _, _ = multipletests(
                results_df["p"].values,
                alpha=0.05,
                method="fdr_bh"
            )

            results_df["p_FDR"] = p_fdr
            results_df["Significativo_FDR_0.05"] = reject

        except Exception:
            results_df["p_FDR"] = np.nan
            results_df["Significativo_FDR_0.05"] = False

        st.subheader("Resumo dos modelos mistos")
        st.dataframe(results_df, use_container_width=True)

        interaction_df = results_df[
            results_df["Termo"].str.contains(":", regex=False)
        ].copy()

        if len(interaction_df) > 0:
            st.subheader("Resumo das interações Grupo × Condição")
            st.dataframe(interaction_df, use_container_width=True)

            st.write(
                """
A presença de interações significativas sugere que a diferença entre OE e CE
não é igual nos grupos. Esse é o resultado mais relevante para investigar se
os pacientes com HIV dependem diferentemente da visão para o controle postural.
"""
            )


# ============================================================
# Transformação para formato largo
# ============================================================

def make_wide_dataframe(df_safe, safe_vars):
    wide_df = df_safe[
        ["ID_safe", "Group_safe", "Condition_safe"] + safe_vars
    ].copy()

    wide_df = wide_df.pivot_table(
        index=["ID_safe", "Group_safe"],
        columns="Condition_safe",
        values=safe_vars,
        aggfunc="mean"
    )

    wide_df.columns = [
        f"{var}_{cond}" for var, cond in wide_df.columns
    ]

    wide_df = wide_df.reset_index()
    wide_df = wide_df.dropna()

    return wide_df


def create_delta_dataframe(wide_df, safe_vars):
    delta_df = wide_df[["ID_safe", "Group_safe"]].copy()
    delta_features = []

    for var in safe_vars:
        col_oe = f"{var}_OE"
        col_ce = f"{var}_CE"

        if col_oe in wide_df.columns and col_ce in wide_df.columns:
            delta_col = f"Delta_{var}_CE_menos_OE"
            delta_df[delta_col] = wide_df[col_ce] - wide_df[col_oe]
            delta_features.append(delta_col)

    delta_df = delta_df.dropna()

    return delta_df, delta_features


def create_ratio_dataframe(wide_df, safe_vars):
    ratio_df = wide_df[["ID_safe", "Group_safe"]].copy()
    ratio_features = []

    for var in safe_vars:
        col_oe = f"{var}_OE"
        col_ce = f"{var}_CE"

        if col_oe in wide_df.columns and col_ce in wide_df.columns:
            ratio_col = f"Ratio_{var}_CE_div_OE"

            oe_values = wide_df[col_oe].replace(0, np.nan)
            ratio_df[ratio_col] = wide_df[col_ce] / oe_values

            ratio_features.append(ratio_col)

    ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan)
    ratio_df = ratio_df.dropna()

    return ratio_df, ratio_features


# ============================================================
# LDA com validação cruzada
# ============================================================

def run_lda_model(df, features, target_col, title):
    st.markdown("---")
    st.subheader(title)

    if len(features) < 1:
        st.warning("Não há variáveis suficientes para este modelo.")
        return None

    model_df = df[[target_col] + features].copy()
    model_df = model_df.dropna()

    X = model_df[features].astype(float).to_numpy()
    y = model_df[target_col].astype(str).to_numpy()

    group_counts = pd.Series(y).value_counts()

    st.write("Número de participantes por grupo:")
    st.dataframe(group_counts.reset_index().rename(
        columns={"index": "Grupo", 0: "n"}
    ), use_container_width=True)

    if len(group_counts) < 2:
        st.warning("A LDA precisa de pelo menos dois grupos.")
        return None

    min_n = group_counts.min()

    if min_n < 2:
        st.warning("Não há participantes suficientes por grupo para validação cruzada.")
        return None

    cv_n = min(10, min_n)

    if len(model_df) <= cv_n:
        st.warning("Número de participantes muito pequeno para esta validação cruzada.")
        return None

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto"
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", lda)
    ])

    cv = StratifiedKFold(
        n_splits=cv_n,
        shuffle=True,
        random_state=42
    )

    try:
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        y_pred_cv = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv
        )

        acc = accuracy_score(y, y_pred_cv)
        bal_acc = balanced_accuracy_score(y, y_pred_cv)

        labels = np.unique(y)
        cm = confusion_matrix(y, y_pred_cv, labels=labels)

        cm_df = pd.DataFrame(
            cm,
            index=[f"Real {label}" for label in labels],
            columns=[f"Predito {label}" for label in labels]
        )

        st.write(f"**Acurácia média ({cv_n}-fold CV):** {scores.mean():.3f}")
        st.write(f"**Desvio-padrão da acurácia:** {scores.std():.3f}")
        st.write(f"**Acurácia por predição cruzada:** {acc:.3f}")
        st.write(f"**Balanced accuracy:** {bal_acc:.3f}")

        st.write("Matriz de confusão por validação cruzada:")
        st.dataframe(cm_df, use_container_width=True)

        report = classification_report(
            y,
            y_pred_cv,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.write("Relatório de classificação:")
        st.dataframe(report_df, use_container_width=True)

        # Ajuste final apenas para extrair importância das variáveis
        pipeline.fit(X, y)

        fitted_lda = pipeline.named_steps["lda"]

        if hasattr(fitted_lda, "coef_"):
            coef = fitted_lda.coef_

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

            st.write("Variáveis com maior contribuição para a discriminação:")
            st.dataframe(importance_df, use_container_width=True)

        result = {
            "Modelo": title,
            "N participantes": len(model_df),
            "N variáveis": len(features),
            "CV folds": cv_n,
            "Acurácia média": scores.mean(),
            "DP acurácia": scores.std(),
            "Acurácia CV": acc,
            "Balanced accuracy": bal_acc
        }

        return result

    except Exception as e:
        st.error(f"Erro ao rodar a LDA para o modelo: {title}")
        st.exception(e)
        return None


def compare_classification_models(wide_df, safe_vars):
    st.subheader("Comparação dos modelos classificatórios")

    results = []

    # ----------------------------
    # Modelo OE
    # ----------------------------
    oe_features = [
        col for col in wide_df.columns
        if col.endswith("_OE")
    ]

    if len(oe_features) > 0:
        res_oe = run_lda_model(
            df=wide_df,
            features=oe_features,
            target_col="Group_safe",
            title="Modelo OE - apenas olhos abertos"
        )

        if res_oe is not None:
            results.append(res_oe)

    # ----------------------------
    # Modelo CE
    # ----------------------------
    ce_features = [
        col for col in wide_df.columns
        if col.endswith("_CE")
    ]

    if len(ce_features) > 0:
        res_ce = run_lda_model(
            df=wide_df,
            features=ce_features,
            target_col="Group_safe",
            title="Modelo CE - apenas olhos fechados"
        )

        if res_ce is not None:
            results.append(res_ce)

    # ----------------------------
    # Modelo OE + CE
    # ----------------------------
    both_features = oe_features + ce_features

    if len(both_features) > 0:
        res_both = run_lda_model(
            df=wide_df,
            features=both_features,
            target_col="Group_safe",
            title="Modelo OE + CE - condições combinadas"
        )

        if res_both is not None:
            results.append(res_both)

    # ----------------------------
    # Modelo Delta CE - OE
    # ----------------------------
    delta_df, delta_features = create_delta_dataframe(wide_df, safe_vars)

    if len(delta_features) > 0:
        res_delta = run_lda_model(
            df=delta_df,
            features=delta_features,
            target_col="Group_safe",
            title="Modelo Delta - CE menos OE"
        )

        if res_delta is not None:
            results.append(res_delta)

    # ----------------------------
    # Modelo Razão CE/OE
    # ----------------------------
    ratio_df, ratio_features = create_ratio_dataframe(wide_df, safe_vars)

    if len(ratio_features) > 0:
        res_ratio = run_lda_model(
            df=ratio_df,
            features=ratio_features,
            target_col="Group_safe",
            title="Modelo Razão - CE dividido por OE"
        )

        if res_ratio is not None:
            results.append(res_ratio)

    # ----------------------------
    # Resumo
    # ----------------------------
    if len(results) > 0:
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="Balanced accuracy",
            ascending=False
        )

        st.subheader("Resumo comparativo dos modelos")
        st.dataframe(results_df, use_container_width=True)

        best_model = results_df.iloc[0]["Modelo"]
        best_bal_acc = results_df.iloc[0]["Balanced accuracy"]

        st.success(
            f"Melhor modelo pela balanced accuracy: {best_model} "
            f"({best_bal_acc:.3f})"
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            results_df["Modelo"],
            results_df["Balanced accuracy"]
        )

        ax.set_ylabel("Balanced accuracy")
        ax.set_xlabel("Modelo")
        ax.set_title("Comparação dos modelos classificatórios")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)

    else:
        st.warning("Nenhum modelo classificatório pôde ser ajustado.")


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

st.write("Base combinada em formato longo:")
st.dataframe(raw_df.head(30).astype(str), use_container_width=True)


# ============================================================
# Configuração da análise
# ============================================================

st.header("3. Configuração da análise")

possible_id_cols = [
    "ID", "Id", "id",
    "Participante", "participante",
    "Subject", "subject",
    "Sujeito", "sujeito",
    "Codigo", "Código", "codigo", "código"
]

possible_group_cols = [
    "Groups", "Group", "Grupo", "grupo",
    "GROUP", "groups",
    "Classe", "classe"
]

default_id = find_default_column(raw_df.columns, possible_id_cols)
default_group = find_default_column(raw_df.columns, possible_group_cols)

col_id, col_group = st.columns(2)

with col_id:
    id_col = st.selectbox(
        "Selecione a coluna que identifica o participante",
        raw_df.columns,
        index=list(raw_df.columns).index(default_id)
    )

with col_group:
    group_col = st.selectbox(
        "Selecione a coluna que identifica os grupos",
        raw_df.columns,
        index=list(raw_df.columns).index(default_group)
    )

numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 1:
    st.error("Não foram encontradas variáveis numéricas para análise.")
    st.stop()

selected_vars = st.multiselect(
    "Selecione as variáveis posturais",
    numeric_cols,
    default=numeric_cols
)

if len(selected_vars) < 1:
    st.warning("Selecione pelo menos uma variável numérica.")
    st.stop()


analysis_df = prepare_analysis_dataframe(
    raw_df=raw_df,
    id_col=id_col,
    group_col=group_col,
    selected_vars=selected_vars
)

if analysis_df.empty:
    st.error("A base final ficou vazia após remover valores ausentes.")
    st.stop()

st.subheader("Base final em formato longo")
st.dataframe(analysis_df.head(40).astype(str), use_container_width=True)

st.write("Número de observações por grupo e condição:")
count_df = (
    analysis_df
    .groupby([group_col, "Condition"])
    .size()
    .reset_index(name="n")
)
st.dataframe(count_df, use_container_width=True)

st.write("Número de participantes únicos por grupo:")
participants_df = (
    analysis_df
    .groupby(group_col)[id_col]
    .nunique()
    .reset_index(name="n_participantes")
)
st.dataframe(participants_df, use_container_width=True)


df_safe, safe_vars = create_safe_dataframe(
    df=analysis_df,
    id_col=id_col,
    group_col=group_col,
    selected_vars=selected_vars
)

st.subheader("Nomes seguros usados internamente")
name_map_df = pd.DataFrame({
    "Nome original": selected_vars,
    "Nome usado no modelo": safe_vars
})
st.dataframe(name_map_df, use_container_width=True)


# ============================================================
# Checagem de medidas repetidas
# ============================================================

st.header("4. Checagem das medidas repetidas")

repeat_check = (
    analysis_df
    .groupby([id_col, group_col])["Condition"]
    .nunique()
    .reset_index(name="n_condicoes")
)

n_complete = (repeat_check["n_condicoes"] == 2).sum()
n_incomplete = (repeat_check["n_condicoes"] < 2).sum()

st.write(f"Participantes com OE e CE completos: **{n_complete}**")
st.write(f"Participantes incompletos: **{n_incomplete}**")

if n_incomplete > 0:
    st.warning(
        "Há participantes sem uma das condições visuais. "
        "Eles poderão ser excluídos das análises em formato largo."
    )

    incomplete_df = repeat_check[repeat_check["n_condicoes"] < 2]
    st.dataframe(incomplete_df, use_container_width=True)


# ============================================================
# Modelos mistos
# ============================================================

st.header("5. Inferência estatística com modelos mistos")

st.write(
    """
Esta etapa considera explicitamente que OE e CE são medidas repetidas do
mesmo participante.

Ela responde principalmente:

- existe diferença entre grupos?
- existe diferença entre OE e CE?
- a mudança entre OE e CE é diferente entre HIV e controles?
"""
)

if st.button("Rodar modelos mistos"):
    run_mixed_models(df_safe, safe_vars)


# ============================================================
# Classificação
# ============================================================

st.header("6. Classificação HIV vs controle")

st.write(
    """
Nesta etapa, os dados são transformados para formato largo, com uma única
linha por participante.

Isso evita pseudorreplicação e permite comparar diretamente:

1. Modelo OE
2. Modelo CE
3. Modelo OE + CE
4. Modelo Delta CE - OE
5. Modelo Razão CE/OE
"""
)

wide_df = make_wide_dataframe(df_safe, safe_vars)

st.subheader("Base em formato largo")
st.dataframe(wide_df.head(30).astype(str), use_container_width=True)

st.write(f"Número de participantes na base larga: **{len(wide_df)}**")

if st.button("Rodar comparação dos modelos classificatórios"):
    compare_classification_models(wide_df, safe_vars)


# ============================================================
# Orientação interpretativa
# ============================================================

st.header("7. Como interpretar")

st.markdown(
    """
### Modelos mistos

O resultado mais importante é a interação:

`Grupo × Condição`

Se essa interação for significativa, isso sugere que a mudança de OE para CE
não é igual nos grupos.

Em termos fisiológicos, isso pode indicar que os pacientes com HIV apresentam
uma dependência visual diferente para o controle postural.

---

### Modelos classificatórios

A comparação dos modelos deve ser feita principalmente pela **balanced accuracy**.

- Se o modelo **OE** for melhor, olhos abertos já carregam informação suficiente para separar os grupos.
- Se o modelo **CE** for melhor, a retirada da visão aumenta a separação entre HIV e controles.
- Se o modelo **OE + CE** for melhor, as duas condições carregam informações complementares.
- Se o modelo **Delta CE - OE** for melhor, a principal diferença está na resposta à retirada da visão.
- Se o modelo **Razão CE/OE** for melhor, a informação mais relevante pode estar na mudança relativa entre as condições.

---

### Cuidado importante

A acurácia aparente, calculada no mesmo conjunto usado para treinar o modelo,
não deve ser usada como principal evidência.

Por isso, este app usa validação cruzada e mostra a matriz de confusão baseada
nas predições cruzadas.
"""
)
