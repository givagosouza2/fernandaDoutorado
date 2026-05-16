import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="Pipeline ML - HIV vs Controle",
    layout="wide"
)

st.title("Pipeline de Aprendizado de Máquina - Controle Postural")

st.write(
    """
Este aplicativo compara diferentes algoritmos de aprendizado de máquina para
classificar participantes em grupos, usando variáveis posturais obtidas em
condições de olhos abertos e olhos fechados.

A normalização é feita **dentro da validação cruzada**, evitando vazamento de
informação entre treino e teste.
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


def safe_dataframe(df, n_rows=None):
    df_show = df.copy()
    df_show.columns = make_unique_columns(df_show.columns)

    if n_rows is not None:
        df_show = df_show.head(n_rows)

    st.dataframe(df_show, use_container_width=True)


def read_database(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Formato não suportado. Use CSV, XLSX ou XLS.")
        return None

    df.columns = make_unique_columns(df.columns)
    return df


def sanitize_text(x):
    return str(x).strip().lower()


def normalize_condition_label(x):
    x = sanitize_text(x)

    if x in ["oe", "open", "open eyes", "olhos abertos", "abertos", "eye open", "eyes open"]:
        return "OE"

    if x in ["ce", "closed", "closed eyes", "olhos fechados", "fechados", "eye closed", "eyes closed"]:
        return "CE"

    return str(x).strip()


def identify_default_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_map = {str(col).lower().strip(): col for col in columns}

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    return columns[0]


def create_wide_dataframe(df, id_col, group_col, condition_col, feature_cols):
    work_df = df[[id_col, group_col, condition_col] + feature_cols].copy()

    work_df[id_col] = work_df[id_col].astype(str)
    work_df[group_col] = work_df[group_col].astype(str)
    work_df[condition_col] = work_df[condition_col].apply(normalize_condition_label)

    for col in feature_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    work_df = work_df.dropna()

    wide_df = work_df.pivot_table(
        index=[id_col, group_col],
        columns=condition_col,
        values=feature_cols,
        aggfunc="mean"
    )

    wide_df.columns = [
        f"{feature}__{condition}" for feature, condition in wide_df.columns
    ]

    wide_df = wide_df.reset_index()
    wide_df.columns = make_unique_columns(wide_df.columns)

    return wide_df


def build_feature_sets(wide_df, original_features):
    feature_sets = {}

    oe_features = []
    ce_features = []

    for feature in original_features:
        oe_col = f"{feature}__OE"
        ce_col = f"{feature}__CE"

        if oe_col in wide_df.columns:
            oe_features.append(oe_col)

        if ce_col in wide_df.columns:
            ce_features.append(ce_col)

    if len(oe_features) > 0:
        feature_sets["OE"] = oe_features

    if len(ce_features) > 0:
        feature_sets["CE"] = ce_features

    if len(oe_features) > 0 and len(ce_features) > 0:
        feature_sets["OE + CE"] = oe_features + ce_features

    # Delta CE - OE
    delta_df = wide_df.copy()
    delta_features = []

    for feature in original_features:
        oe_col = f"{feature}__OE"
        ce_col = f"{feature}__CE"

        if oe_col in wide_df.columns and ce_col in wide_df.columns:
            delta_col = f"Delta__{feature}"
            delta_df[delta_col] = wide_df[ce_col] - wide_df[oe_col]
            delta_features.append(delta_col)

    if len(delta_features) > 0:
        feature_sets["Delta CE - OE"] = delta_features

    # Razão CE/OE
    ratio_df = delta_df.copy()
    ratio_features = []

    for feature in original_features:
        oe_col = f"{feature}__OE"
        ce_col = f"{feature}__CE"

        if oe_col in wide_df.columns and ce_col in wide_df.columns:
            ratio_col = f"Ratio__{feature}"
            denominator = wide_df[oe_col].replace(0, np.nan)
            ratio_df[ratio_col] = wide_df[ce_col] / denominator
            ratio_features.append(ratio_col)

    ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan)

    if len(ratio_features) > 0:
        feature_sets["Razão CE/OE"] = ratio_features

    # OE + CE + Delta
    if len(oe_features) > 0 and len(ce_features) > 0 and len(delta_features) > 0:
        feature_sets["OE + CE + Delta"] = oe_features + ce_features + delta_features

    final_df = ratio_df.copy()

    return final_df, feature_sets


def get_algorithms(random_state=42):
    algorithms = {
        "LDA": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        ),

        "Regressão logística": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state
        ),

        "SVM linear": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),

        "SVM RBF": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB()
    }

    return algorithms


def compute_binary_metrics(y_true, y_pred, y_score, positive_label):
    labels = np.unique(y_true)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    negative_labels = [label for label in labels if label != positive_label]

    if len(negative_labels) == 1:
        negative_label = negative_labels[0]
        specificity = recall_score(y_true, y_pred, pos_label=negative_label, zero_division=0)
    else:
        specificity = np.nan

    auc = np.nan

    if y_score is not None and len(labels) == 2:
        y_true_binary = np.array([1 if y == positive_label else 0 for y in y_true])

        try:
            auc = roc_auc_score(y_true_binary, y_score)
        except Exception:
            auc = np.nan

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc
    }


def bootstrap_ci(values, n_bootstrap=1000, ci=95, random_state=42):
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100 - ci) / 2

    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)

    return lower, upper


def evaluate_model_repeated_cv(
    X,
    y,
    model,
    positive_label,
    n_splits=5,
    n_repeats=50,
    n_bootstrap=1000,
    random_state=42
):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    fold_metrics = []

    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        y_score = None

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_test)
                classes = pipeline.named_steps["model"].classes_

                if positive_label in classes:
                    pos_index = list(classes).index(positive_label)
                    y_score = proba[:, pos_index]
            except Exception:
                y_score = None

        elif hasattr(pipeline.named_steps["model"], "decision_function"):
            try:
                score = pipeline.decision_function(X_test)
                y_score = score
            except Exception:
                y_score = None

        metrics = compute_binary_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            positive_label=positive_label
        )

        fold_metrics.append(metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if y_score is not None:
            all_y_score.extend(y_score)
        else:
            all_y_score.extend([np.nan] * len(y_test))

    metrics_df = pd.DataFrame(fold_metrics)

    summary = {}

    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]:
        values = metrics_df[metric].values

        mean_value = np.nanmean(values)
        sd_value = np.nanstd(values)

        ci_low, ci_high = bootstrap_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci=95,
            random_state=random_state
        )

        summary[f"{metric}_mean"] = mean_value
        summary[f"{metric}_sd"] = sd_value
        summary[f"{metric}_ci95_low"] = ci_low
        summary[f"{metric}_ci95_high"] = ci_high

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    labels = np.unique(y)

    cm = confusion_matrix(
        all_y_true,
        all_y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {label}" for label in labels],
        columns=[f"Predito {label}" for label in labels]
    )

    return summary, metrics_df, cm_df


def fit_final_model_and_importance(X, y, feature_names, model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X, y)

    fitted_model = pipeline.named_steps["model"]

    importance_df = None

    if hasattr(fitted_model, "coef_"):
        coef = fitted_model.coef_

        if coef.ndim == 2:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)

        importance_df = pd.DataFrame({
            "Variável": feature_names,
            "Importância": importance
        }).sort_values("Importância", ascending=False)

    elif hasattr(fitted_model, "feature_importances_"):
        importance = fitted_model.feature_importances_

        importance_df = pd.DataFrame({
            "Variável": feature_names,
            "Importância": importance
        }).sort_values("Importância", ascending=False)

    return pipeline, importance_df


# ============================================================
# Upload da base
# ============================================================

st.header("1. Carregar base de dados")

uploaded_file = st.file_uploader(
    "Envie sua base de dados em CSV, XLSX ou XLS",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Envie a base de dados para iniciar.")
    st.stop()

df = read_database(uploaded_file)

if df is None:
    st.stop()

st.subheader("Pré-visualização da base")
safe_dataframe(df, n_rows=20)

st.write(f"Número de linhas: **{df.shape[0]}**")
st.write(f"Número de colunas: **{df.shape[1]}**")


# ============================================================
# Configuração das colunas
# ============================================================

st.header("2. Configuração das colunas")

columns = df.columns.tolist()

default_group = identify_default_column(
    columns,
    ["Groups", "Group", "Grupo", "grupo", "Classe", "class"]
)

default_condition = identify_default_column(
    columns,
    ["Condition", "condition", "Condição", "Condicao", "condição", "condicao"]
)

default_id = identify_default_column(
    columns,
    ["Participant", "Participante", "ID", "Id", "id", "Subject", "Sujeito"]
)

col1, col2, col3 = st.columns(3)

with col1:
    id_col = st.selectbox(
        "Coluna do participante",
        columns,
        index=columns.index(default_id)
    )

with col2:
    group_col = st.selectbox(
        "Coluna do grupo",
        columns,
        index=columns.index(default_group)
    )

with col3:
    condition_col = st.selectbox(
        "Coluna da condição visual",
        columns,
        index=columns.index(default_condition)
    )

if len(set([id_col, group_col, condition_col])) < 3:
    st.error("As colunas de participante, grupo e condição devem ser diferentes.")
    st.stop()


# ============================================================
# Seleção das variáveis
# ============================================================

st.header("3. Seleção das variáveis posturais")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

exclude_cols = [id_col, group_col, condition_col]

available_features = [
    col for col in numeric_cols
    if col not in exclude_cols
]

if len(available_features) == 0:
    st.error("Nenhuma variável numérica disponível para análise.")
    st.stop()

selected_features = st.multiselect(
    "Selecione as variáveis posturais",
    available_features,
    default=available_features
)

if len(selected_features) == 0:
    st.warning("Selecione pelo menos uma variável.")
    st.stop()

st.write("Variáveis selecionadas:")
st.write(selected_features)


# ============================================================
# Configuração da validação
# ============================================================

st.header("4. Configuração da validação")

col_cv1, col_cv2, col_cv3 = st.columns(3)

with col_cv1:
    n_splits = st.number_input(
        "Número de folds",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

with col_cv2:
    n_repeats = st.number_input(
        "Número de repetições",
        min_value=1,
        max_value=200,
        value=50,
        step=1
    )

with col_cv3:
    n_bootstrap = st.number_input(
        "Número de reamostragens bootstrap",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

random_state = st.number_input(
    "Random state",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)


# ============================================================
# Preparação dos dados
# ============================================================

st.header("5. Preparação dos dados")

df_work = df.copy()
df_work[condition_col] = df_work[condition_col].apply(normalize_condition_label)

st.write("Condições visuais identificadas:")
condition_counts = df_work[condition_col].value_counts().reset_index()
condition_counts.columns = ["Condição", "n"]
safe_dataframe(condition_counts)

st.write("Grupos identificados:")
group_counts = df_work[group_col].value_counts().reset_index()
group_counts.columns = ["Grupo", "n"]
safe_dataframe(group_counts)

if df_work[group_col].nunique() != 2:
    st.error(
        "Este pipeline foi construído para classificação binária. "
        "A coluna de grupo precisa ter exatamente dois grupos."
    )
    st.stop()

groups = sorted(df_work[group_col].astype(str).unique().tolist())

positive_label = st.selectbox(
    "Escolha o grupo considerado positivo para sensibilidade/AUC",
    groups,
    index=0
)

wide_df = create_wide_dataframe(
    df=df_work,
    id_col=id_col,
    group_col=group_col,
    condition_col=condition_col,
    feature_cols=selected_features
)

st.subheader("Base em formato largo")
safe_dataframe(wide_df, n_rows=20)

st.write(f"Número de participantes na base larga: **{wide_df.shape[0]}**")

wide_complete_df, feature_sets = build_feature_sets(
    wide_df=wide_df,
    original_features=selected_features
)

st.subheader("Conjuntos de variáveis criados")

feature_set_summary = pd.DataFrame({
    "Conjunto": list(feature_sets.keys()),
    "Número de variáveis": [len(v) for v in feature_sets.values()]
})

safe_dataframe(feature_set_summary)

if len(feature_sets) == 0:
    st.error("Nenhum conjunto de variáveis pôde ser criado.")
    st.stop()


# ============================================================
# Rodar análise
# ============================================================

st.header("6. Rodar comparação dos algoritmos")

st.write(
    """
Para cada conjunto de variáveis, todos os algoritmos serão avaliados usando:

- normalização dentro do pipeline;
- validação cruzada estratificada repetida;
- intervalo de confiança bootstrap das métricas.
"""
)

run_analysis = st.button("Rodar pipeline completo")

if run_analysis:

    algorithms = get_algorithms(random_state=int(random_state))

    all_results = []
    detailed_metrics = {}
    confusion_matrices = {}
    importances = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_runs = len(feature_sets) * len(algorithms)
    current_run = 0

    for feature_set_name, features in feature_sets.items():

        model_df = wide_complete_df[[group_col] + features].copy()
        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df = model_df.dropna()

        if model_df.shape[0] < 4:
            st.warning(f"Conjunto {feature_set_name}: poucos participantes após remoção de ausentes.")
            continue

        X = model_df[features].astype(float).to_numpy()
        y = model_df[group_col].astype(str).to_numpy()

        group_count_model = pd.Series(y).value_counts()

        if group_count_model.min() < n_splits:
            st.warning(
                f"Conjunto {feature_set_name}: o menor grupo tem {group_count_model.min()} participantes. "
                f"Reduza o número de folds."
            )
            continue

        for algorithm_name, algorithm in algorithms.items():

            current_run += 1
            progress_bar.progress(current_run / total_runs)

            status_text.write(
                f"Rodando: {feature_set_name} | {algorithm_name}"
            )

            try:
                summary, metrics_df, cm_df = evaluate_model_repeated_cv(
                    X=X,
                    y=y,
                    model=algorithm,
                    positive_label=positive_label,
                    n_splits=int(n_splits),
                    n_repeats=int(n_repeats),
                    n_bootstrap=int(n_bootstrap),
                    random_state=int(random_state)
                )

                result_row = {
                    "Conjunto": feature_set_name,
                    "Algoritmo": algorithm_name,
                    "N participantes": model_df.shape[0],
                    "N variáveis": len(features),
                    "Balanced accuracy média": summary["balanced_accuracy_mean"],
                    "Balanced accuracy DP": summary["balanced_accuracy_sd"],
                    "Balanced accuracy IC95% inferior": summary["balanced_accuracy_ci95_low"],
                    "Balanced accuracy IC95% superior": summary["balanced_accuracy_ci95_high"],
                    "Acurácia média": summary["accuracy_mean"],
                    "Sensibilidade média": summary["sensitivity_mean"],
                    "Especificidade média": summary["specificity_mean"],
                    "F1 médio": summary["f1_mean"],
                    "AUC média": summary["auc_mean"],
                    "AUC IC95% inferior": summary["auc_ci95_low"],
                    "AUC IC95% superior": summary["auc_ci95_high"]
                }

                all_results.append(result_row)

                key = f"{feature_set_name} | {algorithm_name}"
                detailed_metrics[key] = metrics_df
                confusion_matrices[key] = cm_df

                final_pipeline, importance_df = fit_final_model_and_importance(
                    X=X,
                    y=y,
                    feature_names=features,
                    model=algorithm
                )

                if importance_df is not None:
                    importances[key] = importance_df

            except Exception as e:
                st.error(f"Erro em {feature_set_name} | {algorithm_name}")
                st.exception(e)

    progress_bar.progress(1.0)
    status_text.write("Análise finalizada.")

    if len(all_results) == 0:
        st.error("Nenhum modelo foi avaliado com sucesso.")
        st.stop()

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="Balanced accuracy média",
        ascending=False
    )

    st.header("7. Ranking dos modelos")

    safe_dataframe(results_df)

    best_row = results_df.iloc[0]

    st.success(
        f"Melhor modelo: {best_row['Conjunto']} | {best_row['Algoritmo']} "
        f"com balanced accuracy média = {best_row['Balanced accuracy média']:.3f}"
    )

    # ========================================================
    # Gráfico dos melhores modelos
    # ========================================================

    st.subheader("Top 15 modelos por balanced accuracy")

    top_df = results_df.head(15).copy()
    top_df["Modelo"] = top_df["Conjunto"] + " | " + top_df["Algoritmo"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        top_df["Modelo"][::-1],
        top_df["Balanced accuracy média"][::-1]
    )

    ax.set_xlabel("Balanced accuracy média")
    ax.set_title("Top 15 modelos")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)

    # ========================================================
    # Detalhes do melhor modelo
    # ========================================================

    st.header("8. Detalhes do melhor modelo")

    best_key = f"{best_row['Conjunto']} | {best_row['Algoritmo']}"

    st.subheader("Métricas por fold/repetição do melhor modelo")
    safe_dataframe(detailed_metrics[best_key])

    st.subheader("Matriz de confusão acumulada do melhor modelo")
    safe_dataframe(confusion_matrices[best_key])

    if best_key in importances:
        st.subheader("Importância das variáveis no melhor modelo")
        safe_dataframe(importances[best_key])

        top_imp = importances[best_key].head(20).copy()

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        ax2.barh(
            top_imp["Variável"][::-1],
            top_imp["Importância"][::-1]
        )

        ax2.set_xlabel("Importância")
        ax2.set_title("Principais variáveis do melhor modelo")
        ax2.grid(axis="x", alpha=0.3)

        st.pyplot(fig2)

    else:
        st.info(
            "Este algoritmo não possui coeficientes ou importâncias diretas "
            "de variáveis disponíveis."
        )

    # ========================================================
    # Download dos resultados
    # ========================================================

    st.header("9. Exportar resultados")

    csv_results = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Baixar tabela de resultados em CSV",
        data=csv_results,
        file_name="resultados_pipeline_ml.csv",
        mime="text/csv"
    )


# ============================================================
# Interpretação
# ============================================================

st.header("10. Como interpretar")

st.markdown(
    """
### O que este pipeline responde?

Ele responde principalmente:

> Qual combinação de condição visual, transformação de variáveis e algoritmo
> separa melhor os grupos?

---

### Interpretação dos conjuntos de variáveis

- **OE**: testa se os grupos já se separam com olhos abertos.
- **CE**: testa se a retirada da visão aumenta a separação.
- **OE + CE**: testa se as duas condições trazem informação complementar.
- **Delta CE - OE**: testa se a resposta à retirada da visão separa os grupos.
- **Razão CE/OE**: testa se a mudança proporcional separa os grupos.
- **OE + CE + Delta**: combina valores absolutos e mudança funcional.

---

### Métrica principal

A métrica principal recomendada é a **balanced accuracy**, especialmente se os
grupos tiverem tamanhos diferentes.

---

### Normalização

A normalização é feita dentro do `Pipeline`, ou seja:

1. o scaler é ajustado apenas no treino;
2. o teste é transformado usando os parâmetros do treino;
3. o modelo nunca vê informações do teste durante o treinamento.

Isso evita vazamento de informação.

---

### Bootstrap

O bootstrap é usado para estimar a incerteza das métricas de desempenho,
gerando intervalos de confiança de 95%.
"""
)
