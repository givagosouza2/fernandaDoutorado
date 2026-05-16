import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    page_title="Pipeline ML Modular - HIV vs Controle",
    layout="wide"
)

st.title("Pipeline Modular de Aprendizado de Máquina")
st.write(
    """
Este aplicativo executa análises de classificação em blocos menores.
Você pode escolher um algoritmo e um ou mais conjuntos de variáveis por vez,
evitando que o Streamlit reinicie durante análises muito longas.

A normalização é feita dentro do pipeline de validação cruzada para evitar
vazamento de informação entre treino e teste.
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

    if x in [
        "oe",
        "open",
        "open eyes",
        "olhos abertos",
        "abertos",
        "eye open",
        "eyes open"
    ]:
        return "OE"

    if x in [
        "ce",
        "closed",
        "closed eyes",
        "olhos fechados",
        "fechados",
        "eye closed",
        "eyes closed"
    ]:
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

    final_df = wide_df.copy()

    if len(oe_features) > 0:
        feature_sets["OE"] = oe_features

    if len(ce_features) > 0:
        feature_sets["CE"] = ce_features

    if len(oe_features) > 0 and len(ce_features) > 0:
        feature_sets["OE + CE"] = oe_features + ce_features

    # Delta CE - OE
    delta_features = []

    for feature in original_features:
        oe_col = f"{feature}__OE"
        ce_col = f"{feature}__CE"

        if oe_col in final_df.columns and ce_col in final_df.columns:
            delta_col = f"Delta__{feature}"
            final_df[delta_col] = final_df[ce_col] - final_df[oe_col]
            delta_features.append(delta_col)

    if len(delta_features) > 0:
        feature_sets["Delta CE - OE"] = delta_features

    # Razão CE/OE
    ratio_features = []

    for feature in original_features:
        oe_col = f"{feature}__OE"
        ce_col = f"{feature}__CE"

        if oe_col in final_df.columns and ce_col in final_df.columns:
            ratio_col = f"Ratio__{feature}"
            denominator = final_df[oe_col].replace(0, np.nan)
            final_df[ratio_col] = final_df[ce_col] / denominator
            ratio_features.append(ratio_col)

    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    if len(ratio_features) > 0:
        feature_sets["Razão CE/OE"] = ratio_features

    # OE + CE + Delta
    if len(oe_features) > 0 and len(ce_features) > 0 and len(delta_features) > 0:
        feature_sets["OE + CE + Delta"] = oe_features + ce_features + delta_features

    return final_df, feature_sets


def get_algorithm(name, random_state=42):
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
            n_estimators=300,
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

    return algorithms[name]


def compute_binary_metrics(y_true, y_pred, y_score, positive_label):
    labels = np.unique(y_true)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0
    )

    negative_labels = [label for label in labels if label != positive_label]

    if len(negative_labels) == 1:
        negative_label = negative_labels[0]

        specificity = recall_score(
            y_true,
            y_pred,
            pos_label=negative_label,
            zero_division=0
        )
    else:
        specificity = np.nan

    auc = np.nan

    if y_score is not None and len(labels) == 2:
        y_true_binary = np.array([
            1 if y == positive_label else 0 for y in y_true
        ])

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


def bootstrap_ci(values, n_bootstrap=500, ci=95, random_state=42):
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
    n_repeats=10,
    n_bootstrap=500,
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

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(X_train, y_train)

        # Predição no treino
        y_train_pred = pipeline.predict(X_train)

        # Predição no teste
        y_test_pred = pipeline.predict(X_test)

        # Scores para AUC no teste
        y_score = None

        try:
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                proba = pipeline.predict_proba(X_test)
                classes = pipeline.named_steps["model"].classes_

                if positive_label in classes:
                    pos_index = list(classes).index(positive_label)
                    y_score = proba[:, pos_index]

            elif hasattr(pipeline.named_steps["model"], "decision_function"):
                y_score = pipeline.decision_function(X_test)

        except Exception:
            y_score = None

        # Métricas no treino
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

        # Métricas no teste
        test_metrics = compute_binary_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_score=y_score,
            positive_label=positive_label
        )

        metrics = {
            "train_accuracy": train_accuracy,
            "train_balanced_accuracy": train_balanced_accuracy,

            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],

            "generalization_gap_accuracy": train_accuracy - test_metrics["accuracy"],
            "generalization_gap_balanced_accuracy": train_balanced_accuracy - test_metrics["balanced_accuracy"],

            "sensitivity": test_metrics["sensitivity"],
            "specificity": test_metrics["specificity"],
            "f1": test_metrics["f1"],
            "auc": test_metrics["auc"]
        }

        fold_metrics.append(metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_test_pred)

    metrics_df = pd.DataFrame(fold_metrics)

    summary = {}

    for metric in [
        "train_accuracy",
        "train_balanced_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "generalization_gap_accuracy",
        "generalization_gap_balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc"
    ]:
        values = metrics_df[metric].values

        summary[f"{metric}_mean"] = np.nanmean(values)
        summary[f"{metric}_sd"] = np.nanstd(values)

        ci_low, ci_high = bootstrap_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci=95,
            random_state=random_state
        )

        summary[f"{metric}_ci95_low"] = ci_low
        summary[f"{metric}_ci95_high"] = ci_high

    labels = np.unique(y)

    cm = confusion_matrix(
        np.array(all_y_true),
        np.array(all_y_pred),
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

    return importance_df


def classify_overfitting(gap):
    if pd.isna(gap):
        return "Não estimado"

    if gap < 0.05:
        return "Baixo"

    if gap < 0.15:
        return "Moderado"

    return "Alto"


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


# ============================================================
# Preparação dos dados
# ============================================================

st.header("4. Preparação dos dados")

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

wide_complete_df, feature_sets = build_feature_sets(
    wide_df=wide_df,
    original_features=selected_features
)

st.subheader("Base em formato largo")
safe_dataframe(wide_complete_df, n_rows=20)

st.write(f"Número de participantes na base larga: **{wide_complete_df.shape[0]}**")

feature_set_summary = pd.DataFrame({
    "Conjunto": list(feature_sets.keys()),
    "Número de variáveis": [len(v) for v in feature_sets.values()]
})

st.subheader("Conjuntos de variáveis disponíveis")
safe_dataframe(feature_set_summary)


# ============================================================
# Escolha modular da análise
# ============================================================

st.header("5. Escolha da análise")

algorithm_name = st.selectbox(
    "Escolha o algoritmo",
    [
        "LDA",
        "Regressão logística",
        "SVM linear",
        "SVM RBF",
        "Random Forest",
        "Gradient Boosting",
        "KNN",
        "Naive Bayes"
    ]
)

default_sets = []

for name in ["OE", "CE", "Delta CE - OE"]:
    if name in feature_sets:
        default_sets.append(name)

if len(default_sets) == 0:
    default_sets = list(feature_sets.keys())[:1]

selected_feature_sets = st.multiselect(
    "Escolha os conjuntos de variáveis a testar",
    list(feature_sets.keys()),
    default=default_sets
)

if len(selected_feature_sets) == 0:
    st.warning("Selecione pelo menos um conjunto de variáveis.")
    st.stop()


# ============================================================
# Configuração da validação
# ============================================================

st.header("6. Configuração da validação")

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
        max_value=100,
        value=10,
        step=1
    )

with col_cv3:
    n_bootstrap = st.number_input(
        "Bootstrap para IC95%",
        min_value=100,
        max_value=3000,
        value=500,
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
# Rodar análise escolhida
# ============================================================

st.header("7. Rodar análise modular")

st.write(
    f"""
Configuração atual:

- Algoritmo: **{algorithm_name}**
- Conjuntos selecionados: **{", ".join(selected_feature_sets)}**
- Validação: **{n_splits} folds × {n_repeats} repetições**
- Bootstrap: **{n_bootstrap} reamostragens**
"""
)

if st.button("Rodar análise selecionada"):

    algorithm = get_algorithm(
        name=algorithm_name,
        random_state=int(random_state)
    )

    summary_rows = []

    progress_bar = st.progress(0)

    for i, feature_set_name in enumerate(selected_feature_sets):

        progress_bar.progress((i + 1) / len(selected_feature_sets))

        st.markdown("---")
        st.subheader(f"Conjunto: {feature_set_name}")

        features = feature_sets[feature_set_name]

        model_df = wide_complete_df[[group_col] + features].copy()
        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df = model_df.dropna()

        st.write(f"N participantes usados: **{model_df.shape[0]}**")
        st.write(f"N variáveis usadas: **{len(features)}**")

        group_count_model = model_df[group_col].astype(str).value_counts()
        group_count_df = group_count_model.reset_index()
        group_count_df.columns = ["Grupo", "n"]
        safe_dataframe(group_count_df)

        if group_count_model.min() < n_splits:
            st.warning(
                f"O menor grupo tem {group_count_model.min()} participantes. "
                f"Reduza o número de folds."
            )
            continue

        X = model_df[features].astype(float).to_numpy()
        y = model_df[group_col].astype(str).to_numpy()

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

        gap_balanced = summary["generalization_gap_balanced_accuracy_mean"]

        row = {
            "Algoritmo": algorithm_name,
            "Conjunto": feature_set_name,
            "N participantes": model_df.shape[0],
            "N variáveis": len(features),

            "Acurácia treino média": summary["train_accuracy_mean"],
            "Acurácia treino DP": summary["train_accuracy_sd"],
            "Acurácia teste média": summary["test_accuracy_mean"],
            "Acurácia teste DP": summary["test_accuracy_sd"],
            "Gap acurácia treino-teste": summary["generalization_gap_accuracy_mean"],

            "Balanced accuracy treino média": summary["train_balanced_accuracy_mean"],
            "Balanced accuracy treino DP": summary["train_balanced_accuracy_sd"],
            "Balanced accuracy teste média": summary["test_balanced_accuracy_mean"],
            "Balanced accuracy teste DP": summary["test_balanced_accuracy_sd"],
            "Gap balanced accuracy treino-teste": gap_balanced,
            "Risco de overfitting": classify_overfitting(gap_balanced),

            "Balanced accuracy teste IC95% inferior": summary["test_balanced_accuracy_ci95_low"],
            "Balanced accuracy teste IC95% superior": summary["test_balanced_accuracy_ci95_high"],

            "Sensibilidade teste média": summary["sensitivity_mean"],
            "Especificidade teste média": summary["specificity_mean"],
            "F1 teste médio": summary["f1_mean"],
            "AUC teste média": summary["auc_mean"],
            "AUC teste IC95% inferior": summary["auc_ci95_low"],
            "AUC teste IC95% superior": summary["auc_ci95_high"]
        }

        summary_rows.append(row)

        st.write("Resumo do desempenho:")
        safe_dataframe(pd.DataFrame([row]))

        st.write("Matriz de confusão acumulada no teste:")
        safe_dataframe(cm_df)

        st.write("Métricas por fold/repetição:")
        safe_dataframe(metrics_df)

        importance_df = fit_final_model_and_importance(
            X=X,
            y=y,
            feature_names=features,
            model=algorithm
        )

        if importance_df is not None:
            st.write("Importância das variáveis:")
            safe_dataframe(importance_df)

            top_imp = importance_df.head(20)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(
                top_imp["Variável"][::-1],
                top_imp["Importância"][::-1]
            )
            ax.set_xlabel("Importância")
            ax.set_title(f"Variáveis mais importantes - {feature_set_name}")
            ax.grid(axis="x", alpha=0.3)

            st.pyplot(fig)

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)

        summary_df = summary_df.sort_values(
            by="Balanced accuracy teste média",
            ascending=False
        )

        st.header("8. Resumo comparativo da rodada")
        safe_dataframe(summary_df)

        # ====================================================
        # Gráfico: desempenho no teste
        # ====================================================

        st.subheader("Balanced accuracy no teste")

        fig2, ax2 = plt.subplots(figsize=(9, 5))

        ax2.barh(
            summary_df["Conjunto"][::-1],
            summary_df["Balanced accuracy teste média"][::-1]
        )

        ax2.set_xlabel("Balanced accuracy no teste")
        ax2.set_title(f"Comparação dos conjuntos - {algorithm_name}")
        ax2.set_xlim(0, 1)
        ax2.grid(axis="x", alpha=0.3)

        st.pyplot(fig2)

        # ====================================================
        # Gráfico: treino versus teste
        # ====================================================

        st.subheader("Comparação treino versus teste")

        plot_df = summary_df.copy()
        plot_df["Modelo"] = plot_df["Conjunto"]

        x = np.arange(len(plot_df))
        width = 0.35

        fig3, ax3 = plt.subplots(figsize=(10, 5))

        ax3.bar(
            x - width / 2,
            plot_df["Balanced accuracy treino média"],
            width,
            label="Treino"
        )

        ax3.bar(
            x + width / 2,
            plot_df["Balanced accuracy teste média"],
            width,
            label="Teste"
        )

        ax3.set_ylabel("Balanced accuracy")
        ax3.set_title(f"Treino versus teste - {algorithm_name}")
        ax3.set_xticks(x)
        ax3.set_xticklabels(plot_df["Modelo"], rotation=45, ha="right")
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        st.pyplot(fig3)

        # ====================================================
        # Gráfico: gap treino-teste
        # ====================================================

        st.subheader("Gap treino - teste")

        fig4, ax4 = plt.subplots(figsize=(9, 5))

        ax4.barh(
            summary_df["Conjunto"][::-1],
            summary_df["Gap balanced accuracy treino-teste"][::-1]
        )

        ax4.set_xlabel("Gap de balanced accuracy")
        ax4.set_title(f"Possível overfitting - {algorithm_name}")
        ax4.grid(axis="x", alpha=0.3)

        st.pyplot(fig4)

        csv_results = summary_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Baixar resultados desta rodada em CSV",
            data=csv_results,
            file_name=f"resultados_{algorithm_name.replace(' ', '_')}.csv",
            mime="text/csv"
        )


# ============================================================
# Orientação
# ============================================================

st.header("9. Estratégia recomendada")

st.markdown(
    """
### Como usar sem travar o Streamlit

Sugestão prática:

1. Primeiro rode com:
   - 5 folds
   - 10 repetições
   - 500 bootstraps

2. Teste um algoritmo por vez.

3. Comece pelos algoritmos mais rápidos:
   - LDA
   - Regressão logística
   - Naive Bayes
   - SVM linear

4. Depois teste:
   - Random Forest
   - Gradient Boosting
   - SVM RBF

5. Quando encontrar os melhores candidatos, aumente apenas neles:
   - 30 ou 50 repetições
   - 1000 bootstraps

---

### Interpretação do desempenho

A métrica principal recomendada é:

`Balanced accuracy teste média`

Ela deve ser interpretada junto com:

`Gap balanced accuracy treino-teste`

---

### Interpretação do gap treino-teste

- **Gap < 0.05**: baixo sinal de overfitting.
- **Gap entre 0.05 e 0.15**: possível overfitting moderado.
- **Gap > 0.15**: provável overfitting importante.

Exemplo:

`Treino = 0.95` e `Teste = 0.62`

indica que o modelo aprendeu muito bem os dados de treino, mas generalizou mal.

---

### Interpretação dos conjuntos

- **OE**: testa se os grupos já se separam com olhos abertos.
- **CE**: testa se a retirada da visão aumenta a separação.
- **Delta CE - OE**: testa se a resposta à retirada da visão separa os grupos.
- **Razão CE/OE**: testa se a mudança proporcional separa os grupos.
- **OE + CE**: testa se as duas condições carregam informação complementar.

---

### Recomendação científica

Um modelo ideal não é necessariamente o que tem maior acurácia no treino.
O melhor modelo deve combinar:

1. boa balanced accuracy no teste;
2. baixo gap treino-teste;
3. boa sensibilidade e especificidade;
4. plausibilidade fisiológica;
5. interpretabilidade.
"""
)
