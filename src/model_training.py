import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "animals_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.joblib")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "animals_with_score.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "notebooks", "figures")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [
    "animal_type_enc",
    "breed_simplified_enc",
    "age_months",
    "is_neutered",
    "has_condition",
    "is_mix",
    "age_group_enc",
    "sex_enc",
]

TARGET = "is_adopted"


def train_model():
    print("=" * 60)
    print("FASE 4 — Modelagem Preditiva")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Dados carregados: {len(df):,} registros")

    df = df[df["animal_type"].isin(["Dog", "Cat"])].copy()
    print(f"   Filtrado para Dog/Cat: {len(df):,} registros")

    label_encoders = {}
    for col, enc_col in [
        ("animal_type", "animal_type_enc"),
        ("breed_simplified", "breed_simplified_enc"),
        ("age_group", "age_group_enc"),
        ("sex", "sex_enc"),
    ]:
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col].fillna("Unknown"))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} classes")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X = X.fillna(0)

    print(f"\n📊 Distribuição da variável alvo:")
    print(f"   Adotado: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"   Não adotado: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n📐 Divisão dos dados:")
    print(f"   Treino: {len(X_train):,} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Teste:  {len(X_test):,} ({len(X_test)/len(X)*100:.0f}%)")

    print(f"\n🌳 Treinando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✅ Modelo treinado!")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📈 Métricas de Avaliação:")
    print(f"   Acurácia:  {acc:.4f} {'✅' if acc >= 0.65 else '❌'} (mínimo: 0.65)")
    print(f"   AUC-ROC:   {auc:.4f} {'✅' if auc >= 0.70 else '❌'} (mínimo: 0.70)")
    print(f"   F1-Score:  {f1:.4f} {'✅' if f1 >= 0.60 else '❌'} (mínimo: 0.60)")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Não Adotado", "Adotado"]))

    print(f"📊 Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {cm}")

    print(f"\n🔑 Importância das Variáveis:")
    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    for _, row in feature_importance.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind="barh", x="feature", y="importance", ax=ax,
                            color="steelblue", legend=False)
    ax.set_title("Importância das Variáveis — Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importância")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "11_feature_importance.png"))
    plt.close()

    print(f"\n🎯 Gerando scores de adotabilidade para todos os animais...")
    df["adoption_score"] = model.predict_proba(X[FEATURES])[:, 1] * 100
    df["adoption_score"] = df["adoption_score"].round(1)

    print(f"   Score médio: {df['adoption_score'].mean():.1f}%")
    print(f"   Score mediano: {df['adoption_score'].median():.1f}%")
    print(f"   Mín: {df['adoption_score'].min():.1f}% | Máx: {df['adoption_score'].max():.1f}%")

    print(f"\n🔍 Gerando explicações SHAP...")
    try:
        import shap
        sample_size = min(1000, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, 1], X_sample, feature_names=FEATURES,
                         show=False, plot_size=(10, 6))
        plt.title("SHAP — Impacto das Variáveis na Predição", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "12_shap_summary.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("   ✅ SHAP summary salvo!")
    except Exception as e:
        print(f"   ⚠️  SHAP falhou (não crítico): {e}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Modelo salvo em: {MODEL_PATH}")

    joblib.dump(label_encoders, ENCODERS_PATH)
    print(f"💾 Encoders salvos em: {ENCODERS_PATH}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Dados com score salvos em: {OUTPUT_PATH}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for species in ["Dog", "Cat"]:
        subset = df[df["animal_type"] == species]["adoption_score"]
        ax.hist(subset, bins=40, alpha=0.6, label=species, edgecolor="white")
    ax.set_title("Distribuição dos Scores de Adotabilidade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score de Adotabilidade (%)")
    ax.set_ylabel("Frequência")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_score_distribution.png"))
    plt.close()

    print(f"\n✅ Modelagem concluída com sucesso!")
    print("=" * 60)

    return model, df


if __name__ == "__main__":
    train_model()
