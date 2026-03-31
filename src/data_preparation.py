import pandas as pd
import numpy as np
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "austin_animal_center.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "animals_clean.csv")
MIN_BREED_COUNT = 50
MIN_BREED_FILTER = 10


def parse_age_to_months(age_str):
    if pd.isna(age_str) or not isinstance(age_str, str):
        return np.nan
    age_str = age_str.strip().lower()
    match = re.match(r"(\d+)\s+(year|month|week|day)s?", age_str)
    if not match:
        return np.nan
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "year":
        return value * 12.0
    elif unit == "month":
        return float(value)
    elif unit == "week":
        return value / 4.33
    elif unit == "day":
        return value / 30.44
    return np.nan


def extract_neutered(sex_str):
    if pd.isna(sex_str):
        return 0
    sex_str = sex_str.lower()
    if "spayed" in sex_str or "neutered" in sex_str:
        return 1
    return 0


def extract_sex(sex_str):
    if pd.isna(sex_str):
        return "Unknown"
    if "male" in sex_str.lower():
        if "female" in sex_str.lower():
            return "Female"
        return "Male"
    if "female" in sex_str.lower():
        return "Female"
    return "Unknown"


def classify_age_group(months):
    if pd.isna(months):
        return "Unknown"
    if months <= 6:
        return "Filhote"
    elif months <= 24:
        return "Jovem"
    elif months <= 96:
        return "Adulto"
    else:
        return "Idoso"


def is_mix(breed):
    if pd.isna(breed):
        return 0
    breed = breed.lower()
    return 1 if ("mix" in breed or "/" in breed) else 0


def clean_breed(breed):
    if pd.isna(breed):
        return "Unknown"
    b = str(breed).split("/")[0]
    b = re.sub(r"(?i)\s+mix$", "", b)
    return b.strip()


def has_health_condition(row):
    subtype = str(row.get("outcome_subtype", "")).lower()
    outcome = str(row.get("outcome_type", "")).lower()
    condition_keywords = [
        "sick", "injured", "medical", "suffering", "rabies",
        "aggressive", "behavior", "underage", "foster"
    ]
    if any(kw in subtype for kw in condition_keywords):
        return 1
    if outcome == "euthanasia":
        if any(kw in subtype for kw in ["sick", "injured", "suffering", "medical"]):
            return 1
    return 0


def prepare_data():
    print("=" * 60)
    print("FASE 1+2 — Preparação dos Dados")
    print("=" * 60)

    print(f"\n📂 Carregando dados de: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"   Registros carregados: {len(df):,}")
    print(f"   Colunas: {list(df.columns)}")

    print(f"\n📊 Valores ausentes por coluna:")
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            pct = missing[col] / len(df) * 100
            print(f"   {col}: {missing[col]:,} ({pct:.1f}%)")

    n_before = len(df)
    df = df.drop_duplicates(subset=["animal_id", "datetime"], keep="first")
    n_after = len(df)
    print(f"\n🔄 Duplicatas removidas: {n_before - n_after:,}")

    df["is_adopted"] = (df["outcome_type"] == "Adoption").astype(int)
    adoption_rate = df["is_adopted"].mean() * 100
    print(f"\n🎯 Variável alvo criada: is_adopted")
    print(f"   Taxa de adoção: {adoption_rate:.1f}%")

    df["age_months"] = df["age_upon_outcome"].apply(parse_age_to_months)
    age_missing = df["age_months"].isna().sum()
    print(f"\n📅 Idade convertida para meses")
    print(f"   Idade média: {df['age_months'].mean():.1f} meses")
    print(f"   Valores ausentes: {age_missing:,}")

    df["age_months"] = df.groupby("animal_type")["age_months"].transform(
        lambda x: x.fillna(x.median())
    )
    df["age_months"] = df["age_months"].fillna(df["age_months"].median())

    df["is_neutered"] = df["sex_upon_outcome"].apply(extract_neutered)
    print(f"\n✂️  Castração: {df['is_neutered'].mean()*100:.1f}% castrados")

    df["sex"] = df["sex_upon_outcome"].apply(extract_sex)

    df["is_mix"] = df["breed"].apply(is_mix)
    print(f"🐾 Mestiços: {df['is_mix'].mean()*100:.1f}%")

    df["breed"] = df["breed"].apply(clean_breed)
    print(f"🐕 Raças únicas após limpeza: {df['breed'].nunique()}")

    df["has_condition"] = df.apply(has_health_condition, axis=1)
    print(f"🏥 Com condição especial: {df['has_condition'].mean()*100:.1f}%")

    df["age_group"] = df["age_months"].apply(classify_age_group)
    print(f"\n📊 Distribuição por faixa etária:")
    for group, count in df["age_group"].value_counts().items():
        print(f"   {group}: {count:,} ({count/len(df)*100:.1f}%)")

    breed_counts = df["breed"].value_counts()
    common_breeds = breed_counts[breed_counts >= MIN_BREED_COUNT].index
    df["breed_simplified"] = df["breed"].where(
        df["breed"].isin(common_breeds), "Other"
    )
    n_breeds = df["breed_simplified"].nunique()
    print(f"\n🐕 Raças simplificadas: {n_breeds} categorias (mínimo {MIN_BREED_COUNT} registros)")

    n_before = len(df)
    breed_counts_filter = df["breed"].value_counts()
    rare_breeds = breed_counts_filter[breed_counts_filter < MIN_BREED_FILTER].index
    df = df[~df["breed"].isin(rare_breeds)]
    n_removed = n_before - len(df)
    print(f"\n🗑️  Raças com menos de {MIN_BREED_FILTER} registros removidas:")
    print(f"   Raças descartadas: {len(rare_breeds)}")
    print(f"   Animais removidos: {n_removed:,}")
    print(f"   Animais restantes: {len(df):,}")

    print(f"\n🐾 Distribuição por espécie:")
    for species, count in df["animal_type"].value_counts().items():
        print(f"   {species}: {count:,} ({count/len(df)*100:.1f}%)")

    columns_to_keep = [
        "animal_id", "name", "animal_type", "breed", "breed_simplified",
        "color", "sex", "age_upon_outcome", "age_months", "age_group",
        "is_neutered", "is_mix", "has_condition",
        "outcome_type", "outcome_subtype", "is_adopted",
        "datetime", "date_of_birth"
    ]
    df = df[columns_to_keep]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n✅ Dados processados salvos em: {PROCESSED_PATH}")
    print(f"   Total de registros: {len(df):,}")
    print(f"   Total de colunas: {len(df.columns)}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    prepare_data()
