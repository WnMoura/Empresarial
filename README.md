# 🐾 Adote — Austin Animal Center

> Pipeline completo de ciência de dados para análise e predição de adotabilidade animal, com dashboard interativo.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly_Dash-2.x-informational?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Ativo-brightgreen)

---

## Sobre o Projeto

Este projeto processa dados históricos do [Austin Animal Center](https://data.austintexas.gov/) para entender quais fatores influenciam a adoção de animais. Um modelo de **Random Forest** é treinado para prever a probabilidade de adoção de cada animal, e os resultados são apresentados em um **dashboard Plotly Dash** com comparativos detalhados entre cães e gatos.

---

## Resultados do Modelo

| Métrica | Resultado | Mínimo |
|---------|:---------:|:------:|
| Acurácia | **80.4%** | 65% ✅ |
| AUC-ROC | **0.86** | 0.70 ✅ |
| F1-Score | **0.81** | 0.60 ✅ |

**Variável mais importante:** castração (`is_neutered`) com **54.9%** de importância relativa no modelo.

---

## Estrutura do Projeto

```
Projeto-adote/
├── app/
│   └── dashboard.py          # Dashboard interativo (Plotly Dash)
├── src/
│   ├── data_preparation.py   # Pipeline de limpeza e feature engineering
│   └── model_training.py     # Treinamento do Random Forest + SHAP
├── data/
│   ├── raw/                  # Dados brutos (CSV original)
│   └── processed/            # Dados limpos e com scores do modelo
├── models/                   # Modelo e encoders treinados (.joblib)
├── notebooks/figures/        # Gráficos gerados pelo pipeline
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Configurar ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar os dados

```bash
python src/data_preparation.py
```

### 3. Treinar o modelo

```bash
python src/model_training.py
```

### 4. Iniciar o dashboard

```bash
python app/dashboard.py
```

Acesse em: **http://127.0.0.1:8050**

---

## Dashboard

O dashboard apresenta uma análise comparativa completa entre cães e gatos:

- **KPIs globais** — total de animais, taxa de adoção, raça com maior probabilidade prevista pelo modelo
- **Ranking de raças** — top 10 mais adotadas por espécie (taxa real + probabilidade do modelo)
- **Fatores de adoção** — impacto de faixa etária, castração e condição de saúde
- **Evolução temporal** — adoções por trimestre ao longo dos anos
- **Tabela completa** — ranking de todas as raças com taxa real e probabilidade prevista pelo modelo, ordenável e filtrável

---

## Stack Tecnológica

| Categoria | Bibliotecas |
|-----------|-------------|
| Dados | `pandas`, `numpy` |
| Modelagem | `scikit-learn` (Random Forest), `joblib` |
| Explicabilidade | `shap` |
| Visualização | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `dash`, `dash-bootstrap-components` |

---

## Limitações

- Os dados refletem o comportamento histórico de um abrigo específico (Austin, TX) e podem não generalizar para outras regiões.
- A probabilidade prevista pelo modelo é uma ferramenta de apoio à decisão, não um julgamento sobre os animais.
- Raças com taxa de adoção baixa não são menos merecedoras — fatores externos (mídia, sazonalidade) não estão nos dados.

---

*Baseado nos dados públicos do [Austin Animal Center](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238).*
