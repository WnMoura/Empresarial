import pandas as pd
import numpy as np
import os

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "animals_with_score.csv")

BREED_PT = {
    "Pit Bull": "Pit Bull",
    "Labrador Retriever": "Labrador Retriever",
    "Chihuahua Shorthair": "Chihuahua (Pelo Curto)",
    "Chihuahua Longhair": "Chihuahua (Pelo Longo)",
    "German Shepherd": "Pastor Alemão",
    "Australian Cattle Dog": "Boiadeiro Australiano",
    "Siberian Husky": "Husky Siberiano",
    "Dachshund": "Dachshund (Salsicha)",
    "Boxer": "Boxer",
    "Border Collie": "Border Collie",
    "Great Pyrenees": "Montanhês dos Pirenéus",
    "Miniature Poodle": "Poodle Miniatura",
    "Australian Shepherd": "Pastor Australiano",
    "Catahoula": "Catahoula",
    "Beagle": "Beagle",
    "Yorkshire Terrier": "Yorkshire Terrier",
    "Miniature Schnauzer": "Schnauzer Miniatura",
    "Jack Russell Terrier": "Jack Russell Terrier",
    "Rat Terrier": "Rat Terrier",
    "Pointer": "Pointer",
    "Staffordshire": "Staffordshire",
    "Rottweiler": "Rottweiler",
    "Cairn Terrier": "Cairn Terrier",
    "Anatol Shepherd": "Pastor Anatólio",
    "Shih Tzu": "Shih Tzu",
    "American Bulldog": "Bulldog Americano",
    "Black Mouth Cur": "Black Mouth Cur",
    "Plott Hound": "Plott Hound",
    "American Pit Bull Terrier": "Pit Bull Terrier Americano",
    "Australian Kelpie": "Kelpie Australiano",
    "Cocker Spaniel": "Cocker Spaniel",
    "Pug": "Pug",
    "Maltese": "Maltês",
    "Pomeranian": "Lulu da Pomerânia",
    "French Bulldog": "Buldogue Francês",
    "English Bulldog": "Buldogue Inglês",
    "Boston Terrier": "Boston Terrier",
    "Chow Chow": "Chow Chow",
    "Golden Retriever": "Golden Retriever",
    "Blue Lacy": "Blue Lacy",
    "Dutch Shepherd": "Pastor Holandês",
    "Redbone Hound": "Redbone Hound",
    "Lhasa Apso": "Lhasa Apso",
    "Pekingese": "Pequinês",
    "Doberman Pinsch": "Dobermann",
    "Weimaraner": "Weimaraner",
    "Bichon Frise": "Bichon Frisé",
    "West Highland": "West Highland Terrier",
    "Standard Poodle": "Poodle Standard",
    "Carolina Dog": "Cão da Carolina",
    "Basenji": "Basenji",
    "Domestic Shorthair": "Pelo Curto Doméstico",
    "Domestic Medium Hair": "Pelo Médio Doméstico",
    "Domestic Longhair": "Pelo Longo Doméstico",
    "Siamese": "Siamês",
    "Snowshoe": "Snowshoe",
    "American Shorthair": "Pelo Curto Americano",
    "Maine Coon": "Maine Coon",
    "Manx": "Manx",
    "Russian Blue": "Azul Russo",
    "Ragdoll": "Ragdoll",
    "Himalayan": "Himalaia",
    "Persian": "Persa",
    "Bengal": "Bengal",
    "Abyssinian": "Abissínio",
    "Balinese": "Balinês",
    "British Shorthair": "Pelo Curto Britânico",
    "Angora": "Angorá",
    "Tonkinese": "Tonquinês",
}


def translate_breed(breed):
    return BREED_PT.get(breed, breed)


df_all = pd.read_csv(DATA_PATH)

if "adoption_score" not in df_all.columns:
    df_all["adoption_score"] = 50.0

df_all["breed_pt"] = df_all["breed"].apply(translate_breed)
df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
df_all["year_month"] = df_all["datetime"].dt.to_period("M").astype(str)
df_all["year"] = df_all["datetime"].dt.year

OUTCOME_PT = {
    "Adoption": "Adoção",
    "Transfer": "Transferência",
    "Return to Owner": "Devolvido ao Dono",
    "Euthanasia": "Eutanásia",
    "Died": "Óbito",
    "Rto-Adopt": "Devolvido/Adotado",
    "Disposal": "Descarte",
    "Missing": "Desaparecido",
    "Relocate": "Realocado",
    "Stolen": "Roubado",
    "Lost": "Perdido",
}
df_all["outcome_pt"] = df_all["outcome_type"].map(OUTCOME_PT).fillna(df_all["outcome_type"])

AGE_GROUP_PT = {
    "Filhote": "Filhote (0-6m)",
    "Jovem": "Jovem (6m-2a)",
    "Adulto": "Adulto (2-8a)",
    "Idoso": "Idoso (8a+)",
}
df_all["age_group_pt"] = df_all["age_group"].map(AGE_GROUP_PT).fillna(df_all["age_group"])

df_dogs = df_all[df_all["animal_type"] == "Dog"].copy()
df_cats = df_all[df_all["animal_type"] == "Cat"].copy()

years_available = sorted(df_all["year"].dropna().unique().astype(int))

COLORS = {
    "bg": "#0a0e1a",
    "card_bg": "#111827",
    "card_border": "#1f2937",
    "dog": "#3B82F6",
    "dog_light": "#60A5FA",
    "cat": "#F59E0B",
    "cat_light": "#FBBF24",
    "accent_green": "#10B981",
    "accent_red": "#EF4444",
    "accent_purple": "#8B5CF6",
    "text": "#F9FAFB",
    "text_muted": "#9CA3AF",
    "text_dim": "#6B7280",
    "grid": "rgba(31,41,55,0.5)",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['card_border']}",
    "padding": "20px",
    "height": "100%",
}

SECTION_HEADER_STYLE = {
    "color": COLORS["text"],
    "fontWeight": "700",
    "fontSize": "18px",
    "marginBottom": "4px",
}

SECTION_DESC_STYLE = {
    "color": COLORS["text_muted"],
    "fontSize": "13px",
    "marginBottom": "16px",
    "lineHeight": "1.5",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="🐾 Adote — Panorama da Adoção Animal",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Dashboard comparativo de adotabilidade animal — Austin Animal Center"},
    ],
    update_title=None,
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
            body { background-color: #0a0e1a; }
            .Select-control { background-color: #111827 !important; border-color: #1f2937 !important; }
            .Select-value-label { color: #fff !important; }
            .Select-menu-outer { background-color: #111827 !important; }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                border: 1px solid #1f2937 !important;
            }
            .js-plotly-plot .plotly .main-svg { will-change: auto !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def kpi_mini(icon, label, value, color, delta=None, delta_color=None):
    children = [
        html.Div(
            [
                html.Span(icon, style={"fontSize": "20px", "marginRight": "8px"}),
                html.Span(label, style={
                    "fontSize": "11px", "fontWeight": "600",
                    "color": COLORS["text_muted"], "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                }),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
        ),
        html.Div(
            value,
            style={
                "fontSize": "28px", "fontWeight": "800", "color": color,
                "lineHeight": "1",
            },
        ),
    ]
    if delta:
        children.append(html.Div(
            delta,
            style={
                "fontSize": "12px", "color": delta_color or COLORS["text_muted"],
                "marginTop": "4px",
            },
        ))
    return html.Div(children, style={**CARD_STYLE, "padding": "16px"})


def section_title(icon, title, description):
    return html.Div(
        [
            html.H3(f"{icon} {title}", style=SECTION_HEADER_STYLE),
            html.P(description, style=SECTION_DESC_STYLE),
        ],
        style={"marginTop": "32px", "marginBottom": "8px"},
    )


total_animals = len(df_all)
total_dogs = len(df_dogs)
total_cats = len(df_cats)
adoption_rate_all = df_all["is_adopted"].mean() * 100
adoption_rate_dogs = df_dogs["is_adopted"].mean() * 100
adoption_rate_cats = df_cats["is_adopted"].mean() * 100
score_dogs = df_dogs["adoption_score"].mean()
score_cats = df_cats["adoption_score"].mean()
total_breeds_dogs = df_dogs["breed"].nunique()
total_breeds_cats = df_cats["breed"].nunique()


def get_breed_prob(df, min_count=30):
    stats = df.groupby("breed_pt").agg(
        total=("is_adopted", "count"),
        prob_adocao=("adoption_score", "mean"),
    ).reset_index()
    stats = stats[stats["total"] >= min_count]
    stats["prob_adocao"] = stats["prob_adocao"].round(1)
    return stats.sort_values("prob_adocao", ascending=False)


breed_prob_dogs = get_breed_prob(df_dogs)
breed_prob_cats = get_breed_prob(df_cats)

top_prob_dog = breed_prob_dogs.iloc[0]["breed_pt"] if len(breed_prob_dogs) > 0 else "—"
top_prob_dog_val = breed_prob_dogs.iloc[0]["prob_adocao"] if len(breed_prob_dogs) > 0 else 0
top_prob_cat = breed_prob_cats.iloc[0]["breed_pt"] if len(breed_prob_cats) > 0 else "—"
top_prob_cat_val = breed_prob_cats.iloc[0]["prob_adocao"] if len(breed_prob_cats) > 0 else 0


def get_top_breeds(df, n=10, min_count=50):
    stats = df.groupby("breed_pt")["is_adopted"].agg(["mean", "count"]).reset_index()
    stats = stats[stats["count"] >= min_count]
    stats["rate"] = stats["mean"] * 100
    return stats.sort_values("rate", ascending=False).head(n)


top_dogs = get_top_breeds(df_dogs, n=10)
top_cats = get_top_breeds(df_cats, n=10)


def get_outcome_dist(df):
    counts = df["outcome_pt"].value_counts()
    top5 = counts.head(5)
    other = counts.iloc[5:].sum()
    if other > 0:
        top5["Outros"] = other
    return top5


outcome_dogs = get_outcome_dist(df_dogs)
outcome_cats = get_outcome_dist(df_cats)

age_order = ["Filhote (0-6m)", "Jovem (6m-2a)", "Adulto (2-8a)", "Idoso (8a+)"]

hm_dogs = df_dogs.pivot_table(
    values="is_adopted", index="age_group_pt", aggfunc="mean"
).reindex([a for a in age_order]).fillna(0) * 100

hm_cats = df_cats.pivot_table(
    values="is_adopted", index="age_group_pt", aggfunc="mean"
).reindex([a for a in age_order]).fillna(0) * 100

neuter_dogs = df_dogs.groupby("is_neutered")["is_adopted"].mean() * 100
neuter_cats = df_cats.groupby("is_neutered")["is_adopted"].mean() * 100

health_dogs = df_dogs.groupby("has_condition")["is_adopted"].mean() * 100
health_cats = df_cats.groupby("has_condition")["is_adopted"].mean() * 100

fig_breeds = make_subplots(
    rows=1, cols=2,
    subplot_titles=["🐕 Cães — Top 10 Raças Mais Adotadas", "🐱 Gatos — Top 10 Raças Mais Adotadas"],
    horizontal_spacing=0.15,
)

fig_breeds.add_trace(go.Bar(
    y=top_dogs["breed_pt"], x=top_dogs["rate"], orientation="h",
    marker_color=COLORS["dog"], name="Cães",
    text=[f"{r:.1f}%" for r in top_dogs["rate"]],
    textposition="outside", textfont=dict(size=11),
    showlegend=False,
), row=1, col=1)

fig_breeds.add_trace(go.Bar(
    y=top_cats["breed_pt"], x=top_cats["rate"], orientation="h",
    marker_color=COLORS["cat"], name="Gatos",
    text=[f"{r:.1f}%" for r in top_cats["rate"]],
    textposition="outside", textfont=dict(size=11),
    showlegend=False,
), row=1, col=2)

breeds_layout = {k: v for k, v in PLOT_LAYOUT.items() if k != "margin"}
fig_breeds.update_layout(
    **breeds_layout, height=500,
    margin=dict(l=10, r=20, t=60, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
fig_breeds.update_yaxes(autorange="reversed", row=1, col=1, gridcolor=COLORS["grid"])
fig_breeds.update_yaxes(autorange="reversed", row=1, col=2, gridcolor=COLORS["grid"])
fig_breeds.update_xaxes(title_text="Taxa (%)", row=1, col=1, gridcolor=COLORS["grid"], range=[0, 115])
fig_breeds.update_xaxes(title_text="Taxa (%)", row=1, col=2, gridcolor=COLORS["grid"], range=[0, 115])
fig_breeds.update_annotations(font=dict(size=12, color=COLORS["text"]))

donut_colors = [COLORS["accent_green"], COLORS["dog"], COLORS["accent_purple"],
                COLORS["accent_red"], COLORS["text_dim"], COLORS["cat"]]

fig_outcomes = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "pie"}, {"type": "pie"}]],
    subplot_titles=["🐕 Desfechos — Cães", "🐱 Desfechos — Gatos"],
)

fig_outcomes.add_trace(go.Pie(
    labels=outcome_dogs.index, values=outcome_dogs.values,
    hole=0.55, marker=dict(colors=donut_colors),
    textinfo="label+percent", textfont=dict(size=11),
    insidetextorientation="auto",
), row=1, col=1)

fig_outcomes.add_trace(go.Pie(
    labels=outcome_cats.index, values=outcome_cats.values,
    hole=0.55, marker=dict(colors=donut_colors),
    textinfo="label+percent", textfont=dict(size=11),
    insidetextorientation="auto",
), row=1, col=2)

fig_outcomes.update_layout(
    **breeds_layout, height=400, showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    margin=dict(l=10, r=10, t=60, b=10)
)
fig_outcomes.update_annotations(font=dict(size=13, color=COLORS["text"]))

fig_age = go.Figure()
fig_age.add_trace(go.Bar(
    x=age_order,
    y=[hm_dogs.loc[a, "is_adopted"] if a in hm_dogs.index else 0 for a in age_order],
    name="🐕 Cães", marker_color=COLORS["dog"],
    text=[f"{hm_dogs.loc[a, 'is_adopted']:.1f}%" if a in hm_dogs.index else "0%" for a in age_order],
    textposition="outside",
))
fig_age.add_trace(go.Bar(
    x=age_order,
    y=[hm_cats.loc[a, "is_adopted"] if a in hm_cats.index else 0 for a in age_order],
    name="🐱 Gatos", marker_color=COLORS["cat"],
    text=[f"{hm_cats.loc[a, 'is_adopted']:.1f}%" if a in hm_cats.index else "0%" for a in age_order],
    textposition="outside",
))
fig_age.update_layout(
    **breeds_layout, height=400, barmode="group",
    xaxis=dict(gridcolor=COLORS["grid"], tickangle=-45),
    yaxis=dict(title="Taxa de Adoção (%)", gridcolor=COLORS["grid"], range=[0, 85]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=40)
)

fig_neuter = go.Figure()
categories = ["Inteiro", "Castrado"]
fig_neuter.add_trace(go.Bar(
    x=categories,
    y=[neuter_dogs.get(0, 0), neuter_dogs.get(1, 0)],
    name="🐕 Cães", marker_color=COLORS["dog"],
    text=[f"{neuter_dogs.get(0, 0):.1f}%", f"{neuter_dogs.get(1, 0):.1f}%"],
    textposition="outside",
))
fig_neuter.add_trace(go.Bar(
    x=categories,
    y=[neuter_cats.get(0, 0), neuter_cats.get(1, 0)],
    name="🐱 Gatos", marker_color=COLORS["cat"],
    text=[f"{neuter_cats.get(0, 0):.1f}%", f"{neuter_cats.get(1, 0):.1f}%"],
    textposition="outside",
))
fig_neuter.update_layout(
    **breeds_layout, height=400, barmode="group",
    xaxis=dict(gridcolor=COLORS["grid"]),
    yaxis=dict(title="Taxa de Adoção (%)", gridcolor=COLORS["grid"], range=[0, 115]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=20)
)

fig_health = go.Figure()
health_cats_labels = ["Saudável", "Com Condição"]
fig_health.add_trace(go.Bar(
    x=health_cats_labels,
    y=[health_dogs.get(0, 0), health_dogs.get(1, 0)],
    name="🐕 Cães", marker_color=COLORS["dog"],
    text=[f"{health_dogs.get(0, 0):.1f}%", f"{health_dogs.get(1, 0):.1f}%"],
    textposition="outside",
))
fig_health.add_trace(go.Bar(
    x=health_cats_labels,
    y=[health_cats.get(0, 0), health_cats.get(1, 0)],
    name="🐱 Gatos", marker_color=COLORS["cat"],
    text=[f"{health_cats.get(0, 0):.1f}%", f"{health_cats.get(1, 0):.1f}%"],
    textposition="outside",
))
fig_health.update_layout(
    **breeds_layout, height=400, barmode="group",
    xaxis=dict(gridcolor=COLORS["grid"]),
    yaxis=dict(title="Taxa de Adoção (%)", gridcolor=COLORS["grid"], range=[0, 115]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=20)
)

monthly = df_all[df_all["animal_type"].isin(["Dog", "Cat"])].copy()
monthly["quarter"] = monthly["datetime"].dt.to_period("Q").astype(str)
quarterly_grouped = monthly.groupby(["quarter", "animal_type"])["is_adopted"].sum().reset_index()
quarterly_grouped.columns = ["quarter", "animal_type", "adoptions"]

fig_time = go.Figure()
for species, color, name in [("Dog", COLORS["dog"], "🐕 Cães"), ("Cat", COLORS["cat"], "🐱 Gatos")]:
    subset = quarterly_grouped[quarterly_grouped["animal_type"] == species].sort_values("quarter")
    fig_time.add_trace(go.Scatter(
        x=subset["quarter"], y=subset["adoptions"],
        mode="lines+markers", name=name,
        line=dict(color=color, width=2),
        marker=dict(size=4),
    ))

fig_time.update_layout(
    **PLOT_LAYOUT, height=350,
    xaxis=dict(gridcolor=COLORS["grid"], tickangle=-45, dtick=2),
    yaxis=dict(title="Adoções por Trimestre", gridcolor=COLORS["grid"]),
    legend=dict(x=0.01, y=0.95),
    hovermode="x unified",
)


def build_breed_table(df):
    stats = df.groupby("breed_pt").agg(
        total=("is_adopted", "count"),
        adotados=("is_adopted", "sum"),
        taxa_adocao=("is_adopted", "mean"),
        prob_adocao=("adoption_score", "mean"),
    ).reset_index()
    stats["taxa_adocao"] = (stats["taxa_adocao"] * 100).round(1)
    stats["prob_adocao"] = stats["prob_adocao"].round(1)
    stats = stats.sort_values("prob_adocao", ascending=False)
    stats.columns = ["Raça", "Total", "Adotados", "Taxa Adoção Real (%)", "Prob. Adoção - Modelo (%)"]
    return stats


breed_table_dogs = build_breed_table(df_dogs)
breed_table_cats = build_breed_table(df_cats)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1(
                            "🐾 Panorama da Adoção Animal",
                            style={
                                "fontWeight": "800", "fontSize": "26px",
                                "color": COLORS["text"], "margin": "0",
                                "background": f"linear-gradient(135deg, {COLORS['dog']}, {COLORS['cat']})",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                            },
                        ),
                        html.P(
                            "Austin Animal Center · Dados históricos de adoção · Comparativo Cães vs Gatos",
                            style={"color": COLORS["text_muted"], "fontSize": "13px", "margin": "4px 0 0 0"},
                        ),
                    ],
                    style={"padding": "20px 0 16px 0"},
                ),
            ),
        ),

        dbc.Row(
            [
                dbc.Col(kpi_mini(
                    "📊", "Total de Animais",
                    f"{total_animals:,}".replace(",", "."),
                    COLORS["text"],
                    f"🐕 {total_dogs:,.0f} cães · 🐱 {total_cats:,.0f} gatos".replace(",", "."),
                ), md=3, sm=6, xs=12, className="mb-3"),
                dbc.Col(kpi_mini(
                    "💚", "Taxa de Adoção",
                    f"{adoption_rate_all:.1f}%",
                    COLORS["accent_green"],
                    f"🐕 {adoption_rate_dogs:.1f}% · 🐱 {adoption_rate_cats:.1f}%",
                ), md=3, sm=6, xs=12, className="mb-3"),
                dbc.Col(kpi_mini(
                    "🤖", "Raça + Provável (Modelo)",
                    f"{top_prob_dog_val:.0f}%",
                    COLORS["accent_purple"],
                    f"🐕 {top_prob_dog} · 🐱 {top_prob_cat}",
                ), md=3, sm=6, xs=12, className="mb-3"),
                dbc.Col(kpi_mini(
                    "🏷️", "Raças Identificadas",
                    f"{total_breeds_dogs + total_breeds_cats}",
                    COLORS["cat"],
                    f"🐕 {total_breeds_dogs} raças · 🐱 {total_breeds_cats} raças",
                ), md=3, sm=6, xs=12, className="mb-3"),
            ],
        ),

        html.Div(
            html.P(
                [
                    "O Austin Animal Center registrou dados de mais de ",
                    html.Strong(f"{total_animals:,}".replace(",", ".")),
                    " animais. Com uma taxa de adoção geral de ",
                    html.Strong(f"{adoption_rate_all:.1f}%"),
                    ", o abrigo demonstra um compromisso significativo com o bem-estar animal. ",
                    "Abaixo, exploramos as diferenças entre cães e gatos — ",
                    "quais raças são mais adotadas, e quais fatores influenciam a adoção.",
                ],
                style={"color": COLORS["text_muted"], "fontSize": "13px", "lineHeight": "1.6", "padding": "8px 0 0 0"},
            ),
            style={**CARD_STYLE, "padding": "16px 20px", "marginBottom": "8px", "borderLeft": f"3px solid {COLORS['accent_green']}"},
        ),

        section_title("🏅", "Comparativo de Raças", "Quais raças têm maior taxa de adoção? A análise revela diferenças marcantes entre cães e gatos."),

        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Graph(figure=fig_breeds, config={"displayModeBar": False, "staticPlot": False}),
                    style=CARD_STYLE,
                ),
                xs=12, className="mb-3",
            ),
        ),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("🐕", style={"fontSize": "24px", "marginRight": "10px"}),
                                    html.Span("CÃES", style={"fontSize": "16px", "fontWeight": "700", "color": COLORS["dog"], "letterSpacing": "2px"}),
                                ],
                                style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([
                                        html.Div("Total", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{total_dogs:,}".replace(",", "."), style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["dog"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Taxa Adoção", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{adoption_rate_dogs:.1f}%", style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["accent_green"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Prob. Média", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{score_dogs:.1f}%", style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["accent_purple"]}),
                                    ], width=4),
                                ],
                            ),
                        ],
                        style={**CARD_STYLE, "borderTop": f"3px solid {COLORS['dog']}"},
                    ),
                    md=6, xs=12, className="mb-3",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("🐱", style={"fontSize": "24px", "marginRight": "10px"}),
                                    html.Span("GATOS", style={"fontSize": "16px", "fontWeight": "700", "color": COLORS["cat"], "letterSpacing": "2px"}),
                                ],
                                style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([
                                        html.Div("Total", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{total_cats:,}".replace(",", "."), style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["cat"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Taxa Adoção", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{adoption_rate_cats:.1f}%", style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["accent_green"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Prob. Média", style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase"}),
                                        html.Div(f"{score_cats:.1f}%", style={"fontSize": "22px", "fontWeight": "700", "color": COLORS["accent_purple"]}),
                                    ], width=4),
                                ],
                            ),
                        ],
                        style={**CARD_STYLE, "borderTop": f"3px solid {COLORS['cat']}"},
                    ),
                    md=6, xs=12, className="mb-3",
                ),
            ],
        ),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H5("Desfechos: Para onde vão os animais?", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "4px"}),
                        html.P("A adoção é o desfecho mais frequente, mas transferências e devoluções também são significativas.",
                               style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "8px"}),
                        dcc.Graph(figure=fig_outcomes, config={"displayModeBar": False}),
                    ],
                    style=CARD_STYLE,
                ),
                xs=12, className="mb-3",
            ),
        ),

        section_title("🔬", "Fatores que Influenciam a Adoção", "Idade, castração e condição de saúde impactam diretamente as chances de adoção. Veja como cada fator se comporta para cães e gatos."),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H6("📅 Adoção por Faixa Etária", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "8px"}),
                            dcc.Graph(figure=fig_age, config={"displayModeBar": False}),
                            html.P("💡 Filhotes têm a maior taxa de adoção. Animais idosos enfrentam mais dificuldade.",
                                   style={"color": COLORS["text_dim"], "fontSize": "11px", "marginTop": "4px"}),
                        ],
                        style=CARD_STYLE,
                    ),
                    lg=4, md=12, xs=12, className="mb-3",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H6("✂️ Impacto da Castração", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "8px"}),
                            dcc.Graph(figure=fig_neuter, config={"displayModeBar": False}),
                            html.P("💡 A castração é o fator mais forte. Animais castrados têm até 10x mais chance de adoção.",
                                   style={"color": COLORS["text_dim"], "fontSize": "11px", "marginTop": "4px"}),
                        ],
                        style=CARD_STYLE,
                    ),
                    lg=4, md=12, xs=12, className="mb-3",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H6("🏥 Condição de Saúde", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "8px"}),
                            dcc.Graph(figure=fig_health, config={"displayModeBar": False}),
                            html.P("💡 Animais com condição especial (foster/sick) têm taxa de adoção maior — geralmente via programas específicos.",
                                   style={"color": COLORS["text_dim"], "fontSize": "11px", "marginTop": "4px"}),
                        ],
                        style=CARD_STYLE,
                    ),
                    lg=4, md=12, xs=12, className="mb-3",
                ),
            ],
        ),

        section_title("📈", "Evolução Temporal das Adoções", "Como o volume de adoções variou ao longo dos anos? Acompanhe a tendência trimestral."),

        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Graph(figure=fig_time, config={"displayModeBar": False, "staticPlot": False}),
                    style=CARD_STYLE,
                ),
                xs=12, className="mb-3",
            ),
        ),

        section_title("🏆", "Ranking Completo de Raças", "Explore todas as raças com seus indicadores. Use as abas para alternar entre cães e gatos."),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="🐕 Cães", tab_id="tab-dogs", label_style={"fontWeight": "600"}),
                                dbc.Tab(label="🐱 Gatos", tab_id="tab-cats", label_style={"fontWeight": "600"}),
                            ],
                            id="breed-tabs",
                            active_tab="tab-dogs",
                            style={"marginBottom": "16px"},
                        ),
                        html.Div(id="breed-table-container"),
                    ],
                    style=CARD_STYLE,
                ),
                xs=12, className="mb-3",
            ),
        ),

        dbc.Row(
            dbc.Col(
                html.Div(
                    html.P(
                        "Austin Animal Center · Projeto Preditivo de Adotabilidade Animal · Dados Históricos Públicos",
                        style={"color": COLORS["text_dim"], "fontSize": "11px", "textAlign": "center", "margin": "24px 0 12px 0"},
                    ),
                ),
            ),
        ),
    ],
    fluid=True,
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "16px 32px"},
)


@app.callback(
    Output("breed-table-container", "children"),
    Input("breed-tabs", "active_tab"),
)
def update_breed_table(active_tab):
    if active_tab == "tab-dogs":
        data = breed_table_dogs
        header_color = COLORS["dog"]
    else:
        data = breed_table_cats
        header_color = COLORS["cat"]

    return dash_table.DataTable(
        data=data.to_dict("records"),
        columns=[{"name": c, "id": c} for c in data.columns],
        style_header={
            "backgroundColor": header_color,
            "color": "white",
            "fontWeight": "600",
            "fontSize": "12px",
            "textAlign": "center",
            "border": "none",
            "padding": "10px 8px",
        },
        style_cell={
            "backgroundColor": COLORS["card_bg"],
            "color": COLORS["text"],
            "fontSize": "12px",
            "textAlign": "center",
            "padding": "8px",
            "border": f"1px solid {COLORS['card_border']}",
            "maxWidth": "250px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Raça"}, "textAlign": "left", "maxWidth": "300px"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#0d1321"},
            {
                "if": {"filter_query": "{Taxa Adoção Real (%)} >= 60", "column_id": "Taxa Adoção Real (%)"},
                "color": COLORS["accent_green"], "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{Taxa Adoção Real (%)} < 30", "column_id": "Taxa Adoção Real (%)"},
                "color": COLORS["accent_red"], "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{Prob. Adoção - Modelo (%)} >= 60", "column_id": "Prob. Adoção - Modelo (%)"},
                "color": COLORS["accent_green"], "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{Prob. Adoção - Modelo (%)} < 30", "column_id": "Prob. Adoção - Modelo (%)"},
                "color": COLORS["accent_red"], "fontWeight": "bold",
            },
        ],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_filter={
            "backgroundColor": "#0d1321",
            "color": COLORS["text"],
            "border": f"1px solid {COLORS['card_border']}",
        },
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🐾 Dashboard Adote — Panorama da Adoção Animal")
    print("=" * 60)
    print(f"📊 Dados carregados: {len(df_all):,} registros")
    print(f"🐕 Cães: {len(df_dogs):,} | 🐱 Gatos: {len(df_cats):,}")
    print(f"🌐 Acesse: http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=8050)
