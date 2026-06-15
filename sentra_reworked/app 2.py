"""SENTRA – Streamlit-App für KI-gestützte Anomalieerkennung im Einkauf."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import detection

PRODUCT = "SENTRA"
TAGLINE = "Procurement Anomaly Intelligence"
DATA_PATH = Path("data/beschaffungsdaten.csv")

st.set_page_config(page_title=f"{PRODUCT} – {TAGLINE}", layout="wide")

RISK_COLORS = {"Keine": "#CBD5E1", "Niedrig": "#94A3B8", "Mittel": "#475569", "Hoch": "#B42318"}
SOURCE_COLORS = {"Unauffällig": "#CBD5E1", "Nur Regel": "#475569", "Nur ML": "#94A3B8", "Regel + ML": "#B42318"}

FIELD_LABELS = {
    "rechnung_id": "Rechnungs-/Beleg-ID",
    "bestelldatum": "Bestelldatum",
    "lieferant": "Lieferant",
    "kategorie": "Kategorie / Warengruppe",
    "menge": "Menge",
    "stueckpreis": "Stückpreis",
    "gesamtbetrag": "Gesamtbetrag",
    "lieferzeit_tage": "Lieferzeit in Tagen",
}
SYNONYMS = {
    "rechnung_id": ["rechnung", "invoice", "beleg", "id"],
    "bestelldatum": ["datum", "date", "bestelldatum", "invoice_date", "order_date"],
    "lieferant": ["lieferant", "supplier", "vendor"],
    "kategorie": ["kategorie", "category", "warengruppe", "material_group"],
    "menge": ["menge", "quantity", "qty", "anzahl"],
    "stueckpreis": ["stückpreis", "stueckpreis", "unit_price", "preis", "price"],
    "gesamtbetrag": ["gesamtbetrag", "betrag", "amount", "spend", "total", "value"],
    "lieferzeit_tage": ["lieferzeit", "lead_time", "delivery_days", "liefertage"],
}


def apply_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, [data-testid="stToolbar"] {display:none;}
        .block-container {max-width: 1240px; padding-top: 1.3rem;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        .hero {border-bottom: 1px solid #E6E8EB; margin-bottom: 1.2rem; padding-bottom: 1rem;}
        .brand {font-size: 2rem; font-weight: 700; letter-spacing: .18em; color: #0F172A;}
        .tag {font-size: .82rem; color: #64748B; text-transform: uppercase; letter-spacing: .16em;}
        .sub {color:#64748B; font-size:.95rem;}
        [data-testid="stMetric"] {background: #fff; border: 1px solid #E6E8EB; border-radius: 10px; padding: .85rem 1rem;}
        [data-testid="stSidebar"] {background:#F6F7F9; border-right: 1px solid #E6E8EB;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF0F2", zeroline=False)
    return fig


@st.cache_data
def load_example() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Beispieldatensatz fehlt. Bitte zuerst `python generate_data.py` ausführen.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def guess_column(target: str, columns: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for word in SYNONYMS.get(target, []):
        for lower, original in lowered.items():
            if word in lower:
                return original
    return None


def existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


apply_css()
st.markdown(f"<div class='hero'><div class='brand'>{PRODUCT}</div><div class='tag'>{TAGLINE}</div></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Datenquelle")
    source = st.radio("Quelle", ["Beispieldatensatz", "Eigene CSV"], label_visibility="collapsed")
    if source == "Eigene CSV":
        uploaded = st.file_uploader("CSV-Datei hochladen", type="csv")
        if uploaded is None:
            st.info("Noch keine Datei hochgeladen. Es wird der Beispieldatensatz verwendet.")
            base_df = load_example()
        else:
            raw_df = pd.read_csv(uploaded)
            st.subheader("Spalten zuordnen")
            none = "(nicht vorhanden)"
            mapping = {}
            for target, label in FIELD_LABELS.items():
                options = [none] + list(raw_df.columns)
                suggestion = guess_column(target, list(raw_df.columns))
                index = options.index(suggestion) if suggestion in options else 0
                selected = st.selectbox(label, options, index=index)
                mapping[target] = None if selected == none else selected
            base_df = detection.standardize_input(raw_df, mapping)
    else:
        base_df = load_example()

    st.header("Regeln")
    days_duplicate = st.slider("Zeitfenster doppelte Rechnung", 0, 30, 5)
    price_threshold = st.slider("Preisabweichung vom Median (%)", 20, 300, 70, step=10)
    approval_limit = st.number_input("Genehmigungsgrenze (€)", min_value=1000, max_value=100000, value=10000, step=1000)
    limit_margin = st.slider("Marge knapp unter Grenze (%)", 1, 20, 5)
    split_window = st.slider("Splitting-Zeitfenster (Tage)", 1, 30, 7)

    st.header("Machine Learning")
    numeric_cols = base_df.select_dtypes("number").columns.tolist()
    default_features = [col for col in detection.DEFAULT_FEATURES if col in numeric_cols]
    features = st.multiselect("Features", numeric_cols, default=default_features or numeric_cols)
    contamination = st.slider("Erwarteter Anomalieanteil", 0.01, 0.20, 0.05, step=0.01)

if not features:
    st.warning("Bitte mindestens ein numerisches Feature für den Isolation Forest auswählen.")
    st.stop()

result = detection.analyze(
    base_df,
    days_window_duplicate=days_duplicate,
    price_threshold_pct=price_threshold,
    approval_limit=float(approval_limit),
    limit_margin_pct=limit_margin,
    split_window=split_window,
    features=features,
    contamination=contamination,
)
summary = detection.kpis(result)
suspicious = result[result["erkennungsquelle"] != "Unauffällig"].copy()

st.markdown("<p class='sub'>Hybrides System aus erklärbaren Einkaufsregeln und Isolation Forest zur Identifikation auffälliger Beschaffungstransaktionen.</p>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Transaktionen", f"{summary['gesamt']}")
m2.metric("Auffällig", f"{summary['auffaellig']}")
m3.metric("Regel-Treffer", f"{summary['regel_treffer']}")
m4.metric("ML-Anomalien", f"{summary['ml_anomalien']}")
m5.metric("Hohe Priorität", f"{summary['hoch']}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Management Dashboard", "Regeln", "Machine Learning", "Prüfliste", "Export & Evaluation"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Auffälligkeiten nach Risikostufe")
        risk_df = result["risikostufe"].value_counts().rename_axis("Risikostufe").reset_index(name="Anzahl")
        fig = px.bar(risk_df, x="Risikostufe", y="Anzahl", color="Risikostufe", color_discrete_map=RISK_COLORS)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        st.subheader("Erkennungsquelle")
        source_df = result["erkennungsquelle"].value_counts().rename_axis("Erkennungsquelle").reset_index(name="Anzahl")
        fig = px.pie(source_df, values="Anzahl", names="Erkennungsquelle", color="Erkennungsquelle", color_discrete_map=SOURCE_COLORS, hole=.45)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top-Lieferanten nach Auffälligkeiten")
        if len(suspicious):
            top = suspicious["lieferant"].value_counts().head(10).rename_axis("Lieferant").reset_index(name="Auffälligkeiten")
            fig = px.bar(top, x="Auffälligkeiten", y="Lieferant", orientation="h")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.caption("Keine Auffälligkeiten vorhanden.")
    with c4:
        st.subheader("Auffälliger Wert nach Kategorie")
        if len(suspicious) and "gesamtbetrag" in suspicious.columns:
            cat = suspicious.groupby("kategorie", as_index=False)["gesamtbetrag"].sum().sort_values("gesamtbetrag", ascending=False)
            fig = px.bar(cat, x="kategorie", y="gesamtbetrag", labels={"gesamtbetrag": "Auffälliger Wert (€)", "kategorie": "Kategorie"})
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.caption("Keine Werte vorhanden.")

with tab2:
    st.subheader("Erklärbare Regel-Treffer")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Doppelte Rechnungen", int(result["regel_doppelte_rechnung"].sum()))
    r2.metric("Preisabweichungen", int(result["regel_preisabweichung"].sum()))
    r3.metric("Knapp unter Grenze", int(result["regel_knapp_unter_grenze"].sum()))
    r4.metric("Auftragssplitting", int(result["regel_auftragssplitting"].sum()))
    rule_hits = result[result["regel_treffer"]].copy()
    cols = existing(rule_hits, ["rechnung_id", "bestelldatum", "lieferant", "kategorie", "menge", "stueckpreis", "gesamtbetrag", "auffaelligkeit_grund", "risikostufe", "empfohlene_massnahme"])
    st.dataframe(rule_hits[cols].sort_values("risikostufe", ascending=False), use_container_width=True)

with tab3:
    st.subheader("Isolation Forest")
    st.caption("Niedrige ML-Scores bedeuten ungewöhnlichere Transaktionen. Das Modell arbeitet unüberwacht und benötigt keine gelabelten Betrugsfälle.")
    fig = px.histogram(result, x="ml_score", color="ml_anomalie", nbins=45, labels={"ml_score": "ML-Score", "ml_anomalie": "ML-Anomalie"})
    st.plotly_chart(style_fig(fig), use_container_width=True)
    if {"menge", "stueckpreis"}.issubset(result.columns):
        fig = px.scatter(result, x="menge", y="stueckpreis", color="erkennungsquelle", color_discrete_map=SOURCE_COLORS, hover_data=existing(result, ["rechnung_id", "lieferant", "kategorie", "gesamtbetrag", "risikostufe"]))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    cols = existing(result, ["rechnung_id", "lieferant", "kategorie", "menge", "stueckpreis", "gesamtbetrag", "lieferzeit_tage", "ml_score", "risikostufe"])
    st.dataframe(result.nsmallest(20, "ml_score")[cols], use_container_width=True)

with tab4:
    st.subheader("Priorisierte Prüfliste")
    if len(suspicious):
        cols = existing(suspicious, ["rechnung_id", "bestelldatum", "lieferant", "kategorie", "gesamtbetrag", "erkennungsquelle", "auffaelligkeit_grund", "prioritaet_score", "risikostufe", "empfohlene_massnahme"])
        st.dataframe(suspicious.sort_values(["prioritaet_score", "gesamtbetrag"], ascending=[False, False])[cols].head(50), use_container_width=True)
    else:
        st.success("Keine auffälligen Vorgänge gefunden.")

with tab5:
    st.subheader("Export")
    csv_full = result.to_csv(index=False).encode("utf-8-sig")
    csv_suspicious = suspicious.to_csv(index=False).encode("utf-8-sig")
    e1, e2 = st.columns(2)
    e1.download_button("Vollständigen Datensatz exportieren", csv_full, "sentra_scored.csv", "text/csv")
    e2.download_button("Nur Auffälligkeiten exportieren", csv_suspicious, "sentra_anomalies.csv", "text/csv")
    if "anomalie_typ" in result.columns:
        st.subheader("Plausibilitätscheck mit synthetischer Ground Truth")
        st.caption("Diese Tabelle dient nur beim Beispieldatensatz zur Evaluation. Bei echten Unternehmensdaten existiert diese Ground-Truth-Spalte nicht.")
        st.dataframe(pd.crosstab(result["anomalie_typ"], result["erkennungsquelle"]), use_container_width=True)
