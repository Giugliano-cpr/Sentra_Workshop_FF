"""
SENTRA – KI-gestützte Anomalieerkennung in Beschaffungstransaktionen.

Kernidee:
1. Erklärbare Regeln erkennen bekannte Auffälligkeiten im Einkauf.
2. Isolation Forest erkennt unbekannte, multivariate Ausreißer.
3. Beide Ergebnisse werden in Priorität, Risikostufe und Handlungsempfehlung übersetzt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

TARGET_FIELDS = [
    "rechnung_id", "bestelldatum", "lieferant", "kategorie", "menge",
    "stueckpreis", "gesamtbetrag", "lieferzeit_tage",
]
NUMERIC_FIELDS = ["menge", "stueckpreis", "gesamtbetrag", "lieferzeit_tage"]
DEFAULT_FEATURES = ["stueckpreis", "menge", "gesamtbetrag", "lieferzeit_tage"]


def has_columns(df: pd.DataFrame, *columns: str) -> bool:
    return all(col in df.columns for col in columns)


def standardize_input(df_raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Mappt frei benannte CSV-Spalten auf das interne Schema."""
    out = pd.DataFrame(index=df_raw.index)
    for target, source in mapping.items():
        if source and source in df_raw.columns:
            out[target] = df_raw[source]

    for col in NUMERIC_FIELDS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "gesamtbetrag" not in out.columns and has_columns(out, "menge", "stueckpreis"):
        out["gesamtbetrag"] = (out["menge"] * out["stueckpreis"]).round(2)
    if "stueckpreis" not in out.columns and has_columns(out, "menge", "gesamtbetrag"):
        out["stueckpreis"] = (out["gesamtbetrag"] / out["menge"].replace(0, np.nan)).round(2)

    if "rechnung_id" not in out.columns:
        out["rechnung_id"] = [f"ROW-{i+1}" for i in range(len(out))]
    if "lieferant" not in out.columns:
        out["lieferant"] = "Unbekannt"
    if "kategorie" not in out.columns:
        out["kategorie"] = "Alle"

    return out


def rule_duplicate_invoices(df: pd.DataFrame, days_window: int = 5) -> pd.Series:
    """Gleicher Lieferant + gleicher Betrag innerhalb eines kurzen Zeitfensters."""
    if not has_columns(df, "lieferant", "gesamtbetrag", "bestelldatum"):
        return pd.Series(False, index=df.index)
    dates = pd.to_datetime(df["bestelldatum"], errors="coerce")
    hit = pd.Series(False, index=df.index)
    for _, idx in df.groupby(["lieferant", "gesamtbetrag"]).groups.items():
        if len(idx) < 2:
            continue
        part = dates.loc[idx].dropna().sort_values()
        for date_idx, date_value in part.items():
            near = (part - date_value).abs().dt.days <= days_window
            if near.sum() >= 2:
                hit.loc[part.index[near]] = True
    return hit


def rule_price_deviation(df: pd.DataFrame, threshold_pct: float = 70.0) -> pd.Series:
    """Stückpreis weicht stark vom Median der Kategorie ab."""
    if not has_columns(df, "stueckpreis", "kategorie"):
        return pd.Series(False, index=df.index)
    price = pd.to_numeric(df["stueckpreis"], errors="coerce")
    med = price.groupby(df["kategorie"]).transform("median").replace(0, np.nan)
    deviation = (price - med).abs() / med * 100
    return deviation > threshold_pct


def rule_just_below_limit(df: pd.DataFrame, limit: float = 10_000.0, margin_pct: float = 5.0) -> pd.Series:
    """Betrag liegt knapp unter einer Genehmigungsgrenze."""
    if not has_columns(df, "gesamtbetrag"):
        return pd.Series(False, index=df.index)
    amount = pd.to_numeric(df["gesamtbetrag"], errors="coerce")
    lower = limit * (1 - margin_pct / 100)
    return (amount >= lower) & (amount < limit)


def rule_order_splitting(df: pd.DataFrame, limit: float = 10_000.0, days_window: int = 7) -> pd.Series:
    """Mehrere Bestellungen je unter Grenze, aber gemeinsam über Grenze."""
    if not has_columns(df, "lieferant", "kategorie", "gesamtbetrag", "bestelldatum"):
        return pd.Series(False, index=df.index)
    dates = pd.to_datetime(df["bestelldatum"], errors="coerce")
    amount = pd.to_numeric(df["gesamtbetrag"], errors="coerce")
    hit = pd.Series(False, index=df.index)

    for _, idx in df.groupby(["lieferant", "kategorie"]).groups.items():
        valid = [i for i in idx if pd.notna(dates.loc[i]) and pd.notna(amount.loc[i]) and amount.loc[i] < limit]
        if len(valid) < 2:
            continue
        valid = sorted(valid, key=lambda i: dates.loc[i])
        for start_idx in valid:
            start = dates.loc[start_idx]
            window = [j for j in valid if 0 <= (dates.loc[j] - start).days <= days_window]
            if len(window) >= 2 and amount.loc[window].sum() > limit:
                hit.loc[window] = True
    return hit


def apply_rules(
    df: pd.DataFrame,
    days_window_duplicate: int = 5,
    price_threshold_pct: float = 70.0,
    approval_limit: float = 10_000.0,
    limit_margin_pct: float = 5.0,
    split_window: int = 7,
) -> pd.DataFrame:
    out = df.copy()
    out["regel_doppelte_rechnung"] = rule_duplicate_invoices(out, days_window_duplicate)
    out["regel_preisabweichung"] = rule_price_deviation(out, price_threshold_pct)
    out["regel_knapp_unter_grenze"] = rule_just_below_limit(out, approval_limit, limit_margin_pct)
    out["regel_auftragssplitting"] = rule_order_splitting(out, approval_limit, split_window)
    rule_cols = [
        "regel_doppelte_rechnung", "regel_preisabweichung",
        "regel_knapp_unter_grenze", "regel_auftragssplitting",
    ]
    out["anzahl_regel_treffer"] = out[rule_cols].sum(axis=1).astype(int)
    out["regel_treffer"] = out["anzahl_regel_treffer"] > 0
    return out


def apply_ml(
    df: pd.DataFrame,
    features: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    features = DEFAULT_FEATURES if features is None else features
    features = [feature for feature in features if feature in out.columns]
    if not features:
        out["ml_score"] = 0.0
        out["ml_anomalie"] = False
        out["ml_features"] = ""
        return out

    X = out[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X_scaled = StandardScaler().fit_transform(X.values)
    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    out["ml_score"] = model.decision_function(X_scaled)
    out["ml_anomalie"] = labels == -1
    out["ml_features"] = ", ".join(features)
    return out


def _reasons(row: pd.Series) -> str:
    reasons = []
    if row.get("regel_doppelte_rechnung", False):
        reasons.append("Mögliche doppelte Rechnung")
    if row.get("regel_preisabweichung", False):
        reasons.append("Starke Preisabweichung")
    if row.get("regel_knapp_unter_grenze", False):
        reasons.append("Knapp unter Genehmigungsgrenze")
    if row.get("regel_auftragssplitting", False):
        reasons.append("Mögliches Auftragssplitting")
    if row.get("ml_anomalie", False):
        reasons.append("ML-Ausreißer")
    return "; ".join(reasons) if reasons else "Keine Auffälligkeit"


def _category(row: pd.Series) -> str:
    rule = bool(row["regel_treffer"])
    ml = bool(row["ml_anomalie"])
    if rule and ml:
        return "Regel + ML"
    if rule:
        return "Nur Regel"
    if ml:
        return "Nur ML"
    return "Unauffällig"


def _priority(row: pd.Series) -> int:
    # 0 bis 100: verständlicher Business-Prioritätswert.
    score = 0
    score += int(row.get("anzahl_regel_treffer", 0)) * 18
    if row.get("ml_anomalie", False):
        score += 28
    if row.get("regel_auftragssplitting", False):
        score += 22
    if row.get("regel_doppelte_rechnung", False):
        score += 18
    if row.get("regel_knapp_unter_grenze", False):
        score += 15
    if row.get("regel_preisabweichung", False):
        score += 12
    return int(min(score, 100))


def _risk_level(priority: int) -> str:
    if priority >= 70:
        return "Hoch"
    if priority >= 35:
        return "Mittel"
    if priority > 0:
        return "Niedrig"
    return "Keine"


def _recommendation(row: pd.Series) -> str:
    if row.get("regel_auftragssplitting", False):
        return "Freigabeprozess und Bestellhistorie prüfen"
    if row.get("regel_doppelte_rechnung", False):
        return "Rechnung vor Zahlung manuell gegenprüfen"
    if row.get("regel_knapp_unter_grenze", False):
        return "Genehmigungsgrenze und Bestellbegründung prüfen"
    if row.get("regel_preisabweichung", False):
        return "Preis mit Kategorie-Median und Vertrag vergleichen"
    if row.get("ml_anomalie", False):
        return "Vorgang fachlich plausibilisieren"
    return "Keine Maßnahme erforderlich"


def analyze(
    df: pd.DataFrame,
    days_window_duplicate: int = 5,
    price_threshold_pct: float = 70.0,
    approval_limit: float = 10_000.0,
    limit_margin_pct: float = 5.0,
    split_window: int = 7,
    features: list[str] | None = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    out = apply_rules(
        df,
        days_window_duplicate=days_window_duplicate,
        price_threshold_pct=price_threshold_pct,
        approval_limit=approval_limit,
        limit_margin_pct=limit_margin_pct,
        split_window=split_window,
    )
    out = apply_ml(out, features=features, contamination=contamination)
    out["erkennungsquelle"] = out.apply(_category, axis=1)
    out["auffaelligkeit_grund"] = out.apply(_reasons, axis=1)
    out["prioritaet_score"] = out.apply(_priority, axis=1)
    out["risikostufe"] = out["prioritaet_score"].apply(_risk_level)
    out["empfohlene_massnahme"] = out.apply(_recommendation, axis=1)
    return out


def kpis(df: pd.DataFrame) -> dict[str, int | float]:
    suspicious = df[df["erkennungsquelle"] != "Unauffällig"]
    return {
        "gesamt": int(len(df)),
        "auffaellig": int(len(suspicious)),
        "regel_treffer": int(df["regel_treffer"].sum()),
        "ml_anomalien": int(df["ml_anomalie"].sum()),
        "beide": int(((df["regel_treffer"]) & (df["ml_anomalie"])).sum()),
        "wert_auffaellig": float(suspicious["gesamtbetrag"].sum()) if "gesamtbetrag" in suspicious.columns else 0.0,
        "hoch": int((df["risikostufe"] == "Hoch").sum()),
    }


# Rückwärtskompatibilität zu älteren Dateien
kombinierte_analyse = analyze
kennzahlen = kpis
standardisiere = standardize_input
STANDARD_FEATURES = DEFAULT_FEATURES

if __name__ == "__main__":
    data = pd.read_csv("data/beschaffungsdaten.csv")
    result = analyze(data)
    print(kpis(result))
    print(result["erkennungsquelle"].value_counts())
    if "anomalie_typ" in result.columns:
        print(pd.crosstab(result["anomalie_typ"], result["erkennungsquelle"]))
