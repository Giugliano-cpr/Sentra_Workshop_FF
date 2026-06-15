"""
Erzeugt einen synthetischen Beschaffungsdatensatz für SENTRA.
Der Datensatz enthält normale Bestellungen sowie gezielt eingebaute Auffälligkeiten.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SUPPLIERS = [
    "Alpha Components GmbH", "Beta Metalltechnik", "Gamma Logistik AG",
    "Delta Elektronik", "Epsilon Rohstoffe", "Zeta Verpackung",
    "Eta Industriebedarf", "Theta Werkzeuge", "Iota Plastics", "Kappa Services",
]
CATEGORIES = {
    "Elektronik": (120.0, 25.0),
    "Rohmaterial": (8.0, 1.5),
    "Verpackung": (2.5, 0.4),
    "Werkzeuge": (45.0, 10.0),
    "Logistik": (300.0, 60.0),
    "Bueromaterial": (15.0, 3.0),
}
START_DATE = pd.Timestamp("2025-01-01")
APPROVAL_LIMIT = 10_000.0

def random_date() -> pd.Timestamp:
    return START_DATE + pd.Timedelta(days=int(rng.integers(0, 365)))

def base_rows(n: int = 750) -> pd.DataFrame:
    rows = []
    for i in range(n):
        category = rng.choice(list(CATEGORIES.keys()))
        mean, std = CATEGORIES[category]
        unit_price = max(0.5, rng.normal(mean, std))
        quantity = int(rng.integers(1, 220))
        rows.append({
            "rechnung_id": f"INV-{100000+i}",
            "bestell_id": f"PO-{500000+i}",
            "bestelldatum": random_date(),
            "lieferant": rng.choice(SUPPLIERS),
            "kategorie": category,
            "material_id": f"MAT-{int(rng.integers(1000, 1120))}",
            "menge": quantity,
            "stueckpreis": round(unit_price, 2),
            "lieferzeit_tage": int(np.clip(rng.normal(14, 5), 1, 65)),
            "anomalie_typ": "normal",
        })
    df = pd.DataFrame(rows)
    df["gesamtbetrag"] = (df["menge"] * df["stueckpreis"]).round(2)
    return df

def inject_price_outliers(df: pd.DataFrame, n: int = 28) -> pd.DataFrame:
    idxs = rng.choice(df.index, size=n, replace=False)
    for idx in idxs:
        factor = rng.choice([rng.uniform(2.8, 6.0), rng.uniform(0.08, 0.35)])
        df.loc[idx, "stueckpreis"] = round(float(df.loc[idx, "stueckpreis"]) * factor, 2)
        df.loc[idx, "gesamtbetrag"] = round(float(df.loc[idx, "stueckpreis"]) * int(df.loc[idx, "menge"]), 2)
        df.loc[idx, "anomalie_typ"] = "preisabweichung"
    return df

def inject_duplicates(df: pd.DataFrame, n: int = 18) -> pd.DataFrame:
    idxs = rng.choice(df.index, size=n, replace=False)
    copies = df.loc[idxs].copy()
    copies["rechnung_id"] = [f"INV-DUP-{i}" for i in range(n)]
    copies["bestelldatum"] = pd.to_datetime(copies["bestelldatum"]) + pd.to_timedelta(rng.integers(0, 4, size=n), unit="D")
    copies["anomalie_typ"] = "doppelte_rechnung"
    return pd.concat([df, copies], ignore_index=True)

def inject_limit_avoidance(df: pd.DataFrame, n: int = 16) -> pd.DataFrame:
    idxs = rng.choice(df.index, size=n, replace=False)
    for idx in idxs:
        amount = round(rng.uniform(APPROVAL_LIMIT * 0.952, APPROVAL_LIMIT * 0.997), 2)
        quantity = max(1, int(df.loc[idx, "menge"]))
        df.loc[idx, "gesamtbetrag"] = amount
        df.loc[idx, "stueckpreis"] = round(amount / quantity, 2)
        df.loc[idx, "anomalie_typ"] = "knapp_unter_grenze"
    return df

def inject_splitting(df: pd.DataFrame, groups: int = 7) -> pd.DataFrame:
    rows = []
    running = 880000
    for g in range(groups):
        supplier = rng.choice(SUPPLIERS)
        category = rng.choice(list(CATEGORIES.keys()))
        material = f"MAT-{int(rng.integers(1000, 1120))}"
        base = random_date()
        parts = int(rng.integers(3, 5))
        for p in range(parts):
            amount = round(rng.uniform(0.52, 0.86) * APPROVAL_LIMIT, 2)
            quantity = int(rng.integers(20, 130))
            rows.append({
                "rechnung_id": f"INV-SPLIT-{g}-{p}",
                "bestell_id": f"PO-{running}",
                "bestelldatum": base + pd.Timedelta(days=int(rng.integers(0, 6))),
                "lieferant": supplier,
                "kategorie": category,
                "material_id": material,
                "menge": quantity,
                "stueckpreis": round(amount / quantity, 2),
                "gesamtbetrag": amount,
                "lieferzeit_tage": int(np.clip(rng.normal(14, 5), 1, 65)),
                "anomalie_typ": "auftragssplitting",
            })
            running += 1
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

def inject_multivariate(df: pd.DataFrame, n: int = 22) -> pd.DataFrame:
    rows = []
    for i in range(n):
        category = rng.choice(list(CATEGORIES.keys()))
        mean, _ = CATEGORIES[category]
        quantity = int(rng.integers(380, 850))
        unit_price = round(mean * rng.uniform(1.6, 2.7), 2)
        rows.append({
            "rechnung_id": f"INV-ML-{i}",
            "bestell_id": f"PO-{920000+i}",
            "bestelldatum": random_date(),
            "lieferant": rng.choice(SUPPLIERS),
            "kategorie": category,
            "material_id": f"MAT-{int(rng.integers(1000, 1120))}",
            "menge": quantity,
            "stueckpreis": unit_price,
            "gesamtbetrag": round(quantity * unit_price, 2),
            "lieferzeit_tage": int(rng.integers(42, 66)),
            "anomalie_typ": "multivariater_ausreisser",
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

def main() -> None:
    df = base_rows()
    df = inject_price_outliers(df)
    df = inject_duplicates(df)
    df = inject_limit_avoidance(df)
    df = inject_splitting(df)
    df = inject_multivariate(df)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["bestelldatum"] = pd.to_datetime(df["bestelldatum"]).dt.date
    cols = ["rechnung_id", "bestell_id", "bestelldatum", "lieferant", "kategorie", "material_id", "menge", "stueckpreis", "gesamtbetrag", "lieferzeit_tage", "anomalie_typ"]
    df = df[cols]
    out = DATA_DIR / "beschaffungsdaten.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Datensatz erzeugt: {out} ({len(df)} Zeilen)")
    print(df["anomalie_typ"].value_counts())

if __name__ == "__main__":
    main()
