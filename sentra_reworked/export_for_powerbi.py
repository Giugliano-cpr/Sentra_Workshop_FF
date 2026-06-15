"""Exportiert die bewerteten Beschaffungstransaktionen für Power BI."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import detection

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/beschaffungsdaten.csv")
OUTPUT = Path("data/sentra_scored.csv")


def main() -> None:
    df = pd.read_csv(INPUT)
    result = detection.analyze(df)
    rename = {
        "rechnung_id": "Rechnung_ID",
        "bestell_id": "Bestell_ID",
        "bestelldatum": "Bestelldatum",
        "lieferant": "Lieferant",
        "kategorie": "Kategorie",
        "material_id": "Material_ID",
        "menge": "Menge",
        "stueckpreis": "Stueckpreis",
        "gesamtbetrag": "Gesamtbetrag",
        "lieferzeit_tage": "Lieferzeit_Tage",
        "regel_doppelte_rechnung": "Flag_Doppelte_Rechnung",
        "regel_preisabweichung": "Flag_Preisabweichung",
        "regel_knapp_unter_grenze": "Flag_Knapp_Unter_Grenze",
        "regel_auftragssplitting": "Flag_Auftragssplitting",
        "regel_treffer": "Flag_Regel",
        "ml_anomalie": "Flag_ML",
        "ml_score": "ML_Score",
        "erkennungsquelle": "Erkennungsquelle",
        "auffaelligkeit_grund": "Auffaelligkeit_Grund",
        "prioritaet_score": "Prioritaet_Score",
        "risikostufe": "Risikostufe",
        "empfohlene_massnahme": "Empfohlene_Massnahme",
    }
    result = result.rename(columns=rename)
    OUTPUT.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Export geschrieben: {OUTPUT} ({len(result)} Zeilen, {len(result.columns)} Spalten)")


if __name__ == "__main__":
    main()
