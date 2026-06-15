# SENTRA – KI-gestützte Anomalieerkennung in Beschaffungstransaktionen

Prototyp für den Workshop **Digital Procurement & Data Science**.

## Thema

**KI-gestützte Erkennung von Anomalien in Beschaffungstransaktionen mittels Python und Power BI**

## Ziel

Das System erkennt auffällige Beschaffungsvorgänge im Einkaufsprozess. Dafür werden zwei Ansätze kombiniert:

1. **Regelbasierte Erkennung** für bekannte Muster:
   - doppelte Rechnungen
   - starke Preisabweichungen vom Kategorie-Median
   - Bestellungen knapp unter einer Genehmigungsgrenze
   - mögliches Auftragssplitting
2. **Machine Learning** mit Isolation Forest:
   - Erkennung unbekannter, multivariater Ausreißer
   - keine gelabelten Anomalien erforderlich

Die Ergebnisse werden in eine **Risikostufe**, einen **Prioritätsscore** und eine **empfohlene Maßnahme** übersetzt.

## Projektstruktur

```text
sentra_reworked/
├── app.py                 # Streamlit-Oberfläche
├── detection.py           # Regeln + Isolation Forest + Priorisierung
├── generate_data.py       # synthetischer Beispieldatensatz
├── export_for_powerbi.py  # Export als CSV für Power BI
├── requirements.txt
└── data/
    ├── beschaffungsdaten.csv
    └── sentra_scored.csv
```

## Installation auf Mac

```bash
cd sentra_reworked
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

## Power BI auf Mac

Power BI Desktop läuft nicht nativ auf macOS. Daher gibt es drei praktikable Wege:

1. **Streamlit als Demo nutzen**: Die App läuft vollständig im Browser und deckt Analyse + Visualisierung ab.
2. **CSV exportieren und auf Windows öffnen**:
   ```bash
   python export_for_powerbi.py
   ```
   Danach `data/sentra_scored.csv` in Power BI Desktop importieren.
3. **Power BI Service im Browser nutzen**: CSV in Power BI Web hochladen und dort einfache Berichte erstellen.

## Export für Power BI

```bash
python export_for_powerbi.py
```

Ergebnis:

```text
data/sentra_scored.csv
```

Diese Datei enthält u. a.:

- Erkennungsquelle
- Risikostufe
- Prioritätsscore
- Grund der Auffälligkeit
- empfohlene Maßnahme
- Regel-Flags
- ML-Score

## Präsentationsargumentation

**Warum Regeln + KI?**

Regeln sind transparent und erkennen bekannte Muster zuverlässig. Der Isolation Forest ergänzt diese Regeln, indem er unbekannte, ungewöhnliche Kombinationen erkennt. Dadurch entsteht ein hybrides System mit Erklärbarkeit und Data-Science-Komponente.

**Warum Power BI?**

Python übernimmt Analyse und Berechnung. Power BI dient der managementgerechten Visualisierung und Entscheidungsunterstützung.
