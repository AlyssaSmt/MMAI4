# MMAI – Open Vocabulary Montagsmaler 🎨🤖

Ein Zeichen-Spiel mit KI:  
Du zeichnest auf einem Canvas und eine **Open-Vocabulary-KI (OpenCLIP)** versucht zu erraten, was du gemalt hast.  
Es gibt Live-Prediction (optional), zufällige Zeichen-Prompts aus dem Vokabular und eine Galerie zum Speichern der Zeichnungen.

---

## Funktionen

- ✏️ Zeichnen auf Canvas
- 🤖 KI-Erkennung (Open Vocabulary, keine festen Klassen)
- 👁 Live-Prediction (ein/aus schaltbar)
- 🎲 Zufälliges Zeichen-Prompt aus `vocab.txt`
- 🔀 Shuffle-Button für neues Wort
- 💾 Galerie mit gespeicherten Zeichnungen (LocalStorage)
- 🗑️ Löschen einzelner Galerie-Einträge

---

## Voraussetzungen

- **Python 3.10 – 3.12** (empfohlen)
- **pip**
- Optional: **Git**

> Hinweis:  
> Python 3.13 kann bei ML-Bibliotheken Probleme machen.  
> Falls etwas nicht installiert werden kann, nutze Python 3.11 oder 3.12.

---
## Konzept / Erklärung (kurz)

Dieses Projekt nutzt **Open-Vocabulary-Erkennung** mit **OpenCLIP**:
- Es gibt **keine festen Klassen** wie bei einem klassisch trainierten CNN.
- Stattdessen wird die Zeichnung mit **Textbeschreibungen aus `vocab.txt`** verglichen.
- Die KI berechnet Ähnlichkeiten zwischen Bild-Embedding und Text-Embeddings und gibt die wahrscheinlichsten Begriffe zurück.

Zusatzfeatures:
- **Live-Prediction** ist bewusst gedrosselt, da OpenCLIP pro Vorhersage viele Textvergleiche berechnet.
- Eine **Confidence-Ampel (🔴🟡🟢)** visualisiert die Unsicherheit.
- Ein **Prompt-Wort** wird zufällig aus dem Vokabular gewählt und dient nur als Zeichenhilfe (nicht als feste Klasse).

---

## bash:

### 1. Virtuelle Umgebung erstellen
```bash
python -m venv .venv

### 2. .venv\Scripts\activate  (mac: source .venv/bin/activate)

### 3. Abhängigkeiten installieren

    pip install -r requirements.txt


### 4. Backend starten (FastAPI)

    uvicorn backend.main:app --reload --port 8001

Test (optional):
    Browser öffnen:
    http://127.0.0.1:8001/docs

### 5. Frontend starten
    frontend/index.html



open vocabulary
ganze clip library verwenden
website verbessern, das es genauer/verständlicher ist












Musste neu trainieren, weil es immer alles als string bean gesehen hat, hab dann string bean gelöscht

ebenfalls falsche daten benutzt, die nicht nur das bild sondern auch viel freiraum hatten.


Nicht alle Klassen sind für kleine CNNs geeignet.
Klassen mit ähnlicher geometrischer Struktur
führten zu systematischen Fehlklassifikationen.
Durch gezielte Klassenselektion mit hoher visueller Varianz
konnte das Modell stabilisiert werden

sehr limitiert, da es immer nur das gleiche errät