# MMAI4 – Open Vocabulary Montagsmaler 🎨🤖

Ein Zeichen-Spiel mit KI:  
Der Benutzer zeichnet auf einem Canvas, und eine **Open-Vocabulary-KI (OpenCLIP)** versucht zu erraten, was gezeichnet wurde.

Im Gegensatz zu den vorherigen Versionen gibt es **keine festen Klassen** mehr.  
Stattdessen wird die Zeichnung mit frei definierbaren Begriffen aus einer Wortliste verglichen.

---

## Funktionen

- ✏️ Zeichnen auf einem Canvas (Browser)
- 🤖 KI-Erkennung mit Open Vocabulary (OpenCLIP)
- 👁 Live-Prediction (ein- und ausschaltbar)
- 🎲 Zufälliges Zeichen-Prompt aus dem Vokabular
- 🔀 Shuffle-Button für neues Zeichenwort
- 💾 Galerie mit gespeicherten Zeichnungen (LocalStorage)
- 🗑️ Löschen einzelner Galerie-Einträge

---

## Voraussetzungen

- **Python 3.10 – 3.12** (empfohlen)
- **pip**
- Optional: **Git**

---

## Konzept / Erklärung (kurz)

Dieses Projekt nutzt **Open-Vocabulary-Erkennung** mit **OpenCLIP**:

- Es existieren **keine festen Klassen** wie bei einem klassisch trainierten CNN.
- Stattdessen wird die Zeichnung mit **Textbeschreibungen aus einer Wortliste (`vocab.txt` / `categories.txt`)** verglichen.
- Die KI berechnet die Ähnlichkeit zwischen **Bild-Embeddings** und **Text-Embeddings** und gibt die wahrscheinlichsten Begriffe zurück.

Zusatzfunktionen:
- Die **Live-Prediction** ist bewusst drosselbar, da OpenCLIP pro Vorhersage viele Textvergleiche durchführen muss.
- Ein zufälliges **Prompt-Wort** wird aus dem Vokabular gewählt und dient nur als Zeichenhilfe, nicht als feste Klasse.

---

## Wortvokabular

Für die Open-Vocabulary-Erkennung wird eine Wortliste verwendet:

- Ursprünglich wurden ca. **10.000 englische Wörter** getestet → ungeeignet, da viele Begriffe nicht skizzierbar sind.
- Anschließend eine **KI-generierte Liste (~700 Wörter)** → besser, aber weiterhin viele Verwechslungen.
- Final wird die **Quick-Draw-Kategorienliste (345 Begriffe)** verwendet → deutlich stabilere Vorhersagen.

Die Wortliste kann jederzeit durch eine andere Textdatei ersetzt werden, um das Verhalten der KI zu verändern.

---

## How to get started

### 1. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate    # macOS / Linux
```

### 2. Abhängigkeiten installieren
```
pip install -r requirements.txt
```

### 3. Backend starten (FastAPI)
```
uvicorn backend.main:app --reload --port 8004
```
Test (optional):
    Browser öffnen:
    http://127.0.0.1:8004/docs

### 4. Frontend starten
```
frontend/index.html
```
