# MMAI

bash:
1. Virtuelle Umgebung erstellen:
    python -m venv .venv
2. .venv\Scripts\activate  (mac: source .venv/bin/activate)

3. Abhängigkeiten installieren
    pip install tensorflow fastapi uvicorn pillow numpy python-multipart
    pip install scikit-learn
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git


4. NDJSON → Bilder konvertieren
    cd backend
    python convert_ndjson_to_png.py

5. KI trainieren
    python train_model.py

6. Backend starten (FastAPI)
    cd backend
    uvicorn main:app --reload --port 8001

Test (optional):
    Browser öffnen:
    http://127.0.0.1:8001/docs

7. Frontend starten
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