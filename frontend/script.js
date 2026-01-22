console.log("✅ script.js geladen");

const STORAGE_KEY = "drawings_gallery";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const captionSpan = document.getElementById("caption");
const confidenceSpan = document.getElementById("confidence");
const topList = document.getElementById("top");

const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");
const saveBtn = document.getElementById("save");
const galleryDiv = document.getElementById("gallery");

const toggleLiveBtn = document.getElementById("toggle-live");
const promptWordSpan = document.getElementById("prompt-word");
const shuffleBtn = document.getElementById("shuffle-word");
const eraserBtn = document.getElementById("eraser");
const matchStatusSpan = document.getElementById("match-status");

// ---------------- State ----------------
let liveEnabled = true;
let lastPredictTime = 0;
const LIVE_INTERVAL = 700;
let predicting = false;

let lastImage = null;
let lastLabel = null;
let lastConfidence = null;
let lastTop = [];

let drawing = false;
let currentPromptWord = null;

let drawMode = "pen";

// ---------------- Canvas Setup ----------------
ctx.lineCap = "round";

function setPen() {
  drawMode = "pen";
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "black";
  ctx.lineWidth = 10;
}

function setEraser() {
  drawMode = "eraser";
  ctx.globalCompositeOperation = "destination-out";
  ctx.lineWidth = 24;
}

setPen();

// ---------------- Helpers ----------------
function normWord(s) {
  return (s ?? "").toString().trim().toLowerCase();
}

function updateMatchStatus() {
  if (!currentPromptWord || !lastLabel) {
    matchStatusSpan.textContent = "–";
    return;
  }
  const ok = normWord(currentPromptWord) === normWord(lastLabel);
  matchStatusSpan.textContent = ok ? "✅ guessed right!" : "❌ not the same";
}

// Snapshot
function snapshot() {
  const tmp = document.createElement("canvas");
  tmp.width = canvas.width;
  tmp.height = canvas.height;
  const t = tmp.getContext("2d");

  t.fillStyle = "white";
  t.fillRect(0, 0, tmp.width, tmp.height);
  t.drawImage(canvas, 0, 0);

  return tmp.toDataURL("image/png");
}

// ---------------- Prompt vom Backend ----------------
async function loadPrompt() {
  try {
    const res = await fetch("http://127.0.0.1:8004/random_prompt");
    const data = await res.json();
    currentPromptWord = data.word ?? null;
    promptWordSpan.textContent = currentPromptWord ?? "–";
    updateMatchStatus();
  } catch (e) {
    currentPromptWord = null;
    promptWordSpan.textContent = "something";
    updateMatchStatus();
  }
}

// ---------------- Drawing Events ----------------
canvas.addEventListener("mousedown", () => {
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener("mouseleave", () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;

  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

  if (!liveEnabled) return;

  const now = Date.now();
  if (now - lastPredictTime > LIVE_INTERVAL && !predicting) {
    lastPredictTime = now;
    livePredict();
  }
});

// ---------------- Buttons ----------------

// Shuffle Prompt
shuffleBtn.onclick = () => {
  loadPrompt();
};

// Live Toggle
toggleLiveBtn.onclick = () => {
  liveEnabled = !liveEnabled;
  toggleLiveBtn.textContent = liveEnabled ? "👁 Live: ON" : "👁 Live: OFF";
};

// Eraser Toggle
eraserBtn.onclick = () => {
  if (drawMode === "eraser") {
    setPen();
    eraserBtn.textContent = "🧽 Eraser";
  } else {
    setEraser();
    eraserBtn.textContent = "✏️ Pen";
  }
};

// Clear
clearBtn.onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  captionSpan.textContent = "–";
  confidenceSpan.textContent = "–";
  topList.innerHTML = "";
  saveBtn.style.display = "none";

  lastImage = null;
  lastLabel = null;
  lastConfidence = null;
  lastTop = [];

  matchStatusSpan.textContent = "–";


  setPen();
  eraserBtn.textContent = "🧽 Eraser";
  loadPrompt();
};

// Predict Button
predictBtn.onclick = async () => {
  captionSpan.textContent = "⏳ …";
  confidenceSpan.textContent = "–";
  topList.innerHTML = "";

  try {
    const img = snapshot();
    lastImage = img;

    const fd = new FormData();
    fd.append("image_base64", img);

    const res = await fetch("http://127.0.0.1:8004/predict", {
      method: "POST",
      body: fd
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${txt}`);
    }

    const data = await res.json();

    lastLabel = data.prediction ?? "–";
    lastConfidence = data.confidence ?? 0;
    lastTop = data.top ?? [];

    captionSpan.textContent = lastLabel;
    const pct = Math.round(lastConfidence * 100);
    confidenceSpan.textContent = `${pct}%`;

    updateMatchStatus();

    topList.innerHTML = "";
    lastTop.forEach((x) => {
      const li = document.createElement("li");
      li.textContent = `${x.label} (${Math.round(x.confidence * 100)}%)`;
      topList.appendChild(li);
    });

    saveBtn.style.display = "inline-block";

  } catch (err) {
    console.error(err);
    captionSpan.textContent = "❌ Fehler: " + (err?.message ?? err);
    confidenceSpan.textContent = "–";
    saveBtn.style.display = "none";
  }
};

// Live Predict
async function livePredict() {
  predicting = true;

  try {
    const img = snapshot();
    lastImage = img;

    const fd = new FormData();
    fd.append("image_base64", img);

    const res = await fetch("http://127.0.0.1:8004/predict", {
      method: "POST",
      body: fd
    });

    if (!res.ok) return;

    const data = await res.json();

    lastLabel = data.prediction ?? "–";
    lastConfidence = data.confidence ?? 0;
    lastTop = data.top ?? [];

    captionSpan.textContent = lastLabel;
    const pct = Math.round(lastConfidence * 100);
    confidenceSpan.textContent = `${pct}%`;

    updateMatchStatus();

    // optional: Top-Liste live aktualisieren
    topList.innerHTML = "";
    lastTop.forEach((x) => {
      const li = document.createElement("li");
      li.textContent = `${x.label} (${Math.round(x.confidence * 100)}%)`;
      topList.appendChild(li);
    });

    saveBtn.style.display = "inline-block";

  } catch (err) {
    // live = leise
  } finally {
    predicting = false;
  }
}

// Save
saveBtn.onclick = () => {
  if (!lastImage || !lastLabel) return;

  const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");

  const entry = {
    id: Date.now(),
    image: lastImage,
    caption: lastLabel,
    confidence: lastConfidence,
    time: new Date().toISOString()
  };

  data.push(entry);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));

  addToGallery(entry);
  saveBtn.style.display = "none";
};

// Gallery
function addToGallery(e) {
  const wrap = document.createElement("div");
  wrap.className = "gallery-item";

  const img = document.createElement("img");
  img.src = e.image;

  const cap = document.createElement("div");
  cap.className = "gallery-caption";
  cap.textContent = `${e.caption}`;

  const conf = document.createElement("div");
  conf.className = "gallery-confidence";
  conf.textContent = `${Math.round((e.confidence ?? 0) * 100)}%`;

  const del = document.createElement("button");
  del.className = "gallery-delete";
  del.textContent = "×";
  del.onclick = () => {
    wrap.remove();
    const arr = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")
      .filter(x => x.id !== e.id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
  };

  wrap.append(img, cap, conf, del);

  galleryDiv.appendChild(wrap);
}

// Load existing
(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")).forEach(addToGallery);

// Prompt beim Laden
loadPrompt();
