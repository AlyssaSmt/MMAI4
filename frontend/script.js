// ======================
// Wörter (MÜSSEN Klassen des Modells sein!)
// ======================
const WORDS = [
  "cat",
  "cup",
  "car",
  "sun",
  "eye",
  "house",
  "pants",
  "tree",
  "strawberry",
  "wine glass"
];

// ======================
// Storage
// ======================
const STORAGE_KEY = "montagsmaler_correct_drawings";

// ======================
// DOM-Elemente
// ======================
const targetWordSpan = document.getElementById("target-word");

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");
const saveBtn = document.getElementById("save-btn");

const predictionSpan = document.getElementById("prediction");
const confidenceSpan = document.getElementById("confidence");
const topList = document.getElementById("top-list");

const captionText = document.getElementById("caption-text");
const captionConf = document.getElementById("caption-conf");
const captionTop = document.getElementById("caption-top");

const galleryDiv = document.getElementById("gallery");

const penBtn = document.getElementById("pen-btn");
const eraserBtn = document.getElementById("eraser-btn");

// ======================
// Zustand
// ======================
let targetWord = null;
let drawing = false;
let hasDrawn = false;

let lastImageDataUrl = null;
let lastConfidence = null;

let lastCaptionText = null;
let lastCaptionConfidence = null;

let mode = "pen"; // "pen" | "eraser"
let lastPredictTime = 0;

// ======================
// Canvas Setup
// ======================
ctx.lineCap = "round";
resetCanvas();

function resetCanvas() {
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  setPen();
}

// ======================
// Zeichenmodi
// ======================
function setPen() {
  mode = "pen";
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "black";
  ctx.lineWidth = 12;
}

function setEraser() {
  mode = "eraser";
  ctx.globalCompositeOperation = "destination-out";
  ctx.lineWidth = 24;
}

// ======================
// Zielwort
// ======================
function chooseNewWord() {
  const i = Math.floor(Math.random() * WORDS.length);
  targetWord = WORDS[i];
  targetWordSpan.textContent = targetWord;
}

// ======================
// Initialisierung
// ======================
chooseNewWord();
loadGallery();
saveBtn.style.display = "none";
setPen();

// ======================
// Events
// ======================
canvas.addEventListener("mousedown", () => {
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();

  if (mode === "pen" && hasDrawn) {
    maybePredict();
    hasDrawn = false;
  }
});

canvas.addEventListener("mouseleave", () => {
  drawing = false;
});

canvas.addEventListener("mousemove", draw);

clearBtn.addEventListener("click", () => {
  resetCanvas();

  predictionSpan.textContent = "–";
  confidenceSpan.textContent = "–";
  topList.innerHTML = "";

  captionText.textContent = "–";
  captionConf.textContent = "–";
  captionTop.innerHTML = "";

  saveBtn.style.display = "none";

  lastImageDataUrl = null;
  lastConfidence = null;
  lastCaptionText = null;
  lastCaptionConfidence = null;
  hasDrawn = false;

  chooseNewWord();
});

predictBtn.addEventListener("click", maybePredict);
penBtn.addEventListener("click", setPen);
eraserBtn.addEventListener("click", setEraser);

// ======================
// Zeichnen
// ======================
function draw(e) {
  if (!drawing) return;

  if (mode === "pen") hasDrawn = true;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

  const now = Date.now();
  if (mode === "pen" && now - lastPredictTime > 400) {
    lastPredictTime = now;
    maybePredict();
  }
}

// ======================
// Ink-Menge
// ======================
function getInkAmount() {
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let count = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    if (a > 200 && (r + g + b) < 600) count++;
  }
  return count;
}

// ======================
// Snapshot mit weißem Hintergrund
// ======================
function snapshotCanvas() {
  const tmp = document.createElement("canvas");
  tmp.width = canvas.width;
  tmp.height = canvas.height;

  const tctx = tmp.getContext("2d");
  tctx.fillStyle = "white";
  tctx.fillRect(0, 0, tmp.width, tmp.height);
  tctx.drawImage(canvas, 0, 0);

  return tmp.toDataURL("image/png");
}

// ======================
// Caption (CLIP)
// ======================
async function fetchCaption(dataUrl) {
  const formData = new FormData();
  formData.append("image_base64", dataUrl);

  const response = await fetch("http://127.0.0.1:8001/caption", {
    method: "POST",
    body: formData
  });

  return await response.json();
}

// ======================
// Prediction
// ======================
async function maybePredict() {
  if (getInkAmount() < 50) {
    predictionSpan.textContent = "noch zu wenig gezeichnet";
    confidenceSpan.textContent = "";
    topList.innerHTML = "";

    captionText.textContent = "–";
    captionConf.textContent = "–";
    captionTop.innerHTML = "";
    return;
  }

  const dataUrl = snapshotCanvas();

  // -------- CNN --------
  const formData = new FormData();
  formData.append("image_base64", dataUrl);

  const response = await fetch("http://127.0.0.1:8001/predict", {
    method: "POST",
    body: formData
  });

  const result = await response.json();
  updateUI(result);

  // -------- CLIP --------
  try {
    const captionResult = await fetchCaption(dataUrl);

    captionText.textContent = captionResult.caption;
    captionConf.textContent =
      Math.round(captionResult.confidence * 100) + "%";

    lastCaptionText = captionResult.caption;
    lastCaptionConfidence = captionResult.confidence;

    captionTop.innerHTML = "";
    captionResult.top.forEach(item => {
      const li = document.createElement("li");
      li.textContent =
        `${item.caption} (${Math.round(item.confidence * 100)}%)`;
      captionTop.appendChild(li);
    });

  } catch (err) {
    captionText.textContent = "Caption nicht verfügbar";
    captionConf.textContent = "";
    captionTop.innerHTML = "";
    console.error(err);
  }
}

// ======================
// UI-Logik
// ======================
function updateUI(result) {
  const conf = Number(result.confidence);
  const prediction = result.prediction;

  saveBtn.style.display = "none";
  lastImageDataUrl = null;
  lastConfidence = null;

  let status = "";

  if (
    prediction === targetWord &&
    conf >= 0.35 &&
    lastCaptionText !== null &&
    lastCaptionConfidence !== null
  ) {
    status = "✅ richtig!";
    lastImageDataUrl = snapshotCanvas();
    lastConfidence = conf;
    saveBtn.style.display = "inline-block";
  } else if (conf < 0.25) {
    status = "unsicher";
  } else {
    status = "❌ falsch";
  }

  predictionSpan.textContent = `${prediction} ${status}`;
  confidenceSpan.textContent = Math.round(conf * 100) + "%";

  topList.innerHTML = "";
  result.top.forEach(item => {
    const li = document.createElement("li");
    li.textContent =
      `${item.label}: ${Math.round(item.confidence * 100)}%`;
    topList.appendChild(li);
  });
}

// ======================
// Speichern
// ======================
saveBtn.addEventListener("click", () => {
  if (!lastImageDataUrl) return;

  saveCorrectDrawing(
    lastImageDataUrl,
    targetWord,
    lastConfidence,
    lastCaptionText,
    lastCaptionConfidence
  );

  saveBtn.style.display = "none";
  lastImageDataUrl = null;
  lastConfidence = null;
  lastCaptionText = null;
  lastCaptionConfidence = null;
});

// ======================
// Galerie
// ======================
function saveCorrectDrawing(image, label, confidence, caption, captionConfidence) {
  const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");

  const entry = {
    image,
    label,
    confidence,
    caption,
    captionConfidence,
    time: Date.now()
  };

  stored.push(entry);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  addImageToGallery(entry);
}

function loadGallery() {
  const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  stored.forEach(addImageToGallery);
}

function addImageToGallery({ image, label, confidence, caption, captionConfidence, time }) {
  const wrapper = document.createElement("div");
  wrapper.className = "gallery-item";

  const img = document.createElement("img");
  img.src = image;
  img.title = `${label} – ${Math.round(confidence * 100)}%`;

  const captionEl = document.createElement("div");
  captionEl.className = "gallery-caption";
  captionEl.textContent = caption || "–";

  const captionConfEl = document.createElement("div");
  captionConfEl.className = "gallery-confidence";
  captionConfEl.textContent =
    captionConfidence ? Math.round(captionConfidence * 100) + "%" : "";

  const del = document.createElement("button");
  del.textContent = "×";
  del.onclick = () => {
    deleteImage(time);
    wrapper.remove();
  };

  wrapper.append(img, captionEl, captionConfEl, del);
  galleryDiv.appendChild(wrapper);
}

function deleteImage(time) {
  const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")
    .filter(e => e.time !== time);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
}
