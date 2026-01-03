const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const captionSpan = document.getElementById("caption");
const confidenceSpan = document.getElementById("confidence");
const topList = document.getElementById("top");

const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");

let drawing = false;

// Canvas Setup
ctx.lineCap = "round";
ctx.lineWidth = 10;
ctx.strokeStyle = "black";

// Zeichnen
canvas.addEventListener("mousedown", () => {
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
});

// Löschen
clearBtn.onclick = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  captionSpan.textContent = "–";
  confidenceSpan.textContent = "–";
  topList.innerHTML = "";
};

// Snapshot mit weißem Hintergrund
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

// Prediction
predictBtn.onclick = async () => {
  const img = snapshot();

  const fd = new FormData();
  fd.append("image_base64", img);

  const res = await fetch("http://127.0.0.1:8000/caption", {
    method: "POST",
    body: fd
  });

  const data = await res.json();

  captionSpan.textContent = data.caption;
  confidenceSpan.textContent = Math.round(data.confidence * 100) + "%";

  topList.innerHTML = "";
  (data.top || []).forEach(x => {
    const li = document.createElement("li");
    li.textContent = `${x.caption} (${Math.round(x.confidence * 100)}%)`;
    topList.appendChild(li);
  });
};
