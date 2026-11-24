const MODEL_URL = "model/sami-face-v2.onnx";
const INPUT_SIZE = 640;
const NUM_BOXES = 8400;
const SAMMY = 1; // class 0 = Other, class 1 = Sammy

const fileInput = document.getElementById("file-input");
const statusBox = document.getElementById("status");
const confidenceInput = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidence-value");
const bestOnly = document.getElementById("best-only");
const resultBox = document.getElementById("result");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let session = null;
let lastRun = null; // { output, scale, dx, dy, img }

// ---- pure functions start ----
function iou(a, b) {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = iw * ih;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

// The model returns 6 rows x 8400 columns:
// rows 0-3 = box centre x, centre y, width, height (in 640 x 640 pixels)
// rows 4-5 = score for class 0 (Other) and class 1 (Sammy)
function decode(output, threshold) {
  const n = NUM_BOXES;
  const candidates = [];
  for (let i = 0; i < n; i++) {
    const s0 = output[4 * n + i];
    const s1 = output[5 * n + i];
    const score = Math.max(s0, s1);
    if (score < threshold) continue;
    const cx = output[i];
    const cy = output[n + i];
    const w = output[2 * n + i];
    const h = output[3 * n + i];
    candidates.push({
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
      score, cls: s1 > s0 ? 1 : 0,
    });
  }

  // Remove overlapping duplicates, keeping the most confident box
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of candidates) {
    if (kept.every((k) => iou(k, c) < 0.5)) kept.push(c);
  }
  return kept;
}

// Convert a box from the 640 x 640 model space back to the original photo
function toImageSpace(box, scale, dx, dy, width, height) {
  return {
    x1: Math.max(0, (box.x1 - dx) / scale),
    y1: Math.max(0, (box.y1 - dy) / scale),
    x2: Math.min(width, (box.x2 - dx) / scale),
    y2: Math.min(height, (box.y2 - dy) / scale),
    score: box.score,
    cls: box.cls,
  };
}
// ---- pure functions end ----

function setStatus(text) {
  statusBox.textContent = text;
}

async function loadModel() {
  try {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
    setStatus("Model ready. Choose a photo.");
    fileInput.disabled = false;
  } catch (error) {
    setStatus("Could not load the model. Check your internet connection and refresh the page.");
    console.error(error);
  }
}

// Resize the photo into a 640 x 640 square (grey padding) as the model expects
function preprocess(img) {
  const scale = Math.min(INPUT_SIZE / img.naturalWidth, INPUT_SIZE / img.naturalHeight);
  const nw = Math.round(img.naturalWidth * scale);
  const nh = Math.round(img.naturalHeight * scale);
  const dx = Math.floor((INPUT_SIZE - nw) / 2);
  const dy = Math.floor((INPUT_SIZE - nh) / 2);

  const work = document.createElement("canvas");
  work.width = INPUT_SIZE;
  work.height = INPUT_SIZE;
  const wctx = work.getContext("2d");
  wctx.fillStyle = "rgb(114,114,114)";
  wctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  wctx.drawImage(img, dx, dy, nw, nh);

  const pixels = wctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const size = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    data[i] = pixels[i * 4] / 255;
    data[size + i] = pixels[i * 4 + 1] / 255;
    data[2 * size + i] = pixels[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]), scale, dx, dy };
}

function render() {
  if (!lastRun) return;
  const { output, scale, dx, dy, img } = lastRun;
  const width = img.naturalWidth;
  const height = img.naturalHeight;

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(img, 0, 0, width, height);

  const threshold = Number(confidenceInput.value) / 100;
  let boxes = decode(output, threshold)
    .filter((b) => b.cls === SAMMY)
    .map((b) => toImageSpace(b, scale, dx, dy, width, height));
  if (bestOnly.checked && boxes.length > 1) boxes = [boxes[0]];

  const longSide = Math.max(width, height);
  const lineWidth = Math.max(3, Math.round(longSide / 250));
  const fontSize = Math.max(16, Math.round(longSide / 40));
  ctx.lineWidth = lineWidth;
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.textBaseline = "bottom";

  boxes.forEach((b) => {
    ctx.strokeStyle = "#FFC400";
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const label = `Sami ${Math.round(b.score * 100)}%`;
    const textWidth = ctx.measureText(label).width;
    const labelY = b.y1 - lineWidth > fontSize + 4 ? b.y1 : b.y1 + fontSize + 6;
    ctx.fillStyle = "#FFC400";
    ctx.fillRect(b.x1 - lineWidth / 2, labelY - fontSize - 4, textWidth + 12, fontSize + 6);
    ctx.fillStyle = "#0D0D0D";
    ctx.fillText(label, b.x1 + 4, labelY);
  });

  canvas.classList.remove("hidden");
  resultBox.classList.remove("hidden");
  if (boxes.length === 0) {
    resultBox.textContent = "Sami's face was not found at this confidence level.";
  } else if (boxes.length === 1) {
    resultBox.textContent = `Sami's face found (confidence ${Math.round(boxes[0].score * 100)}%).`;
  } else {
    resultBox.textContent = `${boxes.length} faces marked as Sami. Raise the confidence or tick "best match only" to narrow it down.`;
  }
}

async function detect(img) {
  setStatus("Detecting...");
  fileInput.disabled = true;
  try {
    const { tensor, scale, dx, dy } = preprocess(img);
    const results = await session.run({ images: tensor });
    const output = results[session.outputNames[0]].data;
    lastRun = { output, scale, dx, dy, img };
    render();
    setStatus("Done. Choose another photo to try again.");
  } catch (error) {
    setStatus("Something went wrong while detecting. Please try another photo.");
    console.error(error);
  }
  fileInput.disabled = false;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  fileInput.value = ""; // lets the same photo be chosen again later
  const img = new Image();
  img.onload = () => {
    detect(img);
  };
  img.onerror = () => setStatus("That file could not be opened as an image.");
  img.src = url;
});

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = confidenceInput.value + "%";
  render();
});
bestOnly.addEventListener("change", render);

fileInput.disabled = true;
loadModel();