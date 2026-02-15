const apiUrlInput = document.getElementById("apiUrl");
const imageFileInput = document.getElementById("imageFile");
const predictForm = document.getElementById("predictForm");
const predictButton = document.getElementById("predictButton");
const statusEl = document.getElementById("status");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const topResult = document.getElementById("topResult");
const topClass = document.getElementById("topClass");
const topConfidence = document.getElementById("topConfidence");
const probabilityCard = document.getElementById("probabilityCard");
const probabilityRows = document.getElementById("probabilityRows");
const classesList = document.getElementById("classesList");
const defaultApiUrl = window.__APP_CONFIG__?.API_URL || "http://127.0.0.1:8000";
apiUrlInput.value = defaultApiUrl;

function setStatus(message, tone = "neutral") {
  statusEl.className = "status";
  if (tone === "error") {
    statusEl.classList.add("error");
  }
  if (tone === "success") {
    statusEl.classList.add("success");
  }
  statusEl.textContent = message;
}

function apiBase() {
  return apiUrlInput.value.trim().replace(/\/+$/, "");
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderProbabilities(rows) {
  probabilityRows.replaceChildren();
  rows.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "prob-row";

    const label = document.createElement("span");
    label.className = "prob-label";
    label.textContent = row.class_name;

    const value = document.createElement("span");
    value.className = "prob-value";
    value.textContent = formatPercent(row.probability);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(row.probability, 1)) * 100}%`;
    track.appendChild(fill);

    wrapper.appendChild(label);
    wrapper.appendChild(value);
    wrapper.appendChild(track);
    probabilityRows.appendChild(wrapper);
  });
}

async function loadClasses() {
  classesList.replaceChildren();
  try {
    const response = await fetch(`${apiBase()}/classes`);
    if (!response.ok) {
      throw new Error(`Could not load classes (${response.status})`);
    }
    const payload = await response.json();
    payload.classes.forEach((row) => {
      const li = document.createElement("li");
      li.textContent = `${row.class_id}: ${row.class_name}`;
      classesList.appendChild(li);
    });
    setStatus("Model classes loaded.", "success");
  } catch (error) {
    setStatus(`Class load failed: ${error.message}`, "error");
  }
}

imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files?.[0];
  if (!file) {
    previewImage.style.display = "none";
    previewImage.removeAttribute("src");
    previewPlaceholder.style.display = "block";
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
});

apiUrlInput.addEventListener("change", () => {
  loadClasses();
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageFileInput.files?.[0];
  if (!file) {
    setStatus("Please choose an image file first.", "error");
    return;
  }

  predictButton.disabled = true;
  setStatus("Running prediction...");
  topResult.classList.add("hidden");
  probabilityCard.classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${apiBase()}/predict`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload?.detail ? `: ${payload.detail}` : "";
      throw new Error(`Prediction failed (${response.status})${detail}`);
    }

    topClass.textContent = payload.top_class_name;
    topConfidence.textContent = formatPercent(payload.top_confidence);
    renderProbabilities(payload.probabilities);
    topResult.classList.remove("hidden");
    probabilityCard.classList.remove("hidden");
    setStatus(`Prediction ready for ${payload.filename}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    predictButton.disabled = false;
  }
});

loadClasses();
