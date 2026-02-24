const form = document.getElementById("risk-form");
const stressRange = document.getElementById("stressRange");
const stressValue = document.getElementById("stressValue");
const resultBox = document.getElementById("result");

stressRange.addEventListener("input", () => {
  stressValue.textContent = stressRange.value;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  // convert to numeric values expected by ML backend
  for (const key of Object.keys(payload)) {
    payload[key] = Number(payload[key]);
  }

  try {
    // Connect this endpoint to your Python backend (Flask/FastAPI) prediction API.
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error("Prediction request failed");

    const data = await response.json();
    const risk = String(data.risk || "").toLowerCase();

    resultBox.classList.remove("hidden", "low", "medium");
    if (risk.includes("low")) {
      resultBox.classList.add("low");
      resultBox.textContent = "✅ Predicted Risk Level: Low Risk";
    } else {
      resultBox.classList.add("medium");
      resultBox.textContent = "🚨 Predicted Risk Level: Medium Risk";
    }
  } catch (error) {
    resultBox.classList.remove("hidden", "low", "medium");
    resultBox.classList.add("medium");
    resultBox.textContent = "Unable to fetch prediction. Start backend API at /predict.";
  }
});
