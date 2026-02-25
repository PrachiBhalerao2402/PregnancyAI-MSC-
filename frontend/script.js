const form = document.getElementById("risk-form");
const stressRange = document.getElementById("stressRange");
const stressValue = document.getElementById("stressValue");

stressRange.addEventListener("input", () => {
  stressValue.textContent = stressRange.value;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  for (const key of Object.keys(payload)) {
    payload[key] = Number(payload[key]);
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error("Prediction request failed");

    const data = await response.json();
    const risk = String(data.risk || "Medium Risk").toLowerCase();
    let encoded = "medium";
    if (risk.includes("high")) encoded = "high";
    else if (risk.includes("low")) encoded = "low";
    window.location.href = `result.html?risk=${encoded}`;
  } catch (error) {
    // fallback demo behavior if backend is not running
    let fallbackRisk = "medium";
    if (payload.StressLevel >= 8 && payload.BloodPressure >= 140) fallbackRisk = "high";
    else if (payload.StressLevel <= 5 && payload.BloodPressure < 130) fallbackRisk = "low";
    window.location.href = `result.html?risk=${fallbackRisk}`;
  }
});
