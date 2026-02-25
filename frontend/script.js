const form = document.getElementById("risk-form");
const stressRange = document.getElementById("stressRange");
const stressValue = document.getElementById("stressValue");
const chatWindow = document.getElementById("chat-window");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chips = document.querySelectorAll(".chip");

stressRange.addEventListener("input", () => {
  stressValue.textContent = stressRange.value;
});

function addBubble(text, who = "bot") {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function botReply(q) {
  const text = q.toLowerCase();
  if (text.includes("diet")) {
    return "Diet plan: include leafy greens, lentils, eggs/paneer, seasonal fruits, whole grains, and 2.5-3L water daily. Limit sugary drinks and fried foods.";
  }
  if (text.includes("exercise") || text.includes("workout") || text.includes("walk")) {
    return "Exercise: 20-30 min brisk walk, prenatal stretching, and breathing exercises 5 days/week (if your doctor allows). Avoid high-impact workouts.";
  }
  if (text.includes("bp") || text.includes("pressure")) {
    return "For high BP: reduce salt, monitor BP at home, avoid stress, stay hydrated, and see your doctor if readings stay high.";
  }
  if (text.includes("sugar") || text.includes("diabetes")) {
    return "For blood sugar: small frequent meals, low-glycemic carbs, fiber-rich food, and regular glucose checks help keep sugar controlled.";
  }
  if (text.includes("stress") || text.includes("anxiety")) {
    return "Stress care: practice deep breathing, sleep 7-8 hrs, take short walks, and share concerns with family/doctor.";
  }
  if (text.includes("high risk")) {
    return "High-risk pregnancy needs close specialist follow-up. Please keep regular doctor visits and do not ignore warning signs.";
  }
  return "I can help with diet plans, exercise, blood pressure, sugar control, stress, and trimester-wise care. Ask me anything specific 🌸";
}

function sendChat(message) {
  if (!message.trim()) return;
  addBubble(message, "user");
  setTimeout(() => addBubble(botReply(message), "bot"), 220);
}

chatSend.addEventListener("click", () => {
  sendChat(chatInput.value);
  chatInput.value = "";
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendChat(chatInput.value);
    chatInput.value = "";
  }
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => sendChat(chip.dataset.q || chip.textContent));
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
    let fallbackRisk = "medium";
    if (payload.StressLevel >= 8 && payload.BloodPressure >= 140) fallbackRisk = "high";
    else if (payload.StressLevel <= 5 && payload.BloodPressure < 130) fallbackRisk = "low";
    window.location.href = `result.html?risk=${fallbackRisk}`;
  }
});
