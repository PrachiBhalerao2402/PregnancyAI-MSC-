const params = new URLSearchParams(window.location.search);
const risk = (params.get("risk") || "low").toLowerCase();

const adviceTitle = document.getElementById("adviceTitle");
const adviceStrip = document.getElementById("adviceStrip");
const adviceBox = document.getElementById("adviceBox");
const adviceText = document.getElementById("adviceText");

if (risk === "medium") {
  adviceTitle.textContent = "Medium Risk";
  adviceTitle.style.color = "#c3334f";
  adviceStrip.textContent = "Medium Risk";
  adviceStrip.style.background = "#f7ccd5";
  adviceStrip.style.borderColor = "#e38c9d";
  adviceStrip.style.color = "#af2742";
  adviceBox.style.background = "#f1d7d2";
  adviceText.textContent =
    "Please consult your gynecologist soon. Follow a low-sodium diet, stay hydrated, monitor blood pressure and blood sugar, and keep regular prenatal follow-up visits.";
} else {
  adviceTitle.textContent = "Low Risk";
  adviceText.textContent =
    "Great news! Your vitals look healthy. Continue balanced nutrition, prenatal vitamins, moderate activity, hydration, and routine check-ups.";
}
