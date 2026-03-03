const params = new URLSearchParams(window.location.search);
const risk = (params.get("risk") || "low").toLowerCase();

const adviceTitle = document.getElementById("adviceTitle");
const adviceStrip = document.getElementById("adviceStrip");
const adviceBox = document.getElementById("adviceBox");
const adviceText = document.getElementById("adviceText");
const adviceReasons = document.getElementById("adviceReasons");
const statusIcon = document.getElementById("statusIcon");

if (risk === "medium") {
  adviceTitle.textContent = "Medium Risk";
  adviceTitle.style.color = "#c3334f";
  adviceStrip.textContent = "Medium Risk";
  adviceStrip.style.background = "#f7ccd5";
  adviceStrip.style.borderColor = "#e38c9d";
  adviceStrip.style.color = "#af2742";
  statusIcon.textContent = "!";
  statusIcon.style.borderColor = "#d34660";
  statusIcon.style.color = "#d34660";

  adviceReasons.innerHTML = `
    <li>Blood pressure and stress are above normal thresholds</li>
    <li>One or more risk factors need active medical follow-up</li>
    <li>Potential signs of complication require timely review</li>
  `;

  adviceBox.style.background = "#f1d7d2";
  adviceText.textContent =
    "Please consult your gynecologist soon. Follow a low-sodium diet, stay hydrated, monitor blood pressure and blood sugar, and keep regular prenatal follow-up visits.";
} else if (risk === "high") {
  adviceTitle.textContent = "High Risk";
  adviceTitle.style.color = "#a10f2d";
  adviceStrip.textContent = "High Risk";
  adviceStrip.style.background = "#f5c0cd";
  adviceStrip.style.borderColor = "#dc6b86";
  adviceStrip.style.color = "#8f112d";
  statusIcon.textContent = "!";
  statusIcon.style.borderColor = "#b31b3f";
  statusIcon.style.color = "#b31b3f";

  adviceReasons.innerHTML = `
    <li>Multiple high-risk indicators are present</li>
    <li>Urgent specialist monitoring is required</li>
    <li>Risk of serious maternal/fetal complications is elevated</li>
  `;

  adviceBox.style.background = "#f0ccd4";
  adviceText.textContent =
    "Please seek immediate medical evaluation. Do not delay antenatal specialist care, monitor warning symptoms, and strictly follow physician guidance.";
} else {
  adviceTitle.textContent = "Low Risk";
  adviceText.textContent =
    "Great news! Your vitals look healthy. Continue your balanced diet, prenatal vitamins, moderate exercise, regular check-ups, and proper rest.";
}
