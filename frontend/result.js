const params = new URLSearchParams(window.location.search);
const risk = (params.get("risk") || "low").toLowerCase();

const heading = document.getElementById("riskHeading");
const strip = document.getElementById("riskStrip");
const card = document.getElementById("resultCard");
const icon = document.getElementById("statusIcon");
const reasonList = document.getElementById("reasonList");
const adviceBox = document.getElementById("adviceBox");
const adviceText = document.getElementById("adviceText");

if (risk === "medium") {
  heading.textContent = "Medium Risk";
  heading.style.color = "#c3334f";
  strip.textContent = "Medium Risk";
  strip.style.background = "#f7ccd5";
  strip.style.borderColor = "#e38c9d";
  strip.style.color = "#af2742";
  icon.textContent = "!";
  icon.style.borderColor = "#d34660";
  icon.style.color = "#d34660";

  reasonList.innerHTML = `
    <li>Blood pressure and stress are above normal thresholds</li>
    <li>One or more risk factors need active medical follow-up</li>
    <li>Potential signs of complication require timely review</li>
  `;

  adviceBox.style.background = "#f1d7d2";
  adviceText.textContent = "Please consult your gynecologist soon. Follow a low-sodium diet, rest well, stay hydrated, and attend regular prenatal monitoring.";
} else {
  heading.textContent = "Low Risk";
}
