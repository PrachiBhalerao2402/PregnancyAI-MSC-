const params = new URLSearchParams(window.location.search);
const risk = (params.get("risk") || "low").toLowerCase();

const heading = document.getElementById("riskHeading");
const strip = document.getElementById("riskStrip");
const icon = document.getElementById("statusIcon");
const reasonList = document.getElementById("reasonList");
const adviceLink = document.getElementById("adviceLink");

const normalized = ["low", "medium", "high"].includes(risk) ? risk : "low";
adviceLink.href = `advice.html?risk=${normalized}`;

if (normalized === "medium") {
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
} else if (normalized === "high") {
  heading.textContent = "High Risk";
  heading.style.color = "#a10f2d";
  strip.textContent = "High Risk";
  strip.style.background = "#f5c0cd";
  strip.style.borderColor = "#dc6b86";
  strip.style.color = "#8f112d";
  icon.textContent = "!";
  icon.style.borderColor = "#b31b3f";
  icon.style.color = "#b31b3f";

  reasonList.innerHTML = `
    <li>Multiple critical markers are above safe pregnancy thresholds</li>
    <li>Immediate specialist supervision is strongly recommended</li>
    <li>Higher chance of complications without close follow-up</li>
  `;
}
