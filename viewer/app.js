const cases = window.RESULT_FRAMES || [];

const caseTitle = document.querySelector("#caseTitle");
const caseSwitcher = document.querySelector(".case-switcher");
const frameImage = document.querySelector("#frameImage");
const maskOverlay = document.querySelector("#maskOverlay");
const sliceLabel = document.querySelector("#sliceLabel");
const frameCount = document.querySelector("#frameCount");
const frameSlider = document.querySelector("#frameSlider");
const prevBtn = document.querySelector("#prevBtn");
const playBtn = document.querySelector("#playBtn");
const playIcon = document.querySelector("#playIcon");
const nextBtn = document.querySelector("#nextBtn");
const speedSelect = document.querySelector("#speedSelect");
const combinedToggle = document.querySelector("#combinedToggle");
const maskToggle = document.querySelector("#maskToggle");
const landmarkToggle = document.querySelector("#landmarkToggle");

let caseIndex = 0;
let frameIndex = 0;
let timer = null;

function currentCase() {
  return cases[caseIndex];
}

function currentFrame() {
  return currentCase().frames[frameIndex];
}

function renderCaseButtons() {
  caseSwitcher.innerHTML = "";

  cases.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-btn${index === caseIndex ? " active" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => {
      caseIndex = index;
      frameIndex = 0;
      pause();
      render();
    });
    caseSwitcher.appendChild(button);
  });
}

function render() {
  const item = currentCase();
  const frame = currentFrame();
  caseTitle.textContent = item.label;
  frameImage.src = frame.image;
  frameImage.classList.toggle("hide-combined", !combinedToggle.checked);
  frameSlider.max = Math.max(item.frames.length - 1, 0);
  frameSlider.value = frameIndex;
  sliceLabel.textContent = `Slice ${frame.slice}`;
  frameCount.textContent = `${frameIndex + 1} / ${item.frames.length}`;

  maskOverlay.classList.toggle("visible", maskToggle.checked);
  maskOverlay.classList.toggle("real-mask", Boolean(frame.mask));
  maskOverlay.style.backgroundImage = frame.mask ? `url("${frame.mask}")` : "";

  frameImage.style.filter = landmarkToggle.checked ? "none" : "saturate(0.7) contrast(0.88)";
  renderCaseButtons();
}

function step(delta) {
  const total = currentCase().frames.length;
  frameIndex = (frameIndex + delta + total) % total;
  render();
}

function play() {
  pause();
  timer = window.setInterval(() => step(1), Number(speedSelect.value));
  playIcon.textContent = "Pause";
  playBtn.setAttribute("aria-label", "Pause frames");
}

function pause() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
  playIcon.textContent = "Play";
  playBtn.setAttribute("aria-label", "Play frames");
}

prevBtn.addEventListener("click", () => step(-1));
nextBtn.addEventListener("click", () => step(1));
playBtn.addEventListener("click", () => (timer ? pause() : play()));

frameSlider.addEventListener("input", (event) => {
  frameIndex = Number(event.target.value);
  render();
});

speedSelect.addEventListener("change", () => {
  if (timer) {
    play();
  }
});

[combinedToggle, maskToggle, landmarkToggle].forEach((control) => {
  control.addEventListener("change", render);
});

document.addEventListener("keydown", (event) => {
  if (event.key === " ") {
    event.preventDefault();
    timer ? pause() : play();
  }
  if (event.key === "ArrowLeft") {
    step(-1);
  }
  if (event.key === "ArrowRight") {
    step(1);
  }
});

if (cases.length) {
  render();
}
