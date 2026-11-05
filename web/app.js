const form = document.querySelector("#draft-form");
const typeSelect = document.querySelector("#clause-type");
const presetSelect = document.querySelector("#preset");
const prefixInput = document.querySelector("#prefix");
const examplesRoot = document.querySelector("#examples");
const statusNode = document.querySelector("#status");
const continuationNode = document.querySelector("#continuation");
const submitButton = document.querySelector("#continue");

const EMPTY_MESSAGE = "Enter the start of a clause.";

function setStatus(message, kind) {
  statusNode.textContent = message;
  statusNode.className = kind ? `status ${kind}` : "status";
}

function showContinuation(text) {
  continuationNode.hidden = false;
  continuationNode.textContent = text;
}

function hideContinuation() {
  continuationNode.hidden = true;
  continuationNode.textContent = "";
}

function fillExample(example) {
  typeSelect.value = example.clause_type;
  prefixInput.value = example.prefix;
  hideContinuation();
  setStatus("");
  prefixInput.focus();
}

async function loadCatalog() {
  const [typesResponse, examplesResponse] = await Promise.all([
    fetch("/api/types"),
    fetch("/api/examples"),
  ]);
  if (!typesResponse.ok || !examplesResponse.ok) {
    setStatus("The clause list could not be loaded.", "error");
    return;
  }
  const types = (await typesResponse.json()).types;
  const examples = (await examplesResponse.json()).examples;
  typeSelect.replaceChildren(
    ...types.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  examplesRoot.replaceChildren(
    ...examples.map((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example";
      button.textContent = example.title;
      button.addEventListener("click", () => fillExample(example));
      return button;
    }),
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prefix = prefixInput.value.trim();
  if (!prefix) {
    hideContinuation();
    setStatus(EMPTY_MESSAGE, "error");
    return;
  }
  submitButton.disabled = true;
  hideContinuation();
  setStatus("Continuing…");
  try {
    const response = await fetch("/api/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clause_type: typeSelect.value,
        prefix,
        preset: presetSelect.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.detail || "The continuation failed.", "error");
      return;
    }
    setStatus("");
    showContinuation(payload.continuation);
  } catch (_error) {
    setStatus("The server did not respond.", "error");
  } finally {
    submitButton.disabled = false;
  }
});

const galleryNote = document.querySelector("#gallery-note");
const galleryStatus = document.querySelector("#gallery-status");
const gallerySummary = document.querySelector("#gallery-summary");
const galleryRows = document.querySelector("#gallery-rows");

const SUMMARY_LABELS = [
  ["base_word_f1", "Base word F1"],
  ["adapter_word_f1", "Adapter word F1"],
  ["base_in_register", "Base in register"],
  ["adapter_in_register", "Adapter in register"],
  ["base_nll", "Base gold NLL"],
  ["adapter_nll", "Adapter gold NLL"],
];

function metric(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function scoreLine(side) {
  if (!side) {
    return "Adapter was not scored for this prompt.";
  }
  const register = side.in_register ? "in register" : "out of register";
  return `word F1 ${metric(side.word_f1)}, ${register}, gold NLL ${metric(side.nll)}.`;
}

function addBlock(parent, title, text, score) {
  const block = document.createElement("div");
  block.className = "gallery-block";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const body = document.createElement("pre");
  body.textContent = text;
  block.append(heading, body);
  if (score) {
    const line = document.createElement("p");
    line.className = "score";
    line.textContent = score;
    block.append(line);
  }
  parent.append(block);
}

function renderGallery(report) {
  galleryNote.textContent = report.note || "";
  if (!report.rows.length) {
    galleryStatus.textContent = "No held-out continuations are saved yet.";
    return;
  }
  galleryStatus.textContent = "";
  gallerySummary.replaceChildren(
    ...SUMMARY_LABELS.map(([key, label]) => {
      const pair = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = label;
      const value = document.createElement("dd");
      value.textContent = metric(report.summary[key]);
      pair.append(term, value);
      return pair;
    }),
  );
  galleryRows.replaceChildren(
    ...report.rows.map((row) => {
      const item = document.createElement("details");
      const title = document.createElement("summary");
      title.textContent = row.clause_type;
      item.append(title);
      addBlock(item, "Prefix", row.prefix);
      addBlock(item, "Gold tail", row.gold);
      addBlock(item, "Base", row.base.text, scoreLine(row.base));
      if (row.adapter) {
        addBlock(item, "Adapter", row.adapter.text, scoreLine(row.adapter));
      } else {
        addBlock(item, "Adapter", "Adapter was not scored for this prompt.");
      }
      return item;
    }),
  );
}

async function loadGallery() {
  try {
    const response = await fetch("/api/gallery");
    if (!response.ok) {
      galleryStatus.textContent = "The saved comparison could not be loaded.";
      galleryStatus.className = "status error";
      return;
    }
    renderGallery(await response.json());
  } catch (_error) {
    galleryStatus.textContent = "The saved comparison could not be loaded.";
    galleryStatus.className = "status error";
  }
}

loadCatalog();
loadGallery();
