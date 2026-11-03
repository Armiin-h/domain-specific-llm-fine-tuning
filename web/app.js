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

loadCatalog();
