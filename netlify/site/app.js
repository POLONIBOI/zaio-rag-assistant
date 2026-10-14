// Sends the question to POST /ask and shows the answer with its source.
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const button = document.getElementById("ask-button");
const count = document.getElementById("count");
const result = document.getElementById("result");
const resultQuestion = document.getElementById("result-question");
const resultAnswer = document.getElementById("result-answer");
const resultSource = document.getElementById("result-source");

function show(kind, question, answer, source) {
  result.hidden = false;
  result.className = `result${kind ? ` result--${kind}` : ""}`;
  resultQuestion.textContent = question;
  resultAnswer.textContent = answer;
  resultSource.replaceChildren();
  if (!source) return;

  const label = document.createElement("strong");
  label.textContent = "Source: ";
  resultSource.append(label);
  if (source.startsWith("http")) {
    const link = document.createElement("a");
    link.href = source;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = source.replace("https://", "");
    resultSource.append(link);
  } else {
    resultSource.append(source);
  }
}

async function ask(question) {
  button.disabled = true;
  button.textContent = "Asking…";
  show("loading", question, "Searching the handbook and zaio.io…");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      show("error", question, data.error ?? `Something went wrong (error ${response.status}). Please try again.`);
    } else if (data.source === null) {
      show("empty", question, data.answer);
    } else {
      show("", question, data.answer, data.source);
    }
  } catch {
    show("error", question, "Couldn't reach the assistant. Check your connection and try again.");
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

input.addEventListener("input", () => {
  count.textContent = `${input.value.length} / 500`;
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) {
    input.focus();
    return;
  }
  ask(question);
});

// Enter sends, Shift+Enter adds a new line.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    input.value = example.textContent;
    count.textContent = `${input.value.length} / 500`;
    ask(example.textContent);
  });
}
