
const API_URL = "http://localhost:8000/chat";

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatLog = document.getElementById("chatLog");

function appendMessage(from, text) {
  const row = document.createElement("div");
  row.className = `msg-row msg ${from}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${from}`;
  bubble.textContent = text;

  row.appendChild(bubble);
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function callMarcusAPI(message) {
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    return data.reply || "Marcus forgot how to talk.";
  } catch (err) {
    return "Lost connection to Marcus. He crawled away.";
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage("user", text);
  chatInput.value = "";

  appendMessage("marcus", "… recalibrating worm neurons …");
  const placeholder = chatLog.lastChild;

  const reply = await callMarcusAPI(text);
  placeholder.querySelector(".bubble.marcus").textContent = reply;
});
