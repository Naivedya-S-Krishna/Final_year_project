const API_URL = "http://localhost:4000";

const messagesDiv = document.getElementById("messages");

function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = text;
  messagesDiv.appendChild(msg);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendReport() {
  const text = document.getElementById("report").value;
  try {
    const report = JSON.parse(text);
    addMessage("📄 Report sent for analysis...", "user");

    const res = await fetch(`${API_URL}/analyze-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(report),
    });
    const data = await res.json();
    if (data.success) {
      addMessage(`🧾 Summary: ${data.result.summary}`, "bot");
      data.result.suggestions.forEach((s) => addMessage("💡 " + s, "bot"));
    } else {
      addMessage("⚠️ Error analyzing report.", "bot");
    }
  } catch (err) {
    addMessage("❌ Invalid JSON or server error.", "bot");
  }
}

function sendMessage() {
  const text = document.getElementById("userInput").value;
  if (!text.trim()) return;
  addMessage(text, "user");

  // simple responses (client-side)
  let reply = "🙂 Try to focus on your breathing and relax.";
  if (text.toLowerCase().includes("sad"))
    reply = "💬 It's okay to feel sad. Try journaling or reaching out to a friend.";
  if (text.toLowerCase().includes("anxious"))
    reply = "🧘 Deep breathing or short meditation can help calm anxiety.";
  if (text.toLowerCase().includes("angry"))
    reply = "😤 Take a short walk and count backwards from 50 slowly.";

  addMessage(reply, "bot");
  document.getElementById("userInput").value = "";
}
