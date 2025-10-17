(() => {
  const BOT_URL = "https://faqbot-4how.onrender.com/widget.html";
  if (window.MyAiToolsetWidget) return;

  // ===== Chat Bubble =====
  const bubble = document.createElement("div");
  bubble.id = "myaitoolset-bubble";
  bubble.innerText = "💬";
  Object.assign(bubble.style, {
    position: "fixed",
    bottom: "24px",
    right: "24px",
    width: "56px",
    height: "56px",
    background: "#3B82F6",
    color: "#fff",
    borderRadius: "50%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontSize: "26px",
    cursor: "pointer",
    zIndex: 999999,
    boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
    transition: "transform 0.2s ease, opacity 0.2s ease",
  });
  bubble.onmouseenter = () => (bubble.style.transform = "scale(1.08)");
  bubble.onmouseleave = () => (bubble.style.transform = "scale(1.0)");
  document.body.appendChild(bubble);

  // ===== Iframe Chat Window =====
  const iframe = document.createElement("iframe");
  iframe.id = "myaitoolset-widget";
  iframe.src = BOT_URL;
  Object.assign(iframe.style, {
    position: "fixed",
    bottom: "90px",
    right: "24px",
    width: "400px",
    height: "560px",
    border: "none",
    borderRadius: "16px",
    boxShadow: "0 8px 30px rgba(0,0,0,0.25)",
    display: "none",
    opacity: 0,
    zIndex: 999998,
    background: "#fff",
    transition: "opacity 0.3s ease",
  });
  document.body.appendChild(iframe);

  // ===== Smart Lead Form Modal (outside iframe) =====
  const leadModal = document.createElement("div");
  Object.assign(leadModal.style, {
    position: "fixed",
    inset: "0",
    background: "rgba(0,0,0,0.45)",
    display: "none",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000000,
  });

  leadModal.innerHTML = `
    <div id="myaitoolset-leadform" style="
      background: white;
      padding: 20px;
      border-radius: 16px;
      width: 320px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
      font-family: system-ui, sans-serif;
      color: #111;
    ">
      <h3 style="margin-bottom: 12px; color:#3B82F6;">Confirm Your Info</h3>
      <label>Name</label>
      <input id="lead-name" placeholder="Your name" style="width:100%;margin:6px 0 12px;padding:8px;border-radius:8px;border:1px solid #ccc"/>
      <label>Email</label>
      <input id="lead-email" placeholder="you@email.com" style="width:100%;margin:6px 0 12px;padding:8px;border-radius:8px;border:1px solid #ccc"/>
      <label>Phone</label>
      <input id="lead-phone" placeholder="(555) 123-4567" style="width:100%;margin:6px 0 16px;padding:8px;border-radius:8px;border:1px solid #ccc"/>
      <label>Message</label>
      <textarea id="lead-message" rows="3" placeholder="Tell us more..." style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc"></textarea>
      <div style="margin-top:14px;display:flex;gap:8px;justify-content:flex-end;">
        <button id="lead-cancel" style="padding:8px 14px;border:none;border-radius:8px;background:#e5e7eb;cursor:pointer;">Cancel</button>
        <button id="lead-confirm" style="padding:8px 14px;border:none;border-radius:8px;background:#3B82F6;color:white;cursor:pointer;">Confirm & Send</button>
      </div>
    </div>`;
  document.body.appendChild(leadModal);

  // ===== Functions =====
  function toggleChat() {
    const showing = iframe.style.display === "block";
    if (showing) {
      iframe.style.opacity = 0;
      setTimeout(() => (iframe.style.display = "none"), 200);
    } else {
      iframe.style.display = "block";
      setTimeout(() => (iframe.style.opacity = 1), 20);
    }
  }

  function openLeadForm(data = {}) {
    leadModal.style.display = "flex";
    document.getElementById("lead-name").value = data.name || "";
    document.getElementById("lead-email").value = data.email || "";
    document.getElementById("lead-phone").value = data.phone || "";
    document.getElementById("lead-message").value = data.message || "";
  }

  function closeLeadForm() {
    leadModal.style.display = "none";
  }

  async function sendLead() {
    const payload = {
      name: document.getElementById("lead-name").value.trim(),
      email: document.getElementById("lead-email").value.trim(),
      phone: document.getElementById("lead-phone").value.trim(),
      message: document.getElementById("lead-message").value.trim(),
    };

    if (!payload.name || !payload.email || !payload.phone) {
      alert("Please fill in your name, email, and phone.");
      return;
    }

    try {
      const res = await fetch("https://faqbot-4how.onrender.com/lead", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("HTTP error " + res.status);
      closeLeadForm();
      alert("✅ Thank you! Your request has been sent.");
    } catch (err) {
      console.error("Lead send failed:", err);
      alert("❌ Something went wrong sending your request.");
    }
  }

  // ===== Event Listeners =====
  bubble.onclick = toggleChat;
  leadModal.querySelector("#lead-cancel").onclick = closeLeadForm;
  leadModal.querySelector("#lead-confirm").onclick = sendLead;

  // ===== Listen for chatbot messages =====
  window.addEventListener("message", (event) => {
    if (!event.data || typeof event.data !== "object") return;
    const { type, data } = event.data;

    if (type === "lead_form_request") {
      console.log("📩 Lead form trigger received:", data);
      openLeadForm(data);
    }
  });

  // ===== Responsive adjustments =====
  function adjustForMobile() {
    if (window.innerWidth < 500) {
      iframe.style.width = "90vw";
      iframe.style.height = "70vh";
      iframe.style.right = "5vw";
      iframe.style.bottom = "80px";
    } else {
      iframe.style.width = "400px";
      iframe.style.height = "560px";
      iframe.style.right = "24px";
      iframe.style.bottom = "90px";
    }
  }
  window.addEventListener("resize", adjustForMobile);
  adjustForMobile();

  // ===== Global Object Handle =====
  window.MyAiToolsetWidget = {
    toggle: toggleChat,
    open: () => {
      iframe.style.display = "block";
      setTimeout(() => (iframe.style.opacity = 1), 20);
    },
    close: () => {
      iframe.style.opacity = 0;
      setTimeout(() => (iframe.style.display = "none"), 200);
    },
    openLeadForm,
  };

  console.log("✅ MyAiToolset widget loader (Lead Lock Mode) active");
})();
