(() => {
  const BOT_URL = "https://faqbot-4how.onrender.com/widget.html";
  if (window.MyAiToolsetWidget) return;

  // Create bubble button
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
    boxShadow: "0 4px 20px rgba(0,0,0,0.25)"
  });
  document.body.appendChild(bubble);

  // Create iframe (hidden initially)
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
    zIndex: 999998,
    background: "#fff"
  });
  document.body.appendChild(iframe);

  // Toggle function
  bubble.onclick = () => {
    iframe.style.display = iframe.style.display === "block" ? "none" : "block";
  };

  window.MyAiToolsetWidget = {
    toggle: () => {
      bubble.click();
    },
    open: () => {
      iframe.style.display = "block";
    },
    close: () => {
      iframe.style.display = "none";
    }
  };

  console.log("✅ MyAiToolset widget loader is active");
})();
