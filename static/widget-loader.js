(function(){
  // ======= BASIC CONFIG (edit only these 3 lines) =======
  const CHAT_URL = "https://faqbot-4how.onrender.com/widget.html";  // your existing widget
  const bubbleColor = "#3B82F6";                                    // brand blue
  const position = "right";                                         // "right" | "left"
  // ======================================================

  // Prevent duplicate loads
  if (window.__myaitoolset_loaded) return; 
  window.__myaitoolset_loaded = true;

  // Create iframe (hidden by default)
  const iframe = document.createElement("iframe");
  iframe.id = "myaitoolset-widget";
  Object.assign(iframe.style, {
    position: "fixed",
    bottom: "90px",
    [position]: "24px",
    width: "400px",
    height: "560px",
    maxWidth: "92vw",
    maxHeight: "80vh",
    border: "none",
    borderRadius: "16px",
    boxShadow: "0 8px 30px rgba(0,0,0,.25)",
    zIndex: 999999,
    display: "none",
    background: "#fff"
  });
  iframe.src = CHAT_URL + (CHAT_URL.includes("?") ? "&" : "?") + "site=" + encodeURIComponent(location.hostname);
  document.addEventListener("DOMContentLoaded", ()=> document.body.appendChild(iframe));

  // Create floating bubble
  const bubble = document.createElement("button");
  bubble.id = "myaitoolset-bubble";
  bubble.type = "button";
  bubble.setAttribute("aria-label", "Open chat");
  bubble.innerHTML = "💬";
  Object.assign(bubble.style, {
    position: "fixed",
    bottom: "24px",
    [position]: "24px",
    width: "64px",
    height: "64px",
    borderRadius: "50%",
    background: bubbleColor,
    color: "#fff",
    fontSize: "26px",
    border: "none",
    cursor: "pointer",
    boxShadow: "0 6px 20px rgba(0,0,0,.3)",
    zIndex: 1000000,
    transition: "transform .2s ease, box-shadow .2s ease"
  });
  bubble.onmouseenter = ()=>{ bubble.style.transform="scale(1.05)"; bubble.style.boxShadow="0 8px 30px rgba(0,0,0,.4)"; };
  bubble.onmouseleave = ()=>{ bubble.style.transform="scale(1.0)"; bubble.style.boxShadow="0 6px 20px rgba(0,0,0,.3)"; };
  bubble.onclick = ()=>{
    const open = iframe.style.display !== "block";
    iframe.style.display = open ? "block" : "none";
    bubble.setAttribute("aria-expanded", open ? "true" : "false");
  };
  document.addEventListener("DOMContentLoaded", ()=> document.body.appendChild(bubble));

  // Optional “hint” toast
  setTimeout(()=>{
    const hint = document.createElement("div");
    hint.textContent = "Questions? Chat with us!";
    Object.assign(hint.style, {
      position:"fixed",
      bottom:"100px",
      [position]: "100px",
      background:"#fff",
      color:"#0f172a",
      border:"1px solid #cbd5e1",
      padding:"10px 14px",
      borderRadius:"10px",
      fontSize:"14px",
      boxShadow:"0 4px 14px rgba(0,0,0,.1)",
      zIndex:999998,
      opacity:"0",
      transition:"opacity .4s ease"
    });
    document.body.appendChild(hint);
    requestAnimationFrame(()=> hint.style.opacity = "1");
    setTimeout(()=>{
      hint.style.opacity = "0";
      setTimeout(()=> hint.remove(), 600);
    }, 5000);
  }, 3500);
})();
