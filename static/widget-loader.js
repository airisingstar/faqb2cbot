// static/widget-loader.js
(function () {
  if (window.MyAiToolsetWidget) return;

  function initWidget() {
    const existing = document.getElementById('aitoolset-frame');
    if (existing) existing.remove();

    // Create chat bubble
    const bubble = document.createElement('div');
    bubble.id = 'aitoolset-bubble';
    bubble.innerHTML = '💬';
    Object.assign(bubble.style, {
      position: 'fixed',
      bottom: '24px',
      right: '24px',
      width: '56px',
      height: '56px',
      background: '#3B82F6',
      color: '#fff',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '26px',
      cursor: 'pointer',
      zIndex: '999999 !important',
      boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
      transition: 'all 0.3s ease',
    });
    document.body.appendChild(bubble);

    // Create iframe chat window
    const frame = document.createElement('iframe');
    frame.id = 'aitoolset-frame';
    frame.src = 'https://faqbot-4how.onrender.com/widget.html';
    Object.assign(frame.style, {
      position: 'fixed',
      bottom: '90px',
      right: '24px',
      width: '420px',
      height: '560px',
      border: 'none',
      borderRadius: '16px',
      boxShadow: '0 8px 30px rgba(0,0,0,0.25)',
      display: 'none',
      zIndex: '999998 !important',
      background: '#fff',
    });
    document.body.appendChild(frame);

    // Show/hide logic
    bubble.addEventListener('click', () => {
      frame.style.display = frame.style.display === 'none' ? 'block' : 'none';
    });

    // Safety watcher (prevents CSS hiding)
    const observer = new MutationObserver(() => {
      if (bubble && bubble.style.display === 'none') {
        bubble.style.display = 'flex';
      }
    });
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });

    // Public API
    window.MyAiToolsetWidget = {
      toggle: () => bubble.click(),
    };
  }

  // Slight delay to wait for host page load
  window.addEventListener('load', () => setTimeout(initWidget, 600));
})();
