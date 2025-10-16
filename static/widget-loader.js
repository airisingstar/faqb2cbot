(function () {
  if (window.MyAiToolsetWidget) return;

  function initWidget() {
    const scriptTag = document.currentScript;
    const tenantColor = scriptTag.dataset.color || "#3B82F6";

    const bubble = document.createElement('div');
    bubble.id = 'aitoolset-bubble';
    bubble.innerHTML = '💬';
    Object.assign(bubble.style, {
      position: 'fixed',
      bottom: '24px',
      right: '24px',
      width: '56px',
      height: '56px',
      background: tenantColor,
      color: '#fff',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '26px',
      cursor: 'pointer',
      zIndex: 999999,
      boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
    });
    document.body.appendChild(bubble);

    const frame = document.createElement('iframe');
    frame.id = 'aitoolset-frame';
    frame.src = '/widget.html';
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
      zIndex: 999998,
      background: '#fff',
    });
    document.body.appendChild(frame);

    bubble.onclick = () => {
      frame.style.display = frame.style.display === 'none' ? 'block' : 'none';
    };

    window.MyAiToolsetWidget = { toggle: () => bubble.click() };
  }

  window.addEventListener('load', () => setTimeout(initWidget, 600));
})();
