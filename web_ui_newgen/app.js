const chatEl = document.getElementById('chat');
const form = document.getElementById('composer');
const input = document.getElementById('input');
const btnSend = document.getElementById('send');
const btnStop = document.getElementById('stop');

let eventSource = null, currentSessionId = null;

const state = { messages: [] };

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function render() {
    chatEl.innerHTML = '';
    for (let i = 0; i < state.messages.length; i++) {
        const m = state.messages[i];
        const row = el('div', 'msg ' + (m.role === 'user' ? 'user' : 'bot'));
        const bubble = el('div', 'bubble');
        
        // Add staggered animation delay
        row.style.animationDelay = `${i * 0.1}s`;
        
        if (m.role === 'assistant') {
            renderAssistant(bubble, m.content);
        } else {
            bubble.textContent = m.content;
        }
        
        row.appendChild(bubble);
        chatEl.appendChild(row);
    }
    
    // Smooth scroll to bottom
    requestAnimationFrame(() => {
        chatEl.scrollTo({
            top: chatEl.scrollHeight,
            behavior: 'smooth'
        });
    });
}

function renderAssistant(container, text) {
    const parts = splitByCode(text);
    for (const part of parts) {
        if (part.type === 'code') {
            const wrap = el('div', 'codeblock');
            const pre = document.createElement('pre');
            const code = document.createElement('code');
            
            if (part.lang) code.dataset.lang = part.lang;
            code.textContent = part.content;
            pre.appendChild(code);
            wrap.appendChild(pre);
            
            const copy = el('button', 'copy-btn', 'Copier');
            copy.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(part.content);
                    copy.textContent = 'Copié !';
                    copy.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        copy.textContent = 'Copier';
                        copy.style.transform = 'scale(1)';
                    }, 1200);
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
            };
            wrap.appendChild(copy);
            container.appendChild(wrap);
        } else if (part.content.trim()) {
            const p = document.createElement('p');
            p.textContent = part.content;
            p.style.margin = '0';
            container.appendChild(p);
        }
    }
}

function splitByCode(text) {
    const re = /```(\w+)?\n([\s\S]*?)```/g;
    let m, last = 0;
    const out = [];
    
    while ((m = re.exec(text)) !== null) {
        if (m.index > last) {
            out.push({ type: 'text', content: text.slice(last, m.index) });
        }
        out.push({ type: 'code', lang: (m[1] || '').toLowerCase(), content: m[2] });
        last = m.index + m[0].length;
    }
    
    if (last < text.length) {
        out.push({ type: 'text', content: text.slice(last) });
    }
    
    return out;
}

function addTyping() {
    const row = el('div', 'msg bot');
    const bubble = el('div', 'bubble');
    const t = el('span', 'typing');
    
    t.appendChild(el('span', 'dot'));
    t.appendChild(el('span', 'dot'));
    t.appendChild(el('span', 'dot'));
    
    bubble.appendChild(t);
    row.appendChild(bubble);
    chatEl.appendChild(row);
    
    // Smooth scroll to show typing indicator
    requestAnimationFrame(() => {
        chatEl.scrollTo({
            top: chatEl.scrollHeight,
            behavior: 'smooth'
        });
    });
    
    return row;
}

function appendStreaming(row, text) {
    const bubble = row.querySelector('.bubble');
    bubble.textContent = (bubble.textContent || '') + text;
    
    // Auto-scroll during streaming
    requestAnimationFrame(() => {
        chatEl.scrollTo({
            top: chatEl.scrollHeight,
            behavior: 'smooth'
        });
    });
}

function replaceWithFinal(row, text) {
    row.remove();
    state.messages.push({ role: 'assistant', content: text });
    render();
}

async function startSSE(messages) {
    try {
        const res = await fetch('/chat/begin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages })
        });
        
        if (!res.ok) throw new Error('HTTP ' + res.status);
        
        const { id } = await res.json();
        currentSessionId = id;

        const rowTyping = addTyping();
        btnSend.disabled = true;
        btnStop.hidden = false;
        
        // Add loading state to send button
        const sendIcon = btnSend.querySelector('.send-icon');
        const originalIcon = sendIcon.textContent;
        sendIcon.textContent = '⏳';
        
        eventSource = new EventSource(`/chat/stream?id=${encodeURIComponent(id)}`);

        let building = false;
        let bufferDuringBuild = '';

        eventSource.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data);
                
                if (data.type === 'typing') {
                    // Typing indicator already shown
                } else if (data.type === 'token') {
                    appendStreaming(rowTyping, data.text);
                } else if (data.type === 'log') {
                    building = true;
                    bufferDuringBuild += (bufferDuringBuild ? '\n' : '') + data.text;
                    appendStreaming(rowTyping, '\n' + data.text);
                } else if (data.type === 'file') {
                    building = true;
                    const line = `[fichier] ${data.path}`;
                    bufferDuringBuild += '\n' + line;
                    appendStreaming(rowTyping, '\n' + line);
                } else if (data.type === 'code') {
                    building = true;
                    const codeMsg = { role: 'assistant', content: data.text };
                    state.messages.push(codeMsg);
                    render();
                } else if (data.type === 'done') {
                    if (building) {
                        replaceWithFinal(rowTyping, bufferDuringBuild || data.text);
                    } else {
                        replaceWithFinal(rowTyping, data.text);
                    }
                    cleanup();
                } else if (data.type === 'stopped') {
                    replaceWithFinal(rowTyping, '⏹️ Interrompu.');
                    cleanup();
                } else if (data.type === 'error') {
                    replaceWithFinal(rowTyping, 'Erreur: ' + data.error);
                    cleanup();
                }
            } catch (e) {
                console.error('bad event', ev.data);
            }
        };

        eventSource.onerror = () => {
            replaceWithFinal(rowTyping, 'Erreur de streaming.');
            cleanup();
        };
        
        // Restore send button icon
        function cleanup() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            currentSessionId = null;
            btnSend.disabled = false;
            btnStop.hidden = true;
            sendIcon.textContent = originalIcon;
        }
        
    } catch (err) {
        state.messages.push({ role: 'assistant', content: 'Erreur: ' + err.message });
        render();
    }
}

function cleanup() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    currentSessionId = null;
    btnSend.disabled = false;
    btnStop.hidden = true;
}

// Auto-resize textarea
function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}

input.addEventListener('input', autoResize);

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    
    // Add visual feedback
    input.style.transform = 'scale(0.98)';
    setTimeout(() => {
        input.style.transform = 'scale(1)';
    }, 100);
    
    input.value = '';
    autoResize();
    
    state.messages.push({ role: 'user', content: text });
    render();
    
    try {
        await startSSE(state.messages);
    } catch (err) {
        state.messages.push({ role: 'assistant', content: 'Erreur: ' + err.message });
        render();
    }
});

btnStop.addEventListener('click', async () => {
    if (!currentSessionId) return;
    
    // Add visual feedback
    btnStop.style.transform = 'scale(0.95)';
    setTimeout(() => {
        btnStop.style.transform = 'scale(1)';
    }, 150);
    
    try {
        await fetch('/chat/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: currentSessionId })
        });
    } catch (err) {
        console.error('Error stopping chat:', err);
    }
    cleanup();
});

// Welcome message with animation
setTimeout(() => {
    state.messages.push({
        role: 'assistant',
        content: "Salut ! 👋 Tout se passe ici dans le chat."
    });
    render();
}, 500);

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        form.dispatchEvent(new Event('submit'));
    }
    
    if (e.key === 'Escape' && !btnStop.hidden) {
        btnStop.click();
    }
});

// Initialize textarea size
autoResize();