const chatEl = document.getElementById('chat');
const form = document.getElementById('composer');
const input = document.getElementById('input');
const btnSend = document.getElementById('send');
const btnStop = document.getElementById('stop');

let eventSource = null, currentSessionId = null;

const state = { messages: [] };

function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e; }

function render(){
  chatEl.innerHTML=''; 
  for(const m of state.messages){
    const row=el('div','msg '+(m.role==='user'?'user':'bot'));
    const bubble=el('div','bubble');
    if(m.role==='assistant') renderAssistant(bubble,m.content); else bubble.textContent=m.content;
    row.appendChild(bubble); chatEl.appendChild(row);
  }
  chatEl.scrollTop=chatEl.scrollHeight;
}

function renderAssistant(container, text){
  const parts = splitByCode(text);
  for(const part of parts){
    if(part.type==='code'){
      const wrap=el('div','codeblock');
      const pre=document.createElement('pre'); const code=document.createElement('code');
      if(part.lang) code.dataset.lang=part.lang; code.textContent=part.content;
      pre.appendChild(code); wrap.appendChild(pre);
      const copy=el('button','copy-btn','Copier');
      copy.onclick=async()=>{ await navigator.clipboard.writeText(part.content); copy.textContent='Copié !'; setTimeout(()=>copy.textContent='Copier',1200); };
      wrap.appendChild(copy);
      container.appendChild(wrap);
    } else if(part.content.trim()){
      const p=document.createElement('p'); p.textContent=part.content; container.appendChild(p);
    }
  }
}

function splitByCode(text){
  const re=/```(\w+)?\n([\s\S]*?)```/g; let m,last=0; const out=[];
  while((m=re.exec(text))!==null){
    if(m.index>last) out.push({type:'text',content:text.slice(last,m.index)});
    out.push({type:'code',lang:(m[1]||'').toLowerCase(),content:m[2]});
    last=m.index+m[0].length;
  }
  if(last<text.length) out.push({type:'text',content:text.slice(last)});
  return out;
}

function addTyping(){
  const row=el('div','msg bot'); const bubble=el('div','bubble');
  const t=el('span','typing'); t.appendChild(el('span','dot')); t.appendChild(el('span','dot')); t.appendChild(el('span','dot'));
  bubble.appendChild(t); row.appendChild(bubble); chatEl.appendChild(row); chatEl.scrollTop=chatEl.scrollHeight; return row;
}
function appendStreaming(row, text){ row.querySelector('.bubble').textContent=(row.querySelector('.bubble').textContent||'')+text; chatEl.scrollTop=chatEl.scrollHeight; }
function replaceWithFinal(row, text){ row.remove(); state.messages.push({role:'assistant',content:text}); render(); }

async function startSSE(messages){
  const res = await fetch('/chat/begin',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages})});
  if(!res.ok) throw new Error('HTTP '+res.status); const {id}=await res.json(); currentSessionId=id;

  const rowTyping=addTyping(); btnSend.disabled=true; btnStop.hidden=false;
  eventSource = new EventSource(`/chat/stream?id=${encodeURIComponent(id)}`);

  let building = false; // si on reçoit des logs/files/codes → mode build
  let bufferDuringBuild = ''; // on remplit un message assistant “narration du bot”

  eventSource.onmessage=(ev)=>{
    try{
      const data=JSON.parse(ev.data);
      if(data.type==='typing'){ /* ignored, bulle déjà là */ }
      else if(data.type==='token'){ appendStreaming(rowTyping, data.text); }
      else if(data.type==='log'){ 
        building=true; bufferDuringBuild += (bufferDuringBuild?'\n':'') + data.text;
        appendStreaming(rowTyping, '\n' + data.text);
      }
      else if(data.type==='file'){ 
        building=true; const line=`[fichier] ${data.path}`; bufferDuringBuild += '\n'+line; appendStreaming(rowTyping, '\n'+line);
      }
      else if(data.type==='code'){ 
        building=true; // injecte immédiatement le code final dans le chat
        const codeMsg = { role:'assistant', content: data.text };
        state.messages.push(codeMsg); render();
      }
      else if(data.type==='done'){ 
        if(building){
          // Remplace la bulle typing par le récap des logs (POV bot)
          replaceWithFinal(rowTyping, bufferDuringBuild || data.text);
        }else{
          replaceWithFinal(rowTyping, data.text);
        }
        cleanup();
      }
      else if(data.type==='stopped'){ replaceWithFinal(rowTyping, '⏹️ Interrompu.'); cleanup(); }
      else if(data.type==='error'){ replaceWithFinal(rowTyping, 'Erreur: '+data.error); cleanup(); }
    }catch(e){ console.error('bad event', ev.data); }
  };
  eventSource.onerror=()=>{ replaceWithFinal(rowTyping,'Erreur de streaming.'); cleanup(); };
}
function cleanup(){ if(eventSource){eventSource.close();eventSource=null;} currentSessionId=null; btnSend.disabled=false; btnStop.hidden=true; }

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const text=input.value.trim(); if(!text) return; input.value='';
  state.messages.push({role:'user',content:text}); render();
  try{ await startSSE(state.messages); }catch(err){ state.messages.push({role:'assistant',content:'Erreur: '+err.message}); render(); }
});
btnStop.addEventListener('click', async ()=>{
  if(!currentSessionId) return;
  try{ await fetch('/chat/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:currentSessionId})}); }catch{}
  cleanup();
});

// Welcome
state.messages.push({role:'assistant',content:"Salut ! Tout se passe ici dans le chat. Dis par ex. « en javascript fais une todolist » ou « génère une page HTML/CSS/JS avec un formulaire »."});
render();
