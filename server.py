from __future__ import annotations
import asyncio, json, os, re, uuid
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ==== Chemins ====
BASE_DIR = Path(__file__).resolve().parent
WEB_UI_DIR = BASE_DIR / "web_ui_newgen"
MODELS_DIR = BASE_DIR / "models"

# ==== App FastAPI ====
app = FastAPI(title="FolieDouce – Chat (SSE + Build/Fix)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques
app.mount("/static", StaticFiles(directory=WEB_UI_DIR), name="static")

@app.get("/")
async def serve_index():
    index_file = WEB_UI_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(404, "web_ui/index.html introuvable")
    return FileResponse(index_file)

# ==== LLM (llama-cpp) ====
try:
    from llama_cpp import Llama
except Exception as e:
    raise SystemExit("pip install llama-cpp-python — " + str(e))

def find_model_path() -> Path:
    envp = os.environ.get("FOLIEDOUCE_MODEL")
    if envp and Path(envp).exists():
        return Path(envp)
    ggufs = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if ggufs:
        return ggufs[0]
    raise FileNotFoundError("Aucun .gguf dans ./models. Place ton modèle ou set FOLIEDOUCE_MODEL.")

MODEL_PATH = find_model_path()
CTX = int(os.environ.get("FOLIEDOUCE_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("FOLIEDOUCE_N_GPU", "0"))

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=CTX,
    n_gpu_layers=N_GPU_LAYERS,
    logits_all=False,
    verbose=False,
)

# ==== Détection de langage + frameworks Lua ====
LANG_ALIASES = {
    "python": ["python", "py"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "html": ["html"],
    "css": ["css"],
    "lua": ["lua", "fivem", "redm", "nui"],
    "csharp": ["c#", "csharp", ".net", "dotnet"],
    "java": ["java"],
    "go": ["go", "golang"],
    "rust": ["rust"],
}
def detect_lang(text: str) -> Optional[str]:
    t = text.lower()
    for lang, aliases in LANG_ALIASES.items():
        for a in aliases:
            if a == "c#" and "c#" in t: return "csharp"
            if a == ".net" and ".net" in t: return "csharp"
            if re.search(rf"\b{re.escape(a)}\b", t): return lang
    return None

FRAMEWORK_ALIASES = {
    "esx": ["esx"],
    "qbcore": ["qb", "qbcore", "qbus"],
    "vorp": ["vorp"],
    "rsg": ["rsg"],
}
def detect_framework(text: str) -> Optional[str]:
    t = text.lower()
    for fw, aliases in FRAMEWORK_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", t):
                return fw
    return None

# ---- Build & Fix intent
FORCE_BUILD_PREFIX = "!!build"
BUILD_KEYWORDS = [
    "génère","genere","crée","cree","construis","build","écris","ecris",
    "fais une page","fais un projet","écrire un fichier","ecrire un fichier","générer un fichier",
    "classe","class","fonction","function","script","resource","fxmanifest",
    "client/main.lua","server/main.lua"
]
def is_build_intent(text: str) -> bool:
    t = text.lower().strip()
    if t.startswith(FORCE_BUILD_PREFIX): return True
    return any(kw in t for kw in BUILD_KEYWORDS)

FIX_KEYWORDS = ["corrige","fix","bug","erreur","optimize","optimise","refactor","refactore","refactorise"]
CODE_BLOCK_RE = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.IGNORECASE)
def is_fix_intent(text: str) -> bool:
    t = text.lower()
    if CODE_BLOCK_RE.search(text): return True
    return any(kw in t for kw in FIX_KEYWORDS)

SYSTEM_BASE = (
    "Tu es une IA locale. Si un langage est mentionné, réponds dans ce langage. "
    "Rends des réponses pédagogiques et concises, avec du code commenté. "
    "Toujours entourer le code d’un bloc triple backticks avec un tag, ex: ```python```, ```javascript```, ```html```, ```css```, ```lua```."
)

STOP = ["\n\n", "[USER]", "[SYSTEM]", "</s>"]

# ==== Modèles Pydantic ====
class ChatMsg(BaseModel):
    role: str
    content: str

class ChatBeginRequest(BaseModel):
    messages: List[ChatMsg]

class ChatBeginResponse(BaseModel):
    id: str

class ChatStopRequest(BaseModel):
    id: str

# ==== Prompt builder ====
def build_system(messages: List[ChatMsg]) -> str:
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    text = last_user.content if last_user else ""
    lang = detect_lang(text)
    fw = detect_framework(text)

    base = SYSTEM_BASE
    if lang == "lua":
        base += (
            " Tu es un expert Lua pour FiveM/RedM. Bonnes pratiques: "
            "séparer client/server/config, logique critique côté serveur, events (fd:*), fxmanifest correct. "
            "Renvoyer le code en ```lua``` et ajouter des commentaires utiles."
        )
        if fw == "esx":
            base += " Framework: ESX. Utilise ESX.GetPlayerFromId, ESX.RegisterUsableItem, ESX.RegisterCommand, callbacks ESX."
        elif fw == "qbcore":
            base += " Framework: QBCore. Utilise exports['qb-core']:GetCoreObject(), QBCore.Functions.CreateUseableItem, RegisterNetEvent."
        elif fw == "vorp":
            base += " Framework: VORP (RedM). Utilise exports.vorp_core, événements VORP, best practices RedM."
        elif fw == "rsg":
            base += " Framework: RSG (RedM). Utilise exports RSG et events server/client adéquats."
    elif lang in {"html","css","javascript"}:
        base += " Si app web, renvoie HTML/CSS/JS séparés ou HTML autonome, et commente le code."
    elif lang:
        base += f" Langage détecté: {lang}. Réponds en {lang} avec commentaires."
    return base

def to_prompt(msgs: List[ChatMsg]) -> str:
    system = build_system(msgs)
    parts = [f"[SYSTEM]\n{system}\n"]
    for m in msgs:
        tag = "[USER]" if m.role == "user" else "[ASSISTANT]"
        parts.append(f"{tag}\n{m.content}\n")
    return "\n".join(parts)

# ==== Sessions SSE & annulation ====
class Session:
    __slots__ = ("id", "messages", "cancel")
    def __init__(self, sid: str, messages: List[ChatMsg]) -> None:
        self.id = sid
        self.messages = messages
        self.cancel = asyncio.Event()

SESSIONS: Dict[str, Session] = {}

def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

@app.post("/chat/begin", response_model=ChatBeginResponse)
async def chat_begin(req: ChatBeginRequest):
    if not req.messages:
        raise HTTPException(400, "messages manquants")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(sid, req.messages)
    return ChatBeginResponse(id=sid)

@app.post("/chat/stop")
async def chat_stop(req: ChatStopRequest):
    s = SESSIONS.get(req.id)
    if s: s.cancel.set()
    return {"ok": True}

# ==== FolieDouce (Build/Fix) ====
import foliedouce as fd  # fichier à part

async def gen_llm_stream(session: Session) -> AsyncGenerator[bytes, None]:
    """Streaming simple : tokens du LLM (chat standard)."""
    prompt = to_prompt(session.messages)
    final = []
    for chunk in llm(prompt, max_tokens=1024, temperature=0.2, top_p=0.9, stop=STOP, stream=True):
        if session.cancel.is_set():
            yield sse({"type": "stopped"}); return
        tok = chunk["choices"][0]["text"]
        final.append(tok)
        yield sse({"type": "token", "text": tok})
        await asyncio.sleep(0)
    text = "".join(final).strip()
    session.messages.append(ChatMsg(role="assistant", content=text))
    yield sse({"type": "done", "text": text})

async def gen_build_done(session: Session) -> AsyncGenerator[bytes, None]:
    """Build: logs console + fichiers écrits; côté chat on renvoie seulement la synthèse + code commenté."""
    user = next((m for m in reversed(session.messages) if m.role == "user"), None)
    goal = user.content if user else ""
    if goal.lower().strip().startswith(FORCE_BUILD_PREFIX):
        goal = goal[len(FORCE_BUILD_PREFIX):].lstrip()
    lang = detect_lang(goal)
    cancel = session.cancel

    print("[server] build start | goal:", goal)
    text = await fd.run_pipeline_build(goal, forced_lang=lang, do_tests=False, cancel_event=cancel)
    if cancel.is_set():
        yield sse({"type":"stopped"}); print("[server] build cancelled"); return
    print("[server] build done")
    yield sse({"type":"done","text": text})

async def gen_fix_done(session: Session) -> AsyncGenerator[bytes, None]:
    """Fix: analyse + correction. Logs console; côté chat on renvoie explications + code corrigé commenté."""
    user = next((m for m in reversed(session.messages) if m.role == "user"), None)
    payload = user.content if user else ""
    cancel = session.cancel
    print("[server] fix start")
    text = await fd.run_fix(payload, cancel_event=cancel)
    if cancel.is_set():
        yield sse({"type":"stopped"}); print("[server] fix cancelled"); return
    print("[server] fix done")
    yield sse({"type":"done","text": text})

@app.get("/chat/stream")
async def chat_stream(id: str = Query(...)):
    session = SESSIONS.get(id)
    if not session: raise HTTPException(404, "session inconnue")

    last_user = next((m for m in reversed(session.messages) if m.role == "user"), None)
    text = last_user.content if last_user else ""
    raw_text = text
    if text.lower().strip().startswith(FORCE_BUILD_PREFIX):
        text = text[len(FORCE_BUILD_PREFIX):].lstrip()
    build_mode = is_build_intent(text)
    fix_mode = is_fix_intent(text)

    print(f"[server] request='{raw_text}' | cleaned='{text}' | build_mode={build_mode} | fix_mode={fix_mode}")

    async def stream():
        try:
            yield sse({"type": "typing", "text": "..."})
            if fix_mode:
                async for chunk in gen_fix_done(session): yield chunk
            elif build_mode:
                async for chunk in gen_build_done(session): yield chunk
            else:
                async for chunk in gen_llm_stream(session): yield chunk
        except Exception as e:
            yield sse({"type":"error","error":str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream")

# ==== Lancement ====
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
