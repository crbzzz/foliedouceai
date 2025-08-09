# -*- coding: utf-8 -*-
"""
FolieDouce – Build & Fix + Memory + KB (10/10) — VERSION LOGGÉE by CRBZZZ
- Ajout d'un système de logs détaillés (niveaux DEBUG/INFO/WARN/ERR)
- Timers contextuels et décorateur @timed
- Traces à chaque étape clé (détection, scaffold, LLM, validation, écriture)
- Messages pédagogiques affichés dans la console pour comprendre le flux

Dépendances: llama-cpp-python
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================
# LOGGING MINIMALISTE
# =========================

class Logger:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, level: str = None):
        env = (os.environ.get("FOLIEDOUCE_LOG", "") or "").upper()
        self.level = self.LEVELS.get(level or env or "DEBUG", 10)
        self._t0 = time.perf_counter()

    def _should(self, lvl: str) -> bool:
        return self.LEVELS[lvl] >= self.level

    def _stamp(self) -> str:
        now = time.perf_counter() - self._t0
        return f"+{now:8.3f}s"

    def _fmt_kv(self, kv: dict) -> str:
        if not kv:
            return ""
        parts = []
        for k, v in kv.items():
            try:
                parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
            except Exception:
                parts.append(f"{k}={v!r}")
        return " " + " ".join(parts)

    def log(self, lvl: str, msg: str, **kv):
        if self._should(lvl):
            print(f"[{lvl:<5}] {self._stamp()} | {msg}{self._fmt_kv(kv)}")

    def debug(self, msg: str, **kv): self.log("DEBUG", msg, **kv)
    def info (self, msg: str, **kv): self.log("INFO",  msg, **kv)
    def warn (self, msg: str, **kv): self.log("WARN",  msg, **kv)
    def error(self, msg: str, **kv): self.log("ERROR", msg, **kv)

LOG = Logger()  # niveau par défaut DEBUG (surchangeable via FOLIEDOUCE_LOG)

class Timer:
    """Contexte de mesure simple."""
    def __init__(self, label: str):
        self.label = label
        self.t0 = 0.0

    def __enter__(self):
        LOG.debug("⏱️ START", step=self.label)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc:
            LOG.error("⏱️ FAIL", step=self.label, duration=f"{dt:.3f}s", error=str(exc))
        else:
            LOG.info("⏱️ DONE", step=self.label, duration=f"{dt:.3f}s")

def timed(label: str):
    """Décorateur pour timer les fonctions."""
    def deco(fn):
        async def _aw(*a, **kw):
            with Timer(label):
                return await fn(*a, **kw)
        def _sw(*a, **kw):
            with Timer(label):
                return fn(*a, **kw)
        return _aw if asyncio.iscoroutinefunction(fn) else _sw
    return deco

# =========================
# CONSTANTES & CHEMINS
# =========================

ROOT = Path.cwd()
WORKDIR = Path(os.environ.get("FOLIEDOUCE_WORKDIR", str(ROOT / "fd_workspace"))); WORKDIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(os.environ.get("FOLIEDOUCE_MODELS_DIR", str(ROOT / "models")))
KB_DIR = Path(os.environ.get("FOLIEDOUCE_KB_DIR", str(ROOT / "kb"))); KB_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH = WORKDIR / "mem.json"

LOG.info("Chemins init",
         ROOT=str(ROOT),
         WORKDIR=str(WORKDIR),
         MODELS_DIR=str(MODELS_DIR),
         KB_DIR=str(KB_DIR),
         MEM_PATH=str(MEM_PATH))

# =========================
# MEMOIRE
# =========================

def ensure_dir(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def load_mem() -> dict:
    if MEM_PATH.exists():
        try:
            data = json.loads(MEM_PATH.read_text(encoding="utf-8"))
            LOG.debug("MEM load ok", size=len(json.dumps(data)))
            return data
        except Exception as e:
            LOG.warn("MEM load failed, fallback {}", error=str(e))
            return {}
    LOG.debug("MEM new (no file)")
    return {}

def save_mem(mem: dict) -> None:
    ensure_dir(MEM_PATH)
    MEM_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.debug("MEM saved", keys=list(mem.keys()), path=str(MEM_PATH))

def mem_get(key: str, default=None):
    val = load_mem().get(key, default)
    LOG.debug("MEM get", key=key, value=val)
    return val

def mem_set(**kwargs):
    LOG.debug("MEM set", **kwargs)
    m = load_mem(); m.update(kwargs); save_mem(m)

def write_file(p: Path, content: str):
    ensure_dir(p)
    p.write_text(content, encoding="utf-8")
    LOG.info("✍️ wrote file", path=str(p), bytes=len(content.encode("utf-8")))

# =========================
# LLM (llama.cpp)
# =========================

try:
    from llama_cpp import Llama
except Exception as e:
    raise SystemExit("pip install llama-cpp-python — " + str(e))

def _largest_model_in_models_dir() -> Path:
    ggufs = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if not ggufs:
        raise FileNotFoundError(f"Aucun .gguf trouvé dans {MODELS_DIR}.")
    LOG.info("Model auto-picked (largest)", model=str(ggufs[0].name), size=ggufs[0].stat().st_size)
    return ggufs[0]

MODEL_CODE_PATH = MODELS_DIR / "DeepSeek-Coder-V2-Lite-Instruct-Q8_1.gguf"  # code
MODEL_TEXT_PATH = MODELS_DIR / "gpt5o-reflexion-q-agi-llama-3.1-8b-i1-Q4_K_M.gguf"  # texte

CTX = int(os.environ.get("FOLIEDOUCE_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("FOLIEDOUCE_N_GPU", "0"))

_code_path = MODEL_CODE_PATH if MODEL_CODE_PATH.exists() else _largest_model_in_models_dir()
_text_path = MODEL_TEXT_PATH if MODEL_TEXT_PATH.exists() else _code_path

LOG.info("LLM paths", code=str(_code_path), text=str(_text_path), ctx=CTX, n_gpu=N_GPU_LAYERS)

with Timer("LLM init (code)"):
    llm_code = Llama(
        model_path=str(_code_path),
        n_ctx=CTX,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=False,
        verbose=False,
    )
with Timer("LLM init (text)"):
    llm_text = Llama(
        model_path=str(_text_path),
        n_ctx=CTX,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=False,
        verbose=False,
    )
_llm = llm_code  # compat

CODE_BLOCK_RE = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.IGNORECASE)
EXT_BY_LANG = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css",
    "lua": ".lua",
    "csharp": ".cs", "java": ".java", "go": ".go", "rust": ".rs",
}

# ---------- 10/10: CONTRAT DE SORTIE ----------
OUTPUT_CONTRACT = """
Commence par un court paragraphe d'introduction **en gras** (ou *italique*) qui met en contexte la solution.
Puis renvoie EXACTEMENT les sections suivantes:

### Résumé
- 3 à 6 puces: fonctionnalité et périmètre

### Sécurité & Framework
- Framework détecté: {fw}
- Règles clés (serveur d'abord, validation inventaire/argent, events 'fd:*', anti-injection)

### API exposée
- Events serveur écoutés (fd:*)
- Events client émis (fd:*)
- Callbacks si utilisés

### Fichiers
Renvoyer des blocs de code nommés avec titre, un bloc par fichier:
```{lang} title=fxmanifest.lua
...code...
```
```{lang} title=config.lua
...code...
```
```{lang} title=server/main.lua
...code...
```
```{lang} title=client/main.lua
...code...
```

Termine par un court paragraphe de conclusion (note d'usage / rappel).

Règles OBLIGATOIRES:
- Tous les events que TU définis doivent commencer par 'fd:'.
- Ne JAMAIS mettre des prix/IDs en dur dans le code; utilise config.lua.
- Toute logique critique (argent, inventaire) doit être côté serveur.
- Le code doit contenir des commentaires concis et utiles (pas du blabla).
- Respecter strictement l'API du framework (détaillée dans KB/MEM).
"""

EVENT_PREFIX = "fd:"

def _uses_config_for_prices(text: str) -> bool:
    return "Config." in text or "CONFIG." in text or "Config[" in text

def _events_prefixed(text: str) -> bool:
    bad = re.findall(r"Register(?:Server)?Event\(['\"](?!fd:)[^'\"]+", text)
    return len(bad) == 0

def _no_gta_native_in_redm(code: str) -> bool:
    gta_signatures = ["DLC_HEIST", "mini@drinking@coffee", "WEAPON_"]
    return not any(sig.lower() in code.lower() for sig in gta_signatures)

def _enforce_event_prefix(code: str) -> str:
    def repl_srv(m):
        name = m.group(1)
        if name.startswith(EVENT_PREFIX) or name.startswith("player"):
            return m.group(0)
        return f"RegisterServerEvent('{EVENT_PREFIX}{name}')"
    def repl_net(m):
        name = m.group(1)
        if name.startswith(EVENT_PREFIX) or name.startswith("player"):
            return m.group(0)
        return f"RegisterNetEvent('{EVENT_PREFIX}{name}')"
    code = re.sub(r"RegisterServerEvent\(['\"]([^'\"]+)['\"]\)", repl_srv, code)
    code = re.sub(r"RegisterNetEvent\(['\"]([^'\"]+)['\"]\)", repl_net, code)
    return code

def _validate_and_autofix(bundle: dict, fw: str, is_redm: bool) -> Tuple[dict, List[str]]:
    LOG.debug("Validator start", files=list(bundle.keys()), fw=fw, is_redm=is_redm)
    warnings = []
    for k in ["server/main.lua", "client/main.lua"]:
        if k in bundle:
            if not _events_prefixed(bundle[k]):
                warnings.append(f"{k}: Events non préfixés → correction 'fd:'")
                bundle[k] = _enforce_event_prefix(bundle[k]); LOG.info("Auto-fix prefix", file=k)
    if "server/main.lua" in bundle and not _uses_config_for_prices(bundle["server/main.lua"]):
        warnings.append("server/main.lua: pas d'usage de Config.* détecté pour les constantes → à vérifier")
    if is_redm:
        for k in ["client/main.lua", "server/main.lua"]:
            if k in bundle and not _no_gta_native_in_redm(bundle[k]):
                warnings.append(f"{k}: natives/anim GTA détectées en RedM → corrige-les manuellement")
    LOG.debug("Validator end", warnings=warnings)
    return bundle, warnings

# ===== Chargement KB locale
def load_kb_text(max_bytes: int = 200_000) -> str:
    if not KB_DIR.exists(): return ""
    chunks: List[str] = []
    exts = {".md",".txt",".yml",".yaml",".lua"}
    total = 0
    count = 0
    for p in sorted(KB_DIR.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            try:
                data = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                LOG.warn("KB read failed", path=str(p), error=str(e)); continue
            if not data.strip(): continue
            size = len(data.encode("utf-8"))
            total += size
            if total > max_bytes: break
            chunks.append(f"\n# KB:{p.relative_to(KB_DIR)}\n{data}\n")
            count += 1
    LOG.info("KB loaded", files=count, bytes_total=total)
    return "\n".join(chunks)

def framework_hint_from_kb() -> str:
    mem = load_mem()
    if "framework_default" in mem:
        return str(mem["framework_default"])
    default_path = KB_DIR / "frameworks" / "default.txt"
    if default_path.exists():
        val = default_path.read_text(encoding="utf-8").strip().lower()
        if val in {"esx","qbcore","vorp","rsg"}:
            mem_set(framework_default=val)
            LOG.info("KB framework default", fw=val)
            return val
    return ""

# ===== Adapters (scaffolds)
@dataclass
class Adapter:
    name: str
    root: str
    files: Dict[str,str]
    main_path: str
    tests_path: Optional[str]
    post_cmds: List[str]
    test_cmd: Optional[str]

ADAPTERS: Dict[str, Adapter] = {
    "python": Adapter(
        "python","py_proj",
        files={"requirements.txt":"pytest>=7.0\n","py_proj/__init__.py":"","py_proj/main.py":"","tests/test_main.py":""},
        main_path="py_proj/main.py", tests_path="tests/test_main.py",
        post_cmds=["pip install -r requirements.txt"], test_cmd="pytest -q"
    ),
    "html": Adapter(
        "html","web_proj",
        files={"index.html":"","style.css":"","app.js":""},
        main_path="index.html", tests_path=None, post_cmds=[], test_cmd=None
    ),
    "typescript": Adapter(
        "typescript","ts_proj",
        files={"package.json":'{ "name":"ts-proj","version":"0.1.0","type":"module","scripts":{"test":"vitest run"}, "devDependencies":{"typescript":"^5.4.0","vitest":"^1.5.0"}}\n',
               "tsconfig.json":'{ "compilerOptions":{"target":"ES2022","module":"ES2022","strict":true,"outDir":"dist"},"include":["src","tests"]}\n',
               "src/main.ts":"","tests/main.test.ts":""},
        main_path="src/main.ts", tests_path="tests/main.test.ts", post_cmds=["npm i"], test_cmd="npm test"
    ),
    "lua_fivem": Adapter(
        "lua","fx_fivem_resource",
        files={
            "fxmanifest.lua": textwrap.dedent("""\
                fx_version 'cerulean'
                game 'gta5'

                author 'FolieDouce'
                description 'Resource générée par FolieDouce (FiveM Lua)'
                version '0.0.1'

                shared_scripts { 'config.lua' }
                client_scripts { 'client/main.lua' }
                server_scripts { '@oxmysql/lib/MySQL.lua', 'server/main.lua' }
            """),
            "config.lua": "-- configuration partagée\nConfig = {}\n",
            "client/main.lua": "-- client-side code ici\n",
            "server/main.lua": "-- server-side code ici\n",
            "README.md": "# Resource FiveM générée\nPlace le dossier dans resources et ajoute `ensure fx_fivem_resource` dans server.cfg.\n",
        },
        main_path="server/main.lua", tests_path=None, post_cmds=[], test_cmd=None
    ),
    "lua_redm": Adapter(
        "lua","fx_redm_resource",
        files={
            "fxmanifest.lua": textwrap.dedent("""\
                fx_version 'cerulean'
                game 'rdr3'
                rdr3_warning 'I acknowledge that this is a prerelease build of RedM, and I am aware my resources may become incompatible.'

                author 'FolieDouce'
                description 'Resource générée par FolieDouce (RedM Lua)'
                version '0.0.1'

                shared_scripts { 'config.lua' }
                client_scripts { 'client/main.lua' }
                server_scripts { 'server/main.lua' }
            """),
            "config.lua": "-- configuration partagée\nConfig = {}\n",
            "client/main.lua": "-- client-side code ici\n",
            "server/main.lua": "-- server-side code ici\n",
            "README.md": "# Resource RedM générée\nPlace le dossier et `ensure fx_redm_resource`.\n",
        },
        main_path="server/main.lua", tests_path=None, post_cmds=[], test_cmd=None
    ),
}

def detect_lang(goal: str) -> str:
    g = goal.lower()
    if any(k in g for k in ["lua","fivem","redm","esx","qbcore","vorp","rsg"]): 
        LOG.info("Lang detected", lang="lua", reason="keywords")
        return "lua"
    if "html" in g: LOG.info("Lang detected", lang="html"); return "html"
    if "typescript" in g or " ts" in g or "ts " in g: LOG.info("Lang detected", lang="typescript"); return "typescript"
    if "python" in g: LOG.info("Lang detected", lang="python"); return "python"
    default = mem_get("lang_default", "python")
    LOG.info("Lang fallback", lang=default)
    return default

def _choose_lua_adapter(goal: str) -> str:
    g = goal.lower()
    if any(k in g for k in ["redm","rdr3","vorp","rsg"]): 
        LOG.info("Adapter chosen", adapter="lua_redm", reason="keywords")
        return "lua_redm"
    fw_default = framework_hint_from_kb() or mem_get("framework_default", "")
    if fw_default in {"vorp","rsg"}:
        LOG.info("Adapter chosen", adapter="lua_redm", reason="KB/MEM framework_default")
        return "lua_redm"
    LOG.info("Adapter chosen", adapter="lua_fivem", reason="default")
    return "lua_fivem"

# ===== LLM helpers (avec KB en contexte)

def _prompt_with_kb(sys_prompt: str, user_prompt: str) -> str:
    kb = load_kb_text()
    kb_hint = (("\n[KB]\n" + kb + "\n") if kb else "")
    mem = json.dumps(load_mem(), ensure_ascii=False)
    full = f"[SYSTEM]\n{sys_prompt}\n{kb_hint}[MEM]\n{mem}\n[USER]\n{user_prompt}\n"
    LOG.debug("Prompt composed", sys_len=len(sys_prompt), kb=bool(kb), mem_len=len(mem), user_len=len(user_prompt))
    return full

def _llm_infer(eng: Llama, prompt: str, **gen):
    """Fine couche pour tracer les appels LLM."""
    params = dict(max_tokens=gen.get("max_tokens", 1024), temperature=gen.get("temperature", 0.2), top_p=gen.get("top_p", 0.9))
    LOG.info("LLM infer", max_tokens=params["max_tokens"], temperature=params["temperature"], top_p=params["top_p"])
    t0 = time.perf_counter()
    out = eng(prompt, **params)
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"]
    LOG.info("LLM infer done", tokens=len(text), seconds=f"{dt:.3f}")
    return text

def _llm_code(user_prompt: str, lang_hint: str) -> str:
    sys_prompt = (
        "Tu es un assistant de codage. Réponds par du code complet et auto-suffisant, "
        "dans un unique bloc ```lang si possible. Ajoute des commentaires (clairs, utiles). "
        "Respecte strictement les conventions de KB/MEM si présentes. "
        "Tu peux utiliser **gras**/*italique* et quelques emojis sobres pour la lisibilité."
    )
    full = _prompt_with_kb(sys_prompt, user_prompt + f"\nLangage cible: {lang_hint}\n")
    out = _llm_infer(llm_code, full, max_tokens=2048, temperature=0.2, top_p=0.9)
    m = CODE_BLOCK_RE.search(out)
    code = (m.group(2).strip() if m else out.strip())
    LOG.debug("LLM code extracted", found_block=bool(m), length=len(code))
    return code

def _llm_explain(user_prompt: str) -> str:
    sys_prompt = (
        "Tu es pédagogue. Donne une explication claire, structurée, concise, en français. "
        "Utilise des listes à puces quand pertinent. Appuie-toi sur KB/MEM si utile. "
        "Tu peux utiliser **gras**/*italique* pour mettre en avant des éléments clés."
    )
    full = _prompt_with_kb(sys_prompt, user_prompt)
    out = _llm_infer(llm_text, full, max_tokens=1024, temperature=0.2, top_p=0.9)
    LOG.debug("LLM explain length", length=len(out))
    return out.strip()

# ===== BUILD (retourne le texte final pour le chat, sections + fichiers titrés)

@timed("BUILD pipeline")
async def run_pipeline_build(goal: str, forced_lang: Optional[str], do_tests: bool, cancel_event: asyncio.Event) -> str:
    if cancel_event.is_set(): return "⏹️ Interrompu."
    LOG.info("BUILD start", goal=goal, forced_lang=forced_lang, do_tests=do_tests)

    lang_key = (forced_lang or "").lower().strip() or detect_lang(goal)
    mem_set(lang_default=lang_key)
    adapter_key = _choose_lua_adapter(goal) if lang_key == "lua" else lang_key
    if adapter_key not in ADAPTERS: 
        LOG.warn("Unknown adapter, fallback python", adapter=adapter_key)
        adapter_key = "python"
    adapter = ADAPTERS[adapter_key]
    proj = WORKDIR / adapter.root; proj.mkdir(parents=True, exist_ok=True)
    LOG.info("Scaffold target", adapter=adapter_key, root=str(proj))

    # heuristique framework préférée
    for fw in ["esx","qbcore","vorp","rsg"]:
        if fw in goal.lower(): mem_set(framework_default=fw); LOG.info("Framework pinned by goal", fw=fw); break

    # Scaffold
    with Timer("Scaffold files"):
        for rel, content in adapter.files.items():
            if cancel_event.is_set(): return "⏹️ Interrompu."
            write_file(proj / rel, content)

    # Post cmds (silence chat)
    with Timer("Post commands"):
        for c in adapter.post_cmds:
            if cancel_event.is_set(): return "⏹️ Interrompu."
            LOG.info("post-cmd run", cmd=c)
            try:
                cp = subprocess.run(c, cwd=proj, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                LOG.debug("post-cmd out", returncode=cp.returncode, bytes=len(cp.stdout or b""))
            except Exception as e:
                LOG.warn("post-cmd error", error=str(e))

    # ==== CONTRAT + Génération par fichier
    is_redm = adapter_key == "lua_redm"
    fw = mem_get("framework_default","")
    for k in ["esx","qbcore","vorp","rsg"]:
        if k in goal.lower(): fw = k; mem_set(framework_default=fw); LOG.info("Framework final", fw=fw); break

    header = _llm_explain(
        f"""Explique l'architecture générée pour: {goal}
        - Contexte: framework={fw or 'auto'}, redm={is_redm}
        - Rappelle les règles de sécurité (serveur d'abord, validation inventaire/argent, events fd:*)
        - Reste concis et utile."""
    )
    contract = OUTPUT_CONTRACT.format(fw=(fw or 'auto'), lang='lua' if adapter_key.startswith('lua_') else adapter.name)

    intro_text = "**✨ Aperçu rapide** — Je te présente d'abord l'idée, puis le code prêt à l'emploi."
    outro_text = "_Astuce_: tu peux adapter la config et regénérer à tout moment. N'hésite pas à demander une variante."

    if adapter_key.startswith("lua_"):
        # server
        srv_prompt = textwrap.dedent(f"""
            {contract}

            Produis le contenu EXACT de `server/main.lua` pour:
            {goal}
            - Respect strict du framework {fw or 'auto'} (KB/MEM)
            - Events préfixés fd:
            - Utilise Config.* pour constantes
            - Commentaires concis et utiles
        """).strip()
        with Timer("LLM generate server/main.lua"):
            code_srv = _llm_code(srv_prompt, "lua")

        # client (optionnel)
        need_client = any(k in goal.lower() for k in ["ui","menu","nui","notification","client","marker","zone","blip"])
        code_cli = ""
        if need_client:
            cli_prompt = textwrap.dedent(f"""
                {contract}

                Produis le contenu EXACT de `client/main.lua` pour compléter `server/main.lua`.
                - RegisterNetEvent/TriggerServerEvent
                - Events fd:
                - Commentaires concis
            """).strip()
            with Timer("LLM generate client/main.lua"):
                code_cli = _llm_code(cli_prompt, "lua")
        else:
            LOG.info("Client part skipped (not requested)")

        # fxmanifest + config
        fxm_prompt = textwrap.dedent(f"""
            {contract}

            Produis un `fxmanifest.lua` correct pour {'RedM' if is_redm else 'FiveM'} (scripts client/server/config).
        """).strip()
        with Timer("LLM generate fxmanifest.lua"):
            code_fxm = _llm_code(fxm_prompt, "lua")

        cfg_prompt = textwrap.dedent(f"""
            {contract}

            Produis `config.lua` pour centraliser constantes/paramètres (ex: prix, noms d'item, etc.) pour:
            {goal}
        """).strip()
        with Timer("LLM generate config.lua"):
            code_cfg = _llm_code(cfg_prompt, "lua")

        bundle = {"fxmanifest.lua": code_fxm, "config.lua": code_cfg, "server/main.lua": code_srv}
        if need_client: bundle["client/main.lua"] = code_cli

        with Timer("Validation & Auto-fix"):
            bundle, warns = _validate_and_autofix(bundle, fw, is_redm)

        # Écriture + réponse
        blocks = []
        with Timer("Write generated files"):
            for fname, code in bundle.items():
                write_file(proj / fname, code)
                lang = "lua"
                blocks.append(f"```{lang} title={fname}\n{code}\n```")

        parts = [
            intro_text,
            "### Résumé\n" + header,
            "### Sécurité & Framework\n"
            f"- Framework détecté: {fw or 'auto'}\n"
            "- Serveur prioritaire pour l'argent/inventaire\n- Événements préfixés `fd:`\n- Pas d'injection client\n",
            "### API exposée\n"
            "- Voir les events et callbacks dans les fichiers générés (titres ci-dessous)\n",
            "### Fichiers\n" + "\n\n".join(blocks),
            f"**Conclusion.** {outro_text}"
        ]
        if warns:
            parts.append("### Warnings du validateur\n- " + "\n- ".join(warns))
        final_text = "\n\n".join(parts)
        LOG.info("BUILD done", project=str(proj))
        return final_text

    # ==== Non-Lua (générique)
    exp = _llm_explain(f"Explique en 4-6 points la solution pour: {goal}")
    instr = textwrap.dedent(f"""
        {OUTPUT_CONTRACT.format(fw='n/a', lang=adapter.name)}

        Donne le contenu EXACT de `{adapter.main_path}`.
    """).strip()
    with Timer("LLM generate main file (non-Lua)"):
        code_main = _llm_code(instr, adapter.name)
    write_file(proj / adapter.main_path, code_main)
    block = f"```{adapter.name} title={adapter.main_path}\n{code_main}\n```"
    final_text = "\n\n".join([
        intro_text,
        "### Résumé\n" + exp,
        "### Fichiers\n" + block,
        f"**Conclusion.** {outro_text}"
    ])
    LOG.info("BUILD done", project=str(proj))
    return final_text

# ===== FIX (retourne explications + code corrigé commenté + auto-fix)

def _extract_code_blocks(text: str) -> List[tuple[str,str]]:
    blks = []
    for m in CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").lower().strip()
        code = m.group(2).strip()
        blks.append((lang, code))
    if blks:
        LOG.debug("Code blocks extracted", count=len(blks))
        return blks
    guess = mem_get("lang_default","python")
    if "RegisterNetEvent" in text or "TriggerServerEvent" in text: guess = "lua"
    LOG.debug("No fenced blocks, guess lang", guess=guess)
    return [(guess, text.strip())]

def _framework_from_code(code: str) -> str:
    c = code.lower()
    if "exports['es_extended']" in c or "esx." in c: return "esx"
    if "exports['qb-core']" in c or "qbcore." in c: return "qbcore"
    if "exports.vorp_core" in c or "vorp" in c: return "vorp"
    if "exports['rsg-core']" in c or "rsgcore" in c: return "rsg"
    return mem_get("framework_default", framework_hint_from_kb())

@timed("FIX pipeline")
async def run_fix(payload: str, cancel_event: asyncio.Event) -> str:
    if cancel_event.is_set(): return "⏹️ Interrompu."
    blocks = _extract_code_blocks(payload)
    explanations: List[str] = []
    outputs: List[str] = []

    LOG.info("FIX start", blocks=len(blocks))
    ts = int(time.time())

    for i, (lang, code) in enumerate(blocks, start=1):
        if cancel_event.is_set(): return "⏹️ Interrompu."
        fw = _framework_from_code(code) if lang == "lua" else ""
        if fw: mem_set(framework_default=fw)
        LOG.info("Analyze block", index=i, lang=lang, fw=fw or "n/a", code_len=len(code))

        # Analyse guidée
        explain_prompt = textwrap.dedent(f"""
            Analyse ce code {lang}{' pour '+fw.upper() if fw else ''} à la lumière de KB/MEM.
            - Liste précise des risques/antipatterns (sécurité serveur, inventaire/argent, events prefix '{EVENT_PREFIX}', API framework)
            - Ce qui enfreint les conventions du framework
            - Plan de correction en 3-6 points, concis
        """) + f"\nCode:\n```{lang}\n{code}\n```"
        with Timer(f"LLM explain block {i}"):
            exp = _llm_explain(explain_prompt)
        explanations.append(exp)

        # Correction
        fix_prompt = textwrap.dedent(f"""
            Corrige le code {lang}{' pour '+fw.upper() if fw else ''} ci-dessous en respectant strictement KB/MEM et le contrat:
            - Events préfixés '{EVENT_PREFIX}'
            - Logique critique côté serveur
            - Utilise Config.* pour constantes
            - API framework exacte (pas d'API inventée)
            Rends UNIQUEMENT le code corrigé commenté dans un bloc:
            ```{lang}
            ...code...
            ```
        """) + f"\nCode:\n```{lang}\n{code}\n```"
        with Timer(f"LLM fix block {i}"):
            fixed = _llm_code(fix_prompt, lang or "python")

        # Auto-fix technique (préfixes, etc.)
        bundle = {"server/main.lua": fixed} if (lang=="lua") else {"main"+EXT_BY_LANG.get(lang or 'text', '.txt'): fixed}
        with Timer(f"Validate/Autofix block {i}"):
            bundle, warns = _validate_and_autofix(bundle, fw, fw in {"vorp","rsg"})
        fixed = bundle.get("server/main.lua", fixed) if (lang=="lua") else list(bundle.values())[0]
        if warns:
            LOG.warn("Validator warnings", index=i, warnings=warns)

        outputs.append(f"```{lang or 'text'}\n{fixed}\n```")

        # Écriture fichier preuve
        ext = EXT_BY_LANG.get(lang or "text", ".txt")
        name_fw = (fw or lang or 'code')
        out_path = WORKDIR / "fixes" / f"corrected_{name_fw}_{i}_{ts}{ext}"
        write_file(out_path, fixed)

    final = []
    if explanations:
        final.append("### Analyse & Corrections proposées\n" + "\n\n".join(explanations))
    if outputs:
        final.append("### Code corrigé\n" + "\n\n".join(outputs))
    body = "\n\n".join(final) if final else "Rien à corriger."

    intro = "**🔧 Correction proposée** — Voici le diagnostic suivi du code corrigé."
    outro = "_Note_: garde les événements `fd:` et la logique sensible côté serveur."

    final_text = "\n\n".join([intro, body, f"**Conclusion.** {outro}"])
    LOG.info("FIX done")
    return final_text


# =========================
# CLI (facultatif)
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FolieDouce — Build/Fix avec logs détaillés")
    parser.add_argument("mode", choices=["build","fix"], help="Mode d'exécution")
    parser.add_argument("--goal", default="", help="Objectif (pour build)")
    parser.add_argument("--lang", default="", help="Langage forcé (ex: lua/python)")
    parser.add_argument("--do-tests", action="store_true", help="Exécuter les tests si disponibles (build)")
    parser.add_argument("--payload", default="", help="Texte ou fichier pour fix (si commence par @, lit le fichier)")
    args = parser.parse_args()

    LOG.info("CLI args", **vars(args))

    cancel = asyncio.Event()
    if args.mode == "build":
        if not args.goal:
            LOG.error("Goal requis pour build"); raise SystemExit(2)
        out = asyncio.run(run_pipeline_build(args.goal, args.lang or None, args.do_tests, cancel))
        print("\n" + "="*80 + "\n" + out + "\n" + "="*80 + "\n")
    else:
        text = args.payload
        if text.startswith("@") and len(text) > 1:
            p = Path(text[1:])
            if not p.exists():
                LOG.error("Payload file introuvable", path=str(p)); raise SystemExit(2)
            text = p.read_text(encoding="utf-8")
            LOG.info("Payload lu depuis fichier", path=str(p), bytes=len(text.encode("utf-8")))
        elif not text:
            LOG.error("Payload requis pour fix"); raise SystemExit(2)
        out = asyncio.run(run_fix(text, cancel))
        print("\n" + "="*80 + "\n" + out + "\n" + "="*80 + "\n")
