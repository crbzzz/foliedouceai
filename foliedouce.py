"""
FolieDouce – Build & Fix + Memory + KB (10/10)
- BUILD : génère des fichiers (logs console uniquement), renvoie au chat: Résumé + Fichiers (blocs titrés)
- FIX   : corrige du code (détection auto) avec règles par framework (ESX/QBCore/VORP/RSG)
- KB    : lit ./kb et ./kb/frameworks pour orienter génération/correction
- MEM   : persiste préférences & contexte dans fd_workspace/mem.json
- 10/10 : contrat de sortie strict + validateurs + auto-fix (préfixe 'fd:', config, GTA vs RedM)
"""

from __future__ import annotations
import asyncio, json, os, re, shlex, subprocess, textwrap, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from llama_cpp import Llama
except Exception as e:
    raise SystemExit("pip install llama-cpp-python — " + str(e))

ROOT = Path.cwd()
WORKDIR = Path(os.environ.get("FOLIEDOUCE_WORKDIR", str(ROOT / "fd_workspace"))); WORKDIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(os.environ.get("FOLIEDOUCE_MODELS_DIR", str(ROOT / "models")))
KB_DIR = Path(os.environ.get("FOLIEDOUCE_KB_DIR", str(ROOT / "kb"))); KB_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH = WORKDIR / "mem.json"

def load_mem() -> dict:
    if MEM_PATH.exists():
        try: return json.loads(MEM_PATH.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}
def save_mem(mem: dict) -> None:
    ensure_dir(MEM_PATH); MEM_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
def mem_get(key: str, default=None):
    return load_mem().get(key, default)
def mem_set(**kwargs):
    m = load_mem(); m.update(kwargs); save_mem(m)

def find_model_path() -> Path:
    envp = os.environ.get("FOLIEDOUCE_MODEL")
    if envp and Path(envp).exists(): return Path(envp)
    ggufs = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if ggufs: return ggufs[0]
    default = ROOT / "models" / "DeepSeek-Coder-V2-Lite-Instruct-Q8_1.gguf"
    if default.exists(): return default
    raise FileNotFoundError("Aucun .gguf trouvé.")

MODEL_PATH = find_model_path()
CTX = int(os.environ.get("FOLIEDOUCE_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("FOLIEDOUCE_N_GPU", "0"))
_llm = Llama(model_path=str(MODELS_DIR / MODEL_PATH.name), n_ctx=CTX, n_gpu_layers=N_GPU_LAYERS, logits_all=False, verbose=False)

# ===== Helpers
def ensure_dir(p: Path): p.parent.mkdir(parents=True, exist_ok=True)
def write_file(p: Path, content: str): ensure_dir(p); p.write_text(content, encoding="utf-8"); print("[fd] wrote", p)

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
Tu DOIS renvoyer au chat au format suivant:

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

Règles OBLIGATOIRES:
- Tous les events que TU définis doivent commencer par 'fd:'.
- Ne JAMAIS mettre des prix/IDs en dur dans le code; utilise config.lua.
- Toute logique critique (argent, inventaire) doit être côté serveur.
- Le code doit contenir des commentaires concis et utiles (pas du blabla).
- Respecter strictement l'API du framework (détaillée dans KB/MEM).
"""

# ---------- 10/10: VALIDATION & AUTOFIX ----------
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

def _validate_and_autofix(bundle: dict, fw: str, is_redm: bool) -> tuple[dict, list]:
    warnings = []
    for k in ["server/main.lua", "client/main.lua"]:
        if k in bundle:
            if not _events_prefixed(bundle[k]):
                warnings.append(f"{k}: Events non préfixés → correction 'fd:'")
                bundle[k] = _enforce_event_prefix(bundle[k])
    if "server/main.lua" in bundle and not _uses_config_for_prices(bundle["server/main.lua"]):
        warnings.append("server/main.lua: pas d'usage de Config.* détecté pour les constantes → à vérifier")
    if is_redm:
        for k in ["client/main.lua", "server/main.lua"]:
            if k in bundle and not _no_gta_native_in_redm(bundle[k]):
                warnings.append(f"{k}: natives/anim GTA détectées en RedM → corrige-les manuellement")
    return bundle, warnings

# ===== Chargement KB locale
def load_kb_text(max_bytes: int = 200_000) -> str:
    if not KB_DIR.exists(): return ""
    chunks: List[str] = []
    exts = {".md",".txt",".yml",".yaml",".lua"}
    total = 0
    for p in sorted(KB_DIR.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            try:
                data = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not data.strip(): continue
            total += len(data.encode("utf-8"))
            if total > max_bytes: break
            chunks.append(f"\n# KB:{p.relative_to(KB_DIR)}\n{data}\n")
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
    if any(k in g for k in ["lua","fivem","redm","esx","qbcore","vorp","rsg"]): return "lua"
    if "html" in g: return "html"
    if "typescript" in g or " ts" in g or "ts " in g: return "typescript"
    if "python" in g: return "python"
    return mem_get("lang_default", "python")

def _choose_lua_adapter(goal: str) -> str:
    g = goal.lower()
    if any(k in g for k in ["redm","rdr3","vorp","rsg"]): return "lua_redm"
    fw_default = framework_hint_from_kb() or mem_get("framework_default", "")
    if fw_default in {"vorp","rsg"}: return "lua_redm"
    return "lua_fivem"

# ===== LLM helpers (avec KB en contexte)
def _prompt_with_kb(sys_prompt: str, user_prompt: str) -> str:
    kb = load_kb_text()
    kb_hint = (("\n[KB]\n" + kb + "\n") if kb else "")
    mem = json.dumps(load_mem(), ensure_ascii=False)
    return f"[SYSTEM]\n{sys_prompt}\n{kb_hint}[MEM]\n{mem}\n[USER]\n{user_prompt}\n"

def _llm_code(prompt: str, lang_hint: str) -> str:
    sys_prompt = (
        "Tu es un assistant de codage. Réponds par du code complet et auto-suffisant, "
        "dans un unique bloc ```lang si possible. Ajoute des commentaires (clairs, utiles). "
        "Respecte strictement les conventions de KB/MEM si présentes."
    )
    full = _prompt_with_kb(sys_prompt, prompt + f"\nLangage cible: {lang_hint}\n")
    out = _llm(full, max_tokens=2048, temperature=0.2, top_p=0.9)["choices"][0]["text"]
    m = CODE_BLOCK_RE.search(out)
    if m: return m.group(2).strip()
    return out.strip()

def _llm_explain(prompt: str) -> str:
    sys_prompt = (
        "Tu es pédagogue. Donne une explication claire, structurée, concise, en français. "
        "Utilise des listes à puces quand pertinent. Appuie-toi sur KB/MEM si utile."
    )
    full = _prompt_with_kb(sys_prompt, prompt)
    out = _llm(full, max_tokens=1024, temperature=0.2, top_p=0.9)["choices"][0]["text"]
    return out.strip()

# ===== BUILD (retourne le texte final pour le chat, sections + fichiers titrés)
async def run_pipeline_build(goal: str, forced_lang: Optional[str], do_tests: bool, cancel_event: asyncio.Event) -> str:
    if cancel_event.is_set(): return "⏹️ Interrompu."
    lang_key = (forced_lang or "").lower().strip() or detect_lang(goal)
    mem_set(lang_default=lang_key)
    adapter_key = _choose_lua_adapter(goal) if lang_key == "lua" else lang_key
    if adapter_key not in ADAPTERS: adapter_key = "python"
    adapter = ADAPTERS[adapter_key]
    proj = WORKDIR / adapter.root; proj.mkdir(parents=True, exist_ok=True)

    # heuristique framework préférée
    for fw in ["esx","qbcore","vorp","rsg"]:
        if fw in goal.lower(): mem_set(framework_default=fw); break

    print("[fd] BUILD start:", goal, "| adapter:", adapter_key)

    # Scaffold
    for rel, content in adapter.files.items():
        if cancel_event.is_set(): return "⏹️ Interrompu."
        write_file(proj / rel, content)

    # Post cmds (silence chat)
    for c in adapter.post_cmds:
        if cancel_event.is_set(): return "⏹️ Interrompu."
        print("[fd] post-cmd:", c)
        try:
            subprocess.run(c, cwd=proj, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        except Exception as e:
            print("[fd] post-cmd error:", e)

    # ==== CONTRAT + Génération par fichier
    is_redm = adapter_key == "lua_redm"
    fw = mem_get("framework_default","")
    for k in ["esx","qbcore","vorp","rsg"]:
        if k in goal.lower(): fw = k; mem_set(framework_default=fw); break

    # 1) Explication pédagogique
    header = _llm_explain(
        f"""Explique l'architecture générée pour: {goal}
        - Contexte: framework={fw or 'auto'}, redm={is_redm}
        - Rappelle les règles de sécurité (serveur d'abord, validation inventaire/argent, events fd:*)
        - Reste concis et utile."""
    )

    contract = OUTPUT_CONTRACT.format(fw=(fw or 'auto'), lang='lua' if adapter_key.startswith('lua_') else adapter.name)

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
            code_cli = _llm_code(cli_prompt, "lua")

        # fxmanifest + config
        fxm_prompt = textwrap.dedent(f"""
            {contract}

            Produis un `fxmanifest.lua` correct pour {'RedM' if is_redm else 'FiveM'} (scripts client/server/config).
        """).strip()
        code_fxm = _llm_code(fxm_prompt, "lua")

        cfg_prompt = textwrap.dedent(f"""
            {contract}

            Produis `config.lua` pour centraliser constantes/paramètres (ex: prix, noms d'item, etc.) pour:
            {goal}
        """).strip()
        code_cfg = _llm_code(cfg_prompt, "lua")

        bundle = {
            "fxmanifest.lua": code_fxm,
            "config.lua": code_cfg,
            "server/main.lua": code_srv
        }
        if need_client:
            bundle["client/main.lua"] = code_cli

        # Validation + auto-fix
        bundle, warns = _validate_and_autofix(bundle, fw, is_redm)

        # Écriture + réponse
        blocks = []
        for fname, code in bundle.items():
            write_file(proj / fname, code)
            lang = "lua" if fname.endswith(".lua") else "lua"
            blocks.append(f"```{lang} title={fname}\n{code}\n```")

        parts = [
            "### Résumé\n" + header,
            "### Sécurité & Framework\n"
            f"- Framework détecté: {fw or 'auto'}\n"
            "- Serveur prioritaire pour l'argent/inventaire\n- Événements préfixés `fd:`\n- Pas d'injection client\n",
            "### API exposée\n"
            "- Voir les events et callbacks dans les fichiers générés (titres ci-dessous)\n",
            "### Fichiers\n" + "\n\n".join(blocks)
        ]
        if warns:
            parts.append("### Warnings du validateur\n- " + "\n- ".join(warns))
        final_text = "\n\n".join(parts)
        print("[fd] BUILD done:", proj)
        return final_text

    # ==== Non-Lua (générique)
    exp = _llm_explain(f"Explique en 4-6 points la solution pour: {goal}")
    instr = textwrap.dedent(f"""
        {OUTPUT_CONTRACT.format(fw='n/a', lang=adapter.name)}

        Donne le contenu EXACT de `{adapter.main_path}`.
    """).strip()
    code_main = _llm_code(instr, adapter.name)
    write_file(proj / adapter.main_path, code_main)
    block = f"```{adapter.name} title={adapter.main_path}\n{code_main}\n```"
    final_text = "### Résumé\n" + exp + "\n\n### Fichiers\n" + block
    print("[fd] BUILD done:", proj)
    return final_text

# ===== FIX (retourne explications + code corrigé commenté + auto-fix)
def _extract_code_blocks(text: str) -> List[tuple[str,str]]:
    blks = []
    for m in CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").lower().strip()
        code = m.group(2).strip()
        blks.append((lang, code))
    if blks:
        return blks
    guess = mem_get("lang_default","python")
    if "RegisterNetEvent" in text or "TriggerServerEvent" in text: guess = "lua"
    return [(guess, text.strip())]

def _framework_from_code(code: str) -> str:
    c = code.lower()
    if "exports['es_extended']" in c or "esx." in c: return "esx"
    if "exports['qb-core']" in c or "qbcore." in c: return "qbcore"
    if "exports.vorp_core" in c or "vorp" in c: return "vorp"
    if "exports['rsg-core']" in c or "rsgcore" in c: return "rsg"
    return mem_get("framework_default", framework_hint_from_kb())

async def run_fix(payload: str, cancel_event: asyncio.Event) -> str:
    if cancel_event.is_set(): return "⏹️ Interrompu."
    blocks = _extract_code_blocks(payload)
    explanations: List[str] = []
    outputs: List[str] = []

    print("[fd] FIX start | blocks:", len(blocks))
    ts = int(time.time())

    for i, (lang, code) in enumerate(blocks, start=1):
        if cancel_event.is_set(): return "⏹️ Interrompu."
        fw = _framework_from_code(code) if lang == "lua" else ""
        if fw: mem_set(framework_default=fw)

        # Analyse guidée
        explain_prompt = textwrap.dedent(f"""
            Analyse ce code {lang}{' pour '+fw.upper() if fw else ''} à la lumière de KB/MEM.
            - Liste précise des risques/antipatterns (sécurité serveur, inventaire/argent, events prefix '{EVENT_PREFIX}', API framework)
            - Ce qui enfreint les conventions du framework
            - Plan de correction en 3-6 points, concis
        """) + f"\nCode:\n```{lang}\n{code}\n```"
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
        fixed = _llm_code(fix_prompt, lang or "python")

        # Auto-fix technique (préfixes, etc.)
        bundle = {"server/main.lua": fixed} if (lang=="lua") else {"main"+EXT_BY_LANG.get(lang or 'text', '.txt'): fixed}
        bundle, warns = _validate_and_autofix(bundle, fw, fw in {"vorp","rsg"})
        fixed = bundle.get("server/main.lua", fixed) if (lang=="lua") else list(bundle.values())[0]

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
    final_text = "\n\n".join(final) if final else "Rien à corriger."
    print("[fd] FIX done")
    return final_text
