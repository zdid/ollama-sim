"""
ollama-sim : Faux serveur Ollama → route vers Mistral API
Implémente fidèlement le protocole Ollama v0.5.x
+ Injection des règles domotiques (regles_mistral.txt)
+ Routage MQTT vers programme TS pour planifications/macros/gestion
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import httpx
import paho.mqtt.client as mqtt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ─── Config ───────────────────────────────────────────────────────────────────

MISTRAL_API_KEY  = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

MQTT_HOST        = os.environ.get("MQTT_HOST",    "192.168.1.51")
MQTT_PORT        = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER        = os.environ.get("MQTT_USER",    "")
MQTT_PASS        = os.environ.get("MQTT_PASS",    "")

TOPIC_COMMANDE   = "domotique/mistral/commande"    # proxy → TS
TOPIC_REPONSE    = "domotique/mistral/reponse"     # TS → proxy
TOPIC_EXECUTION  = "domotique/mistral/execution"   # TS → proxy (déclenchement planifié)

RULES_FILE       = os.environ.get("RULES_FILE", "/app/rules/regles_mistral.txt")
LOG_DIR          = "/app/logs"
MQTT_TIMEOUT_SEC = 15   # délai max d'attente réponse TS

# ─── Types JSON structurés à router vers MQTT ─────────────────────────────────

STRUCTURED_TYPES = {"planification", "macro", "macro_ref", "condition",
                    "sequence", "gestion", "execution"}

# ─── Modèles Mistral ──────────────────────────────────────────────────────────

MODEL_MAP: dict[str, str] = {
    "mistral-medium-3.5":      "mistral-medium-3.5-2604",
    "mistral-small-4":         "mistral-small-2503",
    "mistral-large-3":         "mistral-large-2512",
    "mistral-small":           "mistral-small-latest",
    "mistral-small:latest":    "mistral-small-latest",
    "mistral-large":           "mistral-large-latest",
    "mistral-large:latest":    "mistral-large-latest",
    "ministral-14b":           "ministral-3b-2412",
    "ministral-8b":            "ministral-8b-2410",
    "ministral-3b":            "ministral-3b-2410",
    "codestral":               "codestral-latest",
    "codestral:latest":        "codestral-latest",
    "devstral-2":              "devstral-2512",
    "magistral-medium":        "magistral-medium-2507",
    "magistral-small":         "magistral-small-2507",
    "mistral-nemo":            "open-mistral-nemo",
    "mistral-nemo:latest":     "open-mistral-nemo",
    "mistral":                 "mistral-small-latest",
    "mistral:latest":          "mistral-small-latest",
    "mistral:7b":              "mistral-small-latest",
    "mistral:7b-instruct":     "mistral-small-latest",
}
DEFAULT_MODEL = "mistral-small-latest"

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/ollama-sim.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ollama-sim")

def log_block(label: str, data: dict | str):
    sep  = "─" * 64
    body = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, dict) else data
    log.info(f"\n{sep}\n{label}\n{body}\n{sep}")

def _model_log_path(mistral_model: str) -> str:
    safe = mistral_model.replace("/", "-").replace(":", "-")
    return f"{LOG_DIR}/conv_{safe}.log"

def log_conversation(ollama_model: str, mistral_model: str,
                     question: str, text_response: str, tool_calls: list[dict]):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 64
    lines = [
        f"\n{sep}",
        f"[{ts}]  modèle HA: {ollama_model}  →  Mistral: {mistral_model}",
        f"❓ QUESTION : {question}",
    ]
    if text_response:
        lines.append(f"💬 RÉPONSE  : {text_response}")
    if tool_calls:
        lines.append("🔧 TOOL CALLS JSON :")
        for tc in tool_calls:
            lines.append(json.dumps(tc, indent=2, ensure_ascii=False))
    if not text_response and not tool_calls:
        lines.append("⚠️  Aucune réponse capturée")
    lines.append(sep)
    with open(_model_log_path(mistral_model), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ─── Règles domotiques (watchdog) ─────────────────────────────────────────────

class RulesLoader:
    """Charge et surveille le fichier de règles. Rechargement automatique à la sauvegarde."""

    def __init__(self, path: str):
        self.path    = Path(path)
        self._rules  = ""
        self._lock   = threading.Lock()
        self._load()
        self._start_watcher()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                content = f.read().strip()
            with self._lock:
                self._rules = content
            log.info(f"[rules] Règles chargées depuis {self.path} ({len(content)} caractères)")
        else:
            log.warning(f"[rules] Fichier non trouvé : {self.path} — règles vides")

    def _start_watcher(self):
        loader = self

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(event.src_path).resolve() == loader.path.resolve():
                    log.info(f"[rules] Modification détectée → rechargement")
                    loader._load()

        observer = Observer()
        observer.schedule(Handler(), str(self.path.parent), recursive=False)
        observer.daemon = True
        observer.start()
        log.info(f"[rules] Surveillance active sur {self.path.parent}")

    def get(self) -> str:
        with self._lock:
            return self._rules

rules_loader = RulesLoader(RULES_FILE)

# ─── MQTT ─────────────────────────────────────────────────────────────────────

# Dictionnaire des futures en attente : correlation_id → asyncio.Future
_pending: dict[str, asyncio.Future] = {}
_pending_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None

def _on_mqtt_message(client, userdata, message):
    """Appelé dans le thread MQTT — résout la Future correspondante."""
    try:
        payload = json.loads(message.payload.decode())
        corr_id = payload.get("correlation_id")
        if not corr_id:
            return
        with _pending_lock:
            future = _pending.get(corr_id)
        if future and _loop:
            _loop.call_soon_threadsafe(future.set_result, payload)
    except Exception as e:
        log.error(f"[mqtt] Erreur réception : {e}")

def _on_execution_message(client, userdata, message):
    """Le TS déclenche une exécution planifiée — à traiter séparément."""
    try:
        payload = json.loads(message.payload.decode())
        log.info(f"[mqtt] Déclenchement exécution reçu : {json.dumps(payload)[:200]}")
        # TODO : soumettre à Mistral pour déploiement, puis exécuter via HA
    except Exception as e:
        log.error(f"[mqtt] Erreur exécution : {e}")

mqtt_client = mqtt.Client(client_id="ollama-sim", protocol=mqtt.MQTTv5)
if MQTT_USER:
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)

mqtt_client.message_callback_add(TOPIC_REPONSE,   _on_mqtt_message)
mqtt_client.message_callback_add(TOPIC_EXECUTION, _on_execution_message)

def start_mqtt():
    try:
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        mqtt_client.subscribe([(TOPIC_REPONSE, 1), (TOPIC_EXECUTION, 1)])
        mqtt_client.loop_start()
        log.info(f"[mqtt] Connecté à {MQTT_HOST}:{MQTT_PORT}")
    except Exception as e:
        log.error(f"[mqtt] Connexion impossible : {e}")

async def mqtt_send_and_wait(payload: dict) -> dict:
    """
    Publie sur TOPIC_COMMANDE et attend la réponse sur TOPIC_REPONSE.
    Utilise un correlation_id pour matcher la bonne réponse.
    Timeout : MQTT_TIMEOUT_SEC secondes.
    """
    corr_id = str(uuid.uuid4())
    payload["correlation_id"] = corr_id

    future = asyncio.get_event_loop().create_future()
    with _pending_lock:
        _pending[corr_id] = future

    mqtt_client.publish(TOPIC_COMMANDE, json.dumps(payload, ensure_ascii=False), qos=1)
    log.info(f"[mqtt] Publié sur {TOPIC_COMMANDE} (corr={corr_id[:8]}...)")

    try:
        result = await asyncio.wait_for(future, timeout=MQTT_TIMEOUT_SEC)
        log.info(f"[mqtt] Réponse reçue (corr={corr_id[:8]}...)")
        return result
    except asyncio.TimeoutError:
        log.error(f"[mqtt] Timeout après {MQTT_TIMEOUT_SEC}s (corr={corr_id[:8]}...)")
        return {"error": "timeout", "message": "Le programme de planification ne répond pas"}
    finally:
        with _pending_lock:
            _pending.pop(corr_id, None)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ollama-sim", version="0.2.0")

@app.on_event("startup")
async def startup():
    global _loop
    _loop = asyncio.get_event_loop()
    start_mqtt()

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    body = await request.body()
    log.info(f">>> {request.method} {request.url.path} — body: {body.decode()[:200]}")
    response = await call_next(request)
    log.info(f"<<< {response.status_code}")
    return response

# ─── Helpers ──────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def resolve_model(ollama_model: str) -> str:
    return MODEL_MAP.get(ollama_model.lower().strip(), DEFAULT_MODEL)

def ollama_options_to_mistral(options: dict) -> dict:
    out = {}
    if "temperature" in options: out["temperature"] = options["temperature"]
    if "top_p"       in options: out["top_p"]       = options["top_p"]
    if "num_predict" in options: out["max_tokens"]  = options["num_predict"]
    if "seed"        in options: out["random_seed"] = options["seed"]
    return out

def build_messages(body: dict) -> list[dict]:
    messages = list(body.get("messages", []))
    if not messages:
        system = body.get("system", "")
        prompt = body.get("prompt", "")
        if system:
            messages.append({"role": "system", "content": system})
        if prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        system = body.get("system", "")
        if system and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system}] + messages
    return messages

def inject_rules(messages: list[dict]) -> list[dict]:
    """
    Injecte les règles domotiques à la fin du system prompt existant.
    Si pas de system prompt, en crée un.
    """
    rules = rules_loader.get()
    if not rules:
        return messages

    messages = list(messages)
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            messages[i] = {
                **m,
                "content": m["content"] + "\n\n" + rules
            }
            return messages

    # Pas de system prompt → en créer un
    messages.insert(0, {"role": "system", "content": rules})
    return messages

def extract_question(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""

def make_ollama_done_chunk(model: str, prompt_tokens: int = 0,
                           completion_tokens: int = 0) -> dict:
    return {
        "model":               model,
        "created_at":          now_iso(),
        "message":             {"role": "assistant", "content": ""},
        "done":                True,
        "done_reason":         "stop",
        "total_duration":      0,
        "load_duration":       0,
        "prompt_eval_count":   prompt_tokens,
        "prompt_eval_duration":0,
        "eval_count":          completion_tokens,
        "eval_duration":       0,
    }

def extract_json_from_text(text: str) -> dict | None:
    """
    Tente d'extraire un JSON structuré de la réponse texte de Mistral.
    Retourne le dict si "type" est un type connu, sinon None.
    """
    text = text.strip()
    # Nettoyer les balises markdown
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("type") in STRUCTURED_TYPES:
            return data
    except Exception:
        pass
    return None

def make_confirmation_chunk(model: str, text: str) -> str:
    """Crée un chunk NDJSON Ollama avec un message texte de confirmation."""
    chunk = {
        "model":      model,
        "created_at": now_iso(),
        "message":    {"role": "assistant", "content": text},
        "done":       False,
    }
    return json.dumps(chunk, ensure_ascii=False) + "\n"

# ─── Streaming Mistral → Ollama NDJSON ────────────────────────────────────────

async def stream_mistral_to_ollama(
    mistral_stream: httpx.Response,
    ollama_model:   str,
) -> AsyncIterator[str]:
    prompt_tokens     = 0
    completion_tokens = 0
    text_parts:       list[str]  = []
    tool_calls:       list[dict] = []
    tool_call_buffer: dict[int, dict] = {}

    async for line in mistral_stream.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choice        = chunk.get("choices", [{}])[0]
        delta         = choice.get("delta", {})
        content       = delta.get("content") or ""
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx  = tc_delta.get("index", 0)
            func = tc_delta.get("function", {})
            if idx not in tool_call_buffer:
                tool_call_buffer[idx] = {
                    "id":       tc_delta.get("id", ""),
                    "type":     tc_delta.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                }
            if func.get("name"):
                tool_call_buffer[idx]["function"]["name"]      += func["name"]
            if func.get("arguments"):
                tool_call_buffer[idx]["function"]["arguments"] += func["arguments"]

        usage = chunk.get("usage") or {}
        if usage:
            prompt_tokens     = usage.get("prompt_tokens",     prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

        if finish_reason in ("stop", "tool_calls"):
            if content:
                text_parts.append(content)
                yield json.dumps({
                    "model":      ollama_model,
                    "created_at": now_iso(),
                    "message":    {"role": "assistant", "content": content},
                    "done":       False,
                }, ensure_ascii=False) + "\n"
            break

        if content:
            text_parts.append(content)
            yield json.dumps({
                "model":      ollama_model,
                "created_at": now_iso(),
                "message":    {"role": "assistant", "content": content},
                "done":       False,
            }, ensure_ascii=False) + "\n"

    for tc in tool_call_buffer.values():
        try:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        except Exception:
            pass
        tool_calls.append(tc)

    done_chunk = make_ollama_done_chunk(ollama_model, prompt_tokens, completion_tokens)
    if tool_calls:
        done_chunk["message"]["tool_calls"] = tool_calls

    yield json.dumps(done_chunk, ensure_ascii=False) + "\n"
    yield json.dumps({"__meta__": True, "text": text_parts, "tool_calls": tool_calls}) + "\n"

# ─── Routes Ollama ────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/api/version")
async def version():
    return JSONResponse({"version": "0.5.1"})


@app.get("/api/tags")
async def list_models():
    models = []
    seen   = set()
    for ollama_name in MODEL_MAP:
        if ollama_name in seen:
            continue
        seen.add(ollama_name)
        models.append({
            "name":        ollama_name,
            "model":       ollama_name,
            "modified_at": "2024-06-01T00:00:00Z",
            "size":        4_113_000_000,
            "digest":      f"sha256:sim-{ollama_name.replace(':', '-')}",
            "details": {
                "parent_model":       "",
                "format":             "gguf",
                "family":             "mistral",
                "families":           ["mistral"],
                "parameter_size":     "7B",
                "quantization_level": "Q4_0",
            },
        })
    log.info(f"[/api/tags] {len(models)} modèles exposés")
    return JSONResponse({"models": models})


@app.post("/api/show")
async def show_model(request: Request):
    body  = await request.json()
    model = body.get("model", "mistral")
    return JSONResponse({
        "modelfile":  f"FROM {model}\nSYSTEM You are a helpful assistant.",
        "parameters": "temperature 0.7",
        "template":   "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}",
        "details": {
            "format": "gguf", "family": "mistral",
            "parameter_size": "7B", "quantization_level": "Q4_0",
        },
    })


@app.post("/api/chat")
async def chat(request: Request):
    """Route principale — injection règles, détection JSON structuré, routage MQTT."""
    body = await request.json()
    log_block("📨 HA → PROXY  [/api/chat]", body)

    if not MISTRAL_API_KEY:
        raise HTTPException(503, "MISTRAL_API_KEY non définie")

    ollama_model  = body.get("model", "mistral")
    mistral_model = resolve_model(ollama_model)
    messages      = build_messages(body)
    messages      = inject_rules(messages)        # ← injection des règles
    options       = body.get("options", {})
    question      = extract_question(messages)

    mistral_payload = {
        "model":    mistral_model,
        "messages": messages,
        "stream":   True,
        **ollama_options_to_mistral(options),
    }
    log_block(f"🚀 PROXY → MISTRAL  [{mistral_model}]", mistral_payload)

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream",
    }

    client = httpx.AsyncClient(timeout=120)

    async def generate():
        text_parts: list[str]  = []
        tool_calls: list[dict] = []

        try:
            async with client.stream(
                "POST",
                f"{MISTRAL_BASE_URL}/chat/completions",
                headers=headers,
                json=mistral_payload,
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    log.error(f"Mistral HTTP {resp.status_code}: {error_body.decode()}")
                    err = make_ollama_done_chunk(ollama_model)
                    err["error"] = f"Mistral API error {resp.status_code}"
                    yield json.dumps(err) + "\n"
                    return

                chunks_to_send = []
                async for chunk in stream_mistral_to_ollama(resp, ollama_model):
                    try:
                        parsed = json.loads(chunk)
                        if parsed.get("__meta__"):
                            text_parts = parsed["text"]
                            tool_calls = parsed["tool_calls"]
                            continue
                    except Exception:
                        pass
                    chunks_to_send.append(chunk)

                # ── Détection JSON structuré ───────────────────────────────
                full_text    = "".join(text_parts)
                structured   = extract_json_from_text(full_text)

                if structured:
                    # ── Routage MQTT → programme TS ───────────────────────
                    log_block("📡 MQTT → TS", structured)
                    mqtt_response = await mqtt_send_and_wait(structured)

                    confirmation = mqtt_response.get(
                        "message",
                        "Commande transmise au gestionnaire de planification."
                    )
                    if mqtt_response.get("error"):
                        confirmation = f"Erreur : {mqtt_response.get('message', 'inconnue')}"

                    log.info(f"[mqtt] Confirmation : {confirmation}")
                    yield make_confirmation_chunk(ollama_model, confirmation)
                    done = make_ollama_done_chunk(ollama_model)
                    yield json.dumps(done, ensure_ascii=False) + "\n"

                else:
                    # ── Réponse texte normale → transmettre à HA ──────────
                    for chunk in chunks_to_send:
                        yield chunk

        except httpx.ConnectError as e:
            log.error(f"Connexion Mistral impossible: {e}")
            yield json.dumps({"error": "Cannot reach Mistral API", "done": True}) + "\n"

        finally:
            if question:
                log_conversation(ollama_model, mistral_model, question,
                                 "".join(text_parts), tool_calls)
            await client.aclose()

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
async def generate_compat(request: Request):
    body = await request.json()
    body.setdefault("messages", [])
    request._body = json.dumps(body).encode()
    return await chat(request)


@app.post("/api/embeddings")
async def embeddings(request: Request):
    body  = await request.json()
    model = body.get("model", "mistral")
    return JSONResponse({"model": model, "embedding": [0.0] * 4096})


@app.get("/api/ps")
async def ps():
    return JSONResponse({"models": []})


@app.delete("/api/delete")
async def delete_model():
    return JSONResponse({"status": "ok"})


@app.post("/api/pull")
async def pull_model(request: Request):
    body  = await request.json()
    model = body.get("model", "mistral")
    log.info(f"[/api/pull] Simulation du pull pour : {model}")
    return JSONResponse({"status": "success"})
