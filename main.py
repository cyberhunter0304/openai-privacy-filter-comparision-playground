"""
PII Detection Comparison Tool v2
- Persistent JSON result store (pii_results.json)
- LLM-as-Judge (Azure OpenAI GPT-4o) for accuracy scoring
- Per-tool threshold controls passed from frontend
- Precision / Recall / F1 computation vs optional ground truth
"""

import os, time, asyncio, httpx, json, uuid, hashlib, hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ── Env ───────────────────────────────────────────────────────────────────────
AZURE_CONTENT_SAFETY_KEY      = os.getenv("AZURE_CONTENT_SAFETY_KEY")
AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
AZURE_OPENAI_KEY              = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT         = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT       = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
APP_PASSPHRASE                = os.getenv("APP_PASSPHRASE", "")  # empty = auth disabled (local dev)

# ── Auth helpers ──────────────────────────────────────────────────────────────
def _expected_token() -> str:
    """Derive a stable token from the passphrase using HMAC-SHA256."""
    return hmac.new(APP_PASSPHRASE.encode(), b"pii-shield-token", hashlib.sha256).hexdigest()

def _require_auth(x_app_token: Optional[str] = Header(default=None)):
    """FastAPI dependency — raises 401 if passphrase is set and token is wrong."""
    if not APP_PASSPHRASE:
        return  # auth disabled in local dev
    if x_app_token != _expected_token():
        raise HTTPException(status_code=401, detail="Invalid or missing token")

# ── Persistent store ──────────────────────────────────────────────────────────
RESULTS_FILE = Path("pii_results.json")

def _now():
    return datetime.now(timezone.utc).isoformat()

def load_store() -> dict:
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    return {"runs": [], "meta": {"created": _now(), "total_runs": 0}}

def save_store(store: dict):
    RESULTS_FILE.write_text(json.dumps(store, indent=2, default=str))

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_privacy_filter = None
_presidio_analyzer = None
_presidio_anonymizer = None

def get_privacy_filter():
    global _privacy_filter
    if _privacy_filter is None:
        from transformers import pipeline
        print("[OPF] Loading model…")
        _privacy_filter = pipeline(
            task="token-classification",
            model="openai/privacy-filter",
            aggregation_strategy="simple",
        )
        print("[OPF] Ready.")
    return _privacy_filter

def get_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        print("[Presidio] Loading…")
        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()
        print("[Presidio] Ready.")
    return _presidio_analyzer, _presidio_anonymizer

# ── Pydantic models ───────────────────────────────────────────────────────────
class Thresholds(BaseModel):
    openai_privacy_filter: float = 0.5
    azure_content_safety: float  = 0.5
    presidio: float               = 0.5

class PIIRequest(BaseModel):
    text: str
    tools: list[str] = ["openai_privacy_filter", "azure_content_safety", "presidio"]
    thresholds: Thresholds = Thresholds()
    ground_truth: Optional[list[dict]] = None   # [{text, entity_type}]
    run_label: Optional[str] = None
    enable_judge: bool = True

class PIIEntity(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int
    score: float

class JudgeScore(BaseModel):
    verdict: str            # "GOOD" | "PARTIAL" | "POOR"
    precision_est: float
    recall_est: float
    f1_est: float
    missed: list[str]
    over_redacted: list[str]
    notes: str

class ToolResult(BaseModel):
    tool: str
    status: str
    latency_ms: float
    entities: list[PIIEntity]
    redacted_text: str
    threshold_used: float
    entities_below_threshold: int
    judge: Optional[JudgeScore] = None
    judge_error: Optional[str] = None
    error: Optional[str] = None

# ── Metrics vs ground truth ───────────────────────────────────────────────────
def compute_metrics(detected: list[PIIEntity], ground_truth: list[dict]) -> dict:
    if not ground_truth:
        return {}
    gt  = {g["text"].lower().strip() for g in ground_truth}
    det = {e.text.lower().strip() for e in detected}
    tp  = len(gt & det)
    fp  = len(det - gt)
    fn  = len(gt - det)
    p   = tp / (tp + fp) if (tp + fp) else 0.0
    r   = tp / (tp + fn) if (tp + fn) else 0.0
    f1  = 2*p*r / (p+r) if (p+r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

# ── LLM Judge ─────────────────────────────────────────────────────────────────
async def llm_judge(
    original_text: str,
    tool_name: str,
    entities: list[PIIEntity],
    redacted_text: str,
) -> tuple[Optional[JudgeScore], Optional[str]]:
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        return None, "Azure OpenAI credentials not configured"
    if not AZURE_OPENAI_ENDPOINT.startswith(("http://", "https://")):
        return None, f"AZURE_OPENAI_ENDPOINT is not a valid URL: '{AZURE_OPENAI_ENDPOINT}'"
    if "transcribe" in AZURE_OPENAI_DEPLOYMENT.lower():
        return None, (
            f"AZURE_OPENAI_DEPLOYMENT '{AZURE_OPENAI_DEPLOYMENT}' is a transcription model; "
            "use a chat-capable deployment (e.g., gpt-4o or gpt-4.1)."
        )

    entity_list = [{"text": e.text, "type": e.entity_type, "score": e.score} for e in entities]

    system = (
        "You are an expert PII detection evaluator. "
        "There is no single golden-rule label set here: your job is comparative evaluation.\n"
        "Assess how this tool behaves on the given text, including what it catches, misses, "
        "over-redacts, and how broad vs strict its detection style is.\n"
        "Focus on practical utility for privacy redaction, not perfect taxonomy naming.\n"
        "Return ONLY valid JSON, no markdown, no explanation outside the JSON."
    )
    user = f"""Evaluate the PII detection result below.

ORIGINAL TEXT:
{original_text}

TOOL: {tool_name}
TOOL-DETECTED ENTITIES: {json.dumps(entity_list)}
TOOL-REDACTED OUTPUT: {redacted_text}

Instructions:
- Infer likely sensitive spans in ORIGINAL TEXT using best judgment (no strict gold labels).
- Compare with TOOL-DETECTED ENTITIES and TOOL-REDACTED OUTPUT.
- Identify misses and over-redactions.
- Estimate precision, recall, and F1 as comparative indicators (approximate, not absolute truth).
- In notes, explain what this tool is good/bad at and its detection style (strict, broad, fragmented, etc.).

Return JSON with exactly these fields:
{{
  "verdict": "GOOD" | "PARTIAL" | "POOR",
  "precision_est": <float 0-1>,
  "recall_est": <float 0-1>,
  "f1_est": <float 0-1>,
  "missed": [<PII strings the tool missed>],
  "over_redacted": [<non-PII strings wrongly flagged>],
  "notes": "<one concise sentence summarising performance>"
}}"""

    url     = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"}
    base_body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            candidate_bodies = [
                {**base_body, "max_completion_tokens": 512},
                {**base_body, "max_tokens": 512},
            ]
            last_param_error = None

            for req_body in candidate_bodies:
                try:
                    r = await client.post(url, headers=headers, json=req_body)
                    r.raise_for_status()
                    d = json.loads(r.json()["choices"][0]["message"]["content"])
                    return JudgeScore(
                        verdict=d.get("verdict","PARTIAL"),
                        precision_est=float(d.get("precision_est",0)),
                        recall_est=float(d.get("recall_est",0)),
                        f1_est=float(d.get("f1_est",0)),
                        missed=d.get("missed",[]),
                        over_redacted=d.get("over_redacted",[]),
                        notes=d.get("notes",""),
                    ), None
                except httpx.HTTPStatusError as ex:
                    resp_text = (ex.response.text or "").strip()
                    msg = f"HTTP {ex.response.status_code}: {resp_text[:500]}" if resp_text else str(ex)
                    # Some Azure model deployments accept only one token limit parameter.
                    if ex.response.status_code == 400 and "unsupported_parameter" in resp_text:
                        last_param_error = msg
                        continue
                    print(f"[Judge] {msg}")
                    return None, msg

            if last_param_error:
                print(f"[Judge] {last_param_error}")
                return None, last_param_error
    except httpx.HTTPStatusError as ex:
        resp_text = (ex.response.text or "").strip()
        msg = f"HTTP {ex.response.status_code}: {resp_text[:500]}" if resp_text else str(ex)
        print(f"[Judge] {msg}")
        return None, msg
    except Exception as ex:
        msg = str(ex)
        print(f"[Judge] {msg}")
        return None, msg

# ── Detection runners ─────────────────────────────────────────────────────────
def run_openai_privacy_filter(text: str, threshold: float) -> ToolResult:
    try:
        clf = get_privacy_filter()          # no-op if already loaded
        t0 = time.perf_counter()            # start: text enters model
        spans = clf(text)
        latency_ms = (time.perf_counter() - t0) * 1000
        kept, below = [], 0
        for s in spans:
            if s["score"] >= threshold:
                kept.append(s)
            else:
                below += 1
        entities = [
            PIIEntity(text=text[s["start"]:s["end"]], entity_type=s["entity_group"],
                      start=s["start"], end=s["end"], score=round(s["score"], 4))
            for s in kept
        ]
        redacted = text
        for s in sorted(kept, key=lambda x: x["start"], reverse=True):
            redacted = redacted[:s["start"]] + f"[{s['entity_group'].upper()}]" + redacted[s["end"]:]
        return ToolResult(tool="OpenAI Privacy Filter", status="ok",
                          latency_ms=round(latency_ms,2), entities=entities,
                          redacted_text=redacted, threshold_used=threshold,
                          entities_below_threshold=below)
    except Exception as e:
        return ToolResult(tool="OpenAI Privacy Filter", status="error",
                          latency_ms=0,
                          entities=[], redacted_text=text, threshold_used=threshold,
                          entities_below_threshold=0, error=str(e))


def run_presidio(text: str, threshold: float) -> ToolResult:
    try:
        analyzer, anonymizer = get_presidio()   # no-op if already loaded
        t0 = time.perf_counter()                # start: text enters analyzer
        results_all   = analyzer.analyze(text=text, language="en", score_threshold=0.0)
        results_kept  = [r for r in results_all if r.score >= threshold]
        below         = len(results_all) - len(results_kept)
        entities = [
            PIIEntity(text=text[r.start:r.end], entity_type=r.entity_type,
                      start=r.start, end=r.end, score=round(r.score,4))
            for r in results_kept
        ]
        redacted = text
        for r in sorted(results_kept, key=lambda x: x.start, reverse=True):
            redacted = redacted[:r.start] + f"[{r.entity_type.upper()}]" + redacted[r.end:]
        latency_ms = (time.perf_counter() - t0) * 1000  # end: redacted text ready
        return ToolResult(tool="Microsoft Presidio", status="ok",
                          latency_ms=round(latency_ms,2), entities=entities,
                          redacted_text=redacted, threshold_used=threshold,
                          entities_below_threshold=below)
    except Exception as e:
        return ToolResult(tool="Microsoft Presidio", status="error",
                          latency_ms=0,
                          entities=[], redacted_text=text, threshold_used=threshold,
                          entities_below_threshold=0, error=str(e))


async def run_azure(text: str, threshold: float) -> ToolResult:
    if not AZURE_CONTENT_SAFETY_KEY or not AZURE_CONTENT_SAFETY_ENDPOINT:
        return ToolResult(tool="Azure Content Safety", status="disabled",
                          latency_ms=0, entities=[], redacted_text=text,
                          threshold_used=threshold, entities_below_threshold=0,
                          error="Azure credentials not set in .env")
    url     = f"{AZURE_CONTENT_SAFETY_ENDPOINT.rstrip('/')}/text/analytics/v3.1/entities/recognition/pii"
    headers = {"Ocp-Apim-Subscription-Key": AZURE_CONTENT_SAFETY_KEY, "Content-Type": "application/json"}
    body    = {"documents": [{"id":"1","language":"en","text":text}]}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            t0 = time.perf_counter()            # start: request sent to Azure
            resp = await client.post(url, headers=headers, json=body)
            latency_ms = (time.perf_counter()-t0)*1000  # end: response received
            if resp.status_code != 200:
                return ToolResult(tool="Azure Content Safety", status="error",
                                  latency_ms=round(latency_ms,2), entities=[], redacted_text=text,
                                  threshold_used=threshold, entities_below_threshold=0,
                                  error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            doc  = resp.json().get("documents",[{}])[0]
            raw  = doc.get("entities",[])
            all_e = [
                PIIEntity(text=e.get("text",""), entity_type=e.get("category","UNKNOWN"),
                          start=e.get("offset",0), end=e.get("offset",0)+e.get("length",0),
                          score=round(e.get("confidenceScore",0),4))
                for e in raw
            ]
            kept  = [e for e in all_e if e.score >= threshold]
            below = len(all_e) - len(kept)
            redacted = text
            for e in sorted(kept, key=lambda x: x.start, reverse=True):
                redacted = redacted[:e.start] + f"[{e.entity_type.upper()}]" + redacted[e.end:]
            return ToolResult(tool="Azure Content Safety", status="ok",
                              latency_ms=round(latency_ms,2), entities=kept,
                              redacted_text=redacted, threshold_used=threshold,
                              entities_below_threshold=below)
    except Exception as e:
        return ToolResult(tool="Azure Content Safety", status="error",
                          latency_ms=0,
                          entities=[], redacted_text=text, threshold_used=threshold,
                          entities_below_threshold=0, error=str(e))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PII Comparison v2")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(Path(__file__).resolve().parent.joinpath("index.html").read_text(encoding="utf-8"))

@app.post("/api/auth")
async def auth(body: dict):
    """Validate passphrase and return session token."""
    if not APP_PASSPHRASE:
        return {"ok": True, "token": ""}  # auth disabled
    if body.get("passphrase") == APP_PASSPHRASE:
        return {"ok": True, "token": _expected_token()}
    raise HTTPException(status_code=401, detail="Wrong passphrase")

@app.get("/api/health")
async def health():
    store = load_store()
    return {
        "status": "ok",
        "azure_openai_configured": bool(AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT),
        "azure_pii_configured":    bool(AZURE_CONTENT_SAFETY_KEY and AZURE_CONTENT_SAFETY_ENDPOINT),
        "total_runs_stored":       store.get("meta",{}).get("total_runs", 0),
    }

@app.get("/api/warmup", dependencies=[Depends(_require_auth)])
async def warmup():
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, get_privacy_filter),
        loop.run_in_executor(None, get_presidio),
    )
    return {"status": "warmed up"}

@app.post("/api/detect", dependencies=[Depends(_require_auth)])
async def detect_pii(req: PIIRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Empty text"})

    tool_names = [t.lower() for t in req.tools]
    thr  = req.thresholds
    loop = asyncio.get_event_loop()

    # All tools run concurrently
    coros = []
    if "openai_privacy_filter" in tool_names:
        coros.append(loop.run_in_executor(None, run_openai_privacy_filter, text, thr.openai_privacy_filter))
    if "presidio" in tool_names:
        coros.append(loop.run_in_executor(None, run_presidio, text, thr.presidio))
    if "azure_content_safety" in tool_names:
        coros.append(run_azure(text, thr.azure_content_safety))

    raw_results = await asyncio.gather(*coros, return_exceptions=True)
    tool_results: list[ToolResult] = [
        r if isinstance(r, ToolResult) else ToolResult(
            tool="Unknown", status="error", latency_ms=0,
            entities=[], redacted_text=text,
            threshold_used=0, entities_below_threshold=0, error=str(r)
        )
        for r in raw_results
    ]

    # LLM Judge — parallel across tools
    ok_results = [r for r in tool_results if r.status == "ok"]
    if req.enable_judge and ok_results:
        if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
            for r in ok_results:
                r.judge = None
                r.judge_error = "Azure OpenAI credentials not configured"
        elif not AZURE_OPENAI_ENDPOINT.startswith(("http://", "https://")):
            for r in ok_results:
                r.judge = None
                r.judge_error = f"AZURE_OPENAI_ENDPOINT is not a valid URL: '{AZURE_OPENAI_ENDPOINT}'"
        else:
            judge_scores = await asyncio.gather(*[
                llm_judge(text, r.tool, r.entities, r.redacted_text)
                for r in ok_results
            ], return_exceptions=True)
            for r, j in zip(ok_results, judge_scores):
                if isinstance(j, Exception):
                    r.judge = None
                    r.judge_error = str(j)
                else:
                    score, err = j
                    r.judge = score
                    r.judge_error = err or ("Judge returned empty response" if score is None else None)

    # Ground-truth metrics
    metrics_by_tool = {}
    if req.ground_truth:
        for r in tool_results:
            metrics_by_tool[r.tool] = compute_metrics(r.entities, req.ground_truth)

    # Persist
    store  = load_store()
    run_id = str(uuid.uuid4())[:8]
    ok_tools = [r for r in tool_results if r.status == "ok"]
    store["runs"].append({
        "run_id":       run_id,
        "timestamp":    _now(),
        "label":        req.run_label or f"Run #{store['meta']['total_runs']+1}",
        "input_text":   text,
        "tools_used":   tool_names,
        "thresholds":   thr.model_dump(),
        "ground_truth": req.ground_truth,
        "results": [
            {**r.model_dump(), "metrics_vs_gt": metrics_by_tool.get(r.tool, {})}
            for r in tool_results
        ],
        "summary": {
            "fastest_tool": min(ok_tools, key=lambda r: r.latency_ms).tool if ok_tools else None,
            "most_entities": max(ok_tools, key=lambda r: len(r.entities)).tool if ok_tools else None,
        }
    })
    store["meta"]["total_runs"] += 1
    store["meta"]["last_run"]    = _now()
    save_store(store)

    return {
        "run_id":        run_id,
        "results":       [r.model_dump() for r in tool_results],
        "metrics":       metrics_by_tool,
        "original_text": text,
        "stored":        True,
    }

# ── Store endpoints ───────────────────────────────────────────────────────────
_auth = [Depends(_require_auth)]

@app.get("/api/store", dependencies=_auth)
async def get_store():
    return load_store()

@app.get("/api/store/download", dependencies=_auth)
async def download_store():
    if not RESULTS_FILE.exists():
        return JSONResponse(status_code=404, content={"error": "No results yet"})
    return FileResponse(str(RESULTS_FILE), filename="pii_results.json", media_type="application/json")

@app.delete("/api/store", dependencies=_auth)
async def clear_store():
    save_store({"runs": [], "meta": {"created": _now(), "total_runs": 0}})
    return {"status": "cleared"}

@app.get("/api/store/stats", dependencies=_auth)
async def store_stats():
    runs = load_store().get("runs", [])
    if not runs:
        return {"total_runs": 0, "per_tool": {}}
    tool_stats: dict[str, dict] = {}
    for run in runs:
        for r in run.get("results", []):
            t = r["tool"]
            if t not in tool_stats:
                tool_stats[t] = {"latencies": [], "f1s": [], "verdicts": []}
            if r["status"] == "ok":
                tool_stats[t]["latencies"].append(r["latency_ms"])
            j = r.get("judge")
            if j:
                tool_stats[t]["f1s"].append(j.get("f1_est", 0))
                tool_stats[t]["verdicts"].append(j.get("verdict", ""))
    summary = {}
    for t, s in tool_stats.items():
        lats, f1s = s["latencies"], s["f1s"]
        summary[t] = {
            "runs":            len(lats),
            "avg_latency_ms":  round(sum(lats)/len(lats), 1) if lats else None,
            "min_latency_ms":  round(min(lats), 1) if lats else None,
            "max_latency_ms":  round(max(lats), 1) if lats else None,
            "avg_f1_judge":    round(sum(f1s)/len(f1s), 3) if f1s else None,
            "verdict_counts":  {v: s["verdicts"].count(v) for v in set(s["verdicts"])},
        }
    return {"total_runs": len(runs), "per_tool": summary}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)