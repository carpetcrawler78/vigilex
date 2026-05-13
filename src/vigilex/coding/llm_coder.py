"""
llm_coder.py -- LLM-based final MedDRA PT selection via Ollama.

Why add an LLM as a third stage?
    The CrossEncoder from Stage 2 gives us an excellent ranking, but:
    - It produces a raw score (logit), not an interpretable probability
    - It cannot explain *why* it ranked something highly
    - It cannot catch edge cases like negation ("the patient did NOT experience
      hypoglycaemia") or ambiguous reports mentioning multiple conditions

    The LLM (Large Language Model) running via Ollama adds:
    1. Clinical reasoning -- it reads the full narrative and the top 5 candidates
       and makes a justified selection
    2. Structured confidence -- a 0.0-1.0 score it assigns to its own choice
    3. Rationale -- a 1-2 sentence explanation of why it chose that PT
       (important for regulatory audit trails)
    4. Human review flagging -- cases below confidence 0.5 are flagged

What is Ollama?
    Ollama is a tool that runs LLMs locally on your own hardware, without sending
    data to any external API (like OpenAI or Anthropic). This is essential for
    privacy-by-design: MAUDE data is technically public, but the architecture
    demonstrates GDPR compliance for real clinical data contexts.

    The model runs on the Hetzner server's CPU/RAM.
    We use llama3.2:3b -- a 3-billion parameter model chosen because:
    - It fits in RAM on the CX33 server (8 GB, shared with 9 Docker containers)
    - It produces reliable JSON output when prompted correctly
    - Larger models (7B, 14B) would cause out-of-memory errors

The confidence formula (final stage, in coding.py):
    final_confidence = 0.3 * sigmoid(CrossEncoder_score)
                     + 0.7 * LLM_confidence
    Both components contribute -- the CrossEncoder provides a signal from the
    retrieval stage; the LLM provides its self-assessed certainty.

This module was explored in Notebook 07_meddra_llm_coding.ipynb.
"""

import json
import re
import os
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not installed. Run: pip3 install requests --break-system-packages")

from vigilex.coding.reranker import RerankedResult


# Ollama URL ist NICHT mehr als Modul-Konstante gefuehrt.
# Aufloesung passiert in LLMCoder.__init__ (Argument oder env-var, sonst raise).
# Grund: hardcoded "localhost:11434" hat am 12.05 24h Massen-Fallback erzeugt,
# weil im Docker-Container localhost nicht auf den Host zeigt.
# Siehe CLAUDE.md "Kritischer Befund 2026-05-13".

# The Ollama model to use. Must be pulled on the server first:
#   ollama pull llama3.2
# Bleibt hardcoded -- Architektur-Entscheidung (CX33 RAM-Constraint), nicht Deployment-Config.
OLLAMA_MODEL = "llama3.2:3b"

# ---------------------------------------------------------------------------
# Strict-Mode -- "fail-fast in dev, fail-soft in prod"
# Bei VIGILEX_STRICT=true werden Fehler hart durchgereicht statt
# stillschweigend zu Fallback zu wechseln. Wichtig fuer Development:
# Bugs werden sichtbar, nicht von der graceful-degradation versteckt.
# Production (Default false): Fallback bleibt aktiv, kein Datenverlust.
# ---------------------------------------------------------------------------
STRICT_MODE = os.environ.get("VIGILEX_STRICT", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Groq backend (EXPERIMENTAL -- NOT FOR PRODUCTION)
# WARNING: Using Groq sends report narratives to an external API.
#          Acceptable for benchmarking/capstone dev only.
#          Do NOT use with real patient data or in a production deployment.
#          Privacy-by-Design requires on-premise inference (Ollama).
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"  # fast, free tier, OpenAI-compatible API

# The system prompt tells the LLM what role it should play and what rules to follow.
# Key rules that improve reliability:
#   - "Select exactly ONE PT from the provided candidates" (prevents hallucination)
#   - "do not invent new terms" (prevents the LLM from making up MedDRA codes)
#   - "Output ONLY valid JSON" (structured output for reliable parsing)
#   - "no extra text" (prevents the LLM from adding explanatory prose before/after the JSON)
SYSTEM_PROMPT = """You are a MedDRA coding specialist with expertise in medical device adverse events.
Your task is to select the single most appropriate MedDRA Preferred Term (PT) for a given adverse event narrative.
You will be given:
1. The adverse event narrative (free text from a MAUDE report)
2. A ranked list of candidate MedDRA PTs with their System Organ Class (SOC)

Rules:
- Select exactly ONE PT from the provided candidates -- do not invent new terms
- Base your decision on the PRIMARY adverse event described, not secondary mentions
- Assign a confidence score from 0.0 to 1.0 (1.0 = perfect match, 0.5 = uncertain, <0.3 = flag for review)
- Keep the rationale concise (1-2 sentences)
- Output ONLY valid JSON matching the schema below -- no extra text

Output schema:
{
  "pt_code":    <integer>,
  "pt_name":    "<string>",
  "soc_name":   "<string>",
  "confidence": <float 0.0-1.0>,
  "rationale":  "<string>"
}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(narrative: str, candidates: list[RerankedResult]) -> str:
    """
    Build the user message that contains the narrative and the candidate list.

    Format:
        Adverse event narrative:
        <the report text...>

        Candidate MedDRA PTs (ranked by relevance, best first):
          1. PT: Hypoglycaemia (code: 10020993) | SOC: Metabolism disorders | score: 8.45
          2. PT: Blood glucose decreased (code: 10005881) | SOC: ... | score: 3.21
          ...

        Select the single best PT and respond with JSON only.

    Why include the CrossEncoder score?
        Showing the candidates in order with their scores gives the LLM context
        about how confident the retrieval pipeline was. A very high score for
        the top candidate may reinforce the LLM's choice; a cluster of similar
        scores may make the LLM more cautious (lower confidence output).
    """
    lines = [
        "Adverse event narrative:",
        f'"""{narrative}"""',
        "",
        "Candidate MedDRA PTs (ranked by relevance, best first):",
    ]
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"  {i}. PT: {c.pt_name} (code: {c.pt_code}) | SOC: {c.soc_name} "
            f"| relevance score: {c.crossencoder_score:.2f}"
        )
    lines += [
        "",
        "Select the single best PT and respond with JSON only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodingResult:
    """
    The final output of the three-stage MedDRA coding pipeline for one MAUDE report.

    Fields:
        pt_code:      MedDRA numeric code for the selected Preferred Term
        pt_name:      Human-readable PT name, e.g. "Hypoglycaemia"
        soc_name:     System Organ Class, e.g. "Metabolism and nutrition disorders"
        confidence:   0.0-1.0 confidence score (LLM's self-assessment),
                      or None when LLM is in fallback mode (no value ascertained).
                      Maps to SQL NULL in processed.coding_results.llm_confidence.
        rationale:    1-2 sentence explanation from the LLM (or fallback note)
        raw_response: Full LLM output string (or error message in fallback)
        flagged:      True if confidence < threshold OR fallback occurred
    """
    pt_code:      int
    pt_name:      str
    soc_name:     str
    confidence:   Optional[float]    # None on fallback -> NULL in DB
    rationale:    str
    raw_response: str
    flagged:      bool = False  # set to True if confidence < CONFIDENCE_THRESHOLD or fallback


# ---------------------------------------------------------------------------
# LLM Coder
# ---------------------------------------------------------------------------

class LLMCoder:
    """
    Final MedDRA coding stage using a local LLM via Ollama.

    Communicates with Ollama via its REST API (HTTP POST to /api/chat).
    No internet connection required after the model is pulled -- all inference
    runs locally on the Hetzner server.

    Parameters:
        ollama_url:           Base URL of the Ollama server.
        model:                Model name as known to Ollama (e.g. "llama3.2").
        confidence_threshold: Cases where LLM confidence < this value are flagged
                              for human review in the Streamlit frontend.
        temperature:          Controls randomness of LLM output.
                              0.0 = fully deterministic (always same output for same input)
                              1.0 = very random
                              We use 0.1 for near-deterministic structured JSON output.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,    # vorher hardcoded Default, jetzt None
        model: str = OLLAMA_MODEL,
        confidence_threshold: float = 0.5,
        temperature: float = 0.1,
        use_groq: bool = False,
        groq_api_key: Optional[str] = None,
    ):
        # Ollama-URL-Aufloesung (nur wenn nicht im Groq-Modus):
        #   1. Explizit uebergebenes ollama_url Argument hat Vorrang
        #   2. OLLAMA_BASE_URL aus Environment-Variable
        #   3. RuntimeError -- keine stillen Defaults
        # Im Groq-Modus brauchen wir keine Ollama-URL (nur den API-Key).
        if not use_groq:
            if ollama_url is None:
                ollama_url = os.environ.get("OLLAMA_BASE_URL")
            if not ollama_url:
                raise RuntimeError(
                    "LLMCoder needs ollama_url -- pass it explicitly OR "
                    "set OLLAMA_BASE_URL in the environment. "
                    "No silent localhost default (see CLAUDE.md Befund 2026-05-13)."
                )
            self.ollama_url = ollama_url.rstrip("/")
        else:
            # Groq-Modus: Ollama-URL irrelevant
            self.ollama_url = ""

        self.model = model
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        # Groq backend (experimental, external API -- see WARNING above)
        self.use_groq = use_groq
        self.groq_api_key = groq_api_key
        # Verify the backend is reachable when the coder is initialised
        self._check_connection()

    def _check_connection(self) -> None:
        """
        Verify that the configured backend is reachable.

        In STRICT_MODE: any problem raises RuntimeError -- worker will not start.
        In production mode: warnings only, worker continues (graceful degradation).

        This is the first error-trap: catches config problems before the
        coding loop hides them in fallback records.
        """
        # Groq path
        if self.use_groq:
            if not self.groq_api_key:
                msg = (
                    "--groq requested but GROQ_API_KEY is not set. "
                    "Export GROQ_API_KEY=<your_key> before starting the worker."
                )
                if STRICT_MODE:
                    raise RuntimeError(msg)
                print(f"WARNING: {msg}")
            else:
                print(f"Groq backend active (model: {GROQ_MODEL}). "
                      "WARNING: narratives will be sent to external API.")
            return

        # Ollama path
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                msg = (
                    f"model '{self.model}' not found in Ollama at {self.ollama_url}. "
                    f"Available models: {models}. "
                    f"Run 'ollama pull {self.model}' on the server."
                )
                if STRICT_MODE:
                    raise RuntimeError(msg)
                print(f"WARNING: {msg}")
        except requests.RequestException as e:
            msg = (
                f"Cannot reach Ollama at {self.ollama_url}: {e}. "
                f"Stage 3 would fall back to CrossEncoder results."
            )
            if STRICT_MODE:
                # Dev: harter Abbruch -- Container Crashloop, sichtbar in docker ps
                raise RuntimeError(msg) from e
            print(f"WARNING: {msg}")

    def _call_ollama(self, user_prompt: str) -> str:
        """
        Send a chat request to Ollama and return the model's response text.

        Uses the /api/chat endpoint (multi-turn chat format) which supports
        system + user messages. The system message sets the model's persona
        and rules; the user message contains the actual coding task.

        Parameters:
            stream=False:    wait for the full response before returning
                             (streaming would require handling partial JSON chunks)
            temperature=0.1: low value = near-deterministic output
            num_predict=256: maximum number of tokens in the response
                             (256 is plenty for a short JSON object)

        Returns:
            The model's response text (raw string, may contain markdown formatting).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,
            },
        }
        r = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=60,  # 60 seconds is generous; typical response time is 5-15s on CPU
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _call_groq(self, user_prompt: str) -> str:
        """
        Send a chat request to Groq's OpenAI-compatible API.

        WARNING: This sends the report narrative to an external server (Groq).
                 For benchmarking / capstone development only.
                 NOT for production use with real patient data.

        Model: llama-3.1-8b-instant (fast free-tier model, ~300 tokens/sec)
        API:   https://api.groq.com/openai/v1/chat/completions
        Auth:  Bearer token via GROQ_API_KEY environment variable

        Returns:
            The model's response text (same format as _call_ollama).
        """
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it before running with --groq."
            )
        payload = {
            "model":       GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens":  256,
        }
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type":  "application/json",
        }
        r = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str, candidates: list[RerankedResult]) -> dict:
        """
        Extract and validate the JSON object from the LLM's response.

        LLMs sometimes wrap JSON in markdown code blocks like:
            ```json
            { "pt_code": 10020993, ... }
            ```
        We strip these before parsing.

        Validation:
            - All required fields must be present
            - confidence is clamped to [0.0, 1.0] in case the LLM outputs 1.2 or -0.1

        If parsing fails (invalid JSON, missing fields), the caller's except block
        catches it and falls back to the top CrossEncoder candidate.

        Args:
            raw:        Raw string from the LLM (may include markdown)
            candidates: The candidates that were shown to the LLM (not used here,
                        but available for future validation of pt_code membership)

        Returns:
            Parsed and validated dict with pt_code, pt_name, soc_name, confidence, rationale.
        """
        # Strip markdown code fences: ```json ... ``` or just ``` ... ```
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Find the first {...} block in the cleaned text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")

        parsed = json.loads(match.group())

        # Check that all required fields are present
        required = {"pt_code", "pt_name", "soc_name", "confidence", "rationale"}
        missing = required - set(parsed.keys())
        if missing:
            raise ValueError(f"LLM response JSON is missing required fields: {missing}")

        # Clamp confidence to valid range [0.0, 1.0]
        # (LLMs occasionally output values like 0.95000001 or 1.1 due to floating point)
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))

        return parsed

    def code(
        self,
        narrative: str,
        candidates: list[RerankedResult],
    ) -> CodingResult:
        """
        Select the best MedDRA PT for a MAUDE adverse event narrative.

        This is the main function called by the CodingWorker.
        It orchestrates the prompt construction, LLM call, response parsing,
        and fallback handling.

        Fallback behaviour:
            If the LLM call fails (Ollama down, JSON parse error, timeout, etc.),
            we fall back to the top CrossEncoder candidate with confidence=0.3 and
            flagged=True. This means:
            - The report still gets a coding (no null in the database)
            - The low confidence marks it for human review
            - The pipeline continues -- one failure does not block the batch

        Args:
            narrative:  Full mdr_text from raw.maude_reports (the MAUDE narrative).
            candidates: Top-5 RerankedResult objects from CrossEncoderReranker.rerank().

        Returns:
            CodingResult with the chosen PT code, confidence, rationale, and flagged status.
        """
        if not candidates:
            raise ValueError(
                "No candidates provided. Run hybrid search + reranker first "
                "before calling LLMCoder.code()."
            )

        user_prompt = _build_user_prompt(narrative, candidates)

        try:
            if self.use_groq:
                raw = self._call_groq(user_prompt)
            else:
                raw = self._call_ollama(user_prompt)
            parsed = self._parse_response(raw, candidates)
        except Exception as e:
            if STRICT_MODE:
                # Dev: keine stillen Fallbacks -- Worker bricht ab.
                # So sehen wir Bugs sofort statt nach 24h im DB-Spike (siehe Befund 13.05).
                # Diagnose-Info VOR dem raise, damit man sieht welcher Record es war.
                top = candidates[0]
                print(
                    f"STRICT MODE: LLM coding failed -- aborting worker.\n"
                    f"  top CE candidate: {top.pt_name} (code {top.pt_code}, "
                    f"score {top.crossencoder_score:.2f})\n"
                    f"  error: {type(e).__name__}: {e}"
                )
                raise
            # Production mode: graceful fallback.
            # confidence=None landet in der DB als NULL -- semantisch korrekt
            # ("LLM-Antwort nicht ermittelt"). Aggregations-Funktionen (AVG, MEAN)
            # ueberspringen NULL automatisch -- kein Bias.
            # flagged=True ist zusaetzliche Code-Wahrheit; confidence IS NULL ist
            # die DB-Wahrheit auch ohne flagged-Spalte (Tech-Debt: flagged
            # wird aktuell nicht persistiert).
            print(f"LLM coding failed ({e}). Falling back to top CrossEncoder candidate.")
            top = candidates[0]
            return CodingResult(
                pt_code      = top.pt_code,
                pt_name      = top.pt_name,
                soc_name     = top.soc_name,
                confidence   = None,    # NULL in DB -- LLM ist ausgefallen
                rationale    = f"LLM fallback -- using CrossEncoder top result. Error: {e}",
                raw_response = str(e),
                flagged      = True,   # always flag LLM fallback cases for human review
            )

        # Build the structured result from the parsed LLM output
        result = CodingResult(
            pt_code      = int(parsed["pt_code"]),
            pt_name      = parsed["pt_name"],
            soc_name     = parsed["soc_name"],
            confidence   = parsed["confidence"],
            rationale    = parsed["rationale"],
            raw_response = raw,
            # Flag if the LLM's self-reported confidence is below the threshold
            flagged      = parsed["confidence"] < self.confidence_threshold,
        )
        return result
