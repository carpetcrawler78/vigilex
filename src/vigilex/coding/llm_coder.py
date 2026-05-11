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
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not installed. Run: pip3 install requests --break-system-packages")

from vigilex.coding.reranker import RerankedResult


# Default Ollama server URL. When running locally with an SSH tunnel:
#   ssh -L 11434:localhost:11434 cap@46.225.109.99
# the server at localhost:11434 forwards to the Hetzner server's Ollama.
OLLAMA_DEFAULT_URL = "http://localhost:11434"

# The Ollama model to use. Must be pulled on the server first:
#   ollama pull llama3.2
OLLAMA_MODEL = "llama3.2:3b"

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
        """the report text..."""

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
        confidence:   0.0-1.0 confidence score (LLM's self-assessment)
        rationale:    1-2 sentence explanation from the LLM
        raw_response: Full LLM output string (stored for debugging)
        flagged:      True if confidence < threshold (signals need for human review)
    """
    pt_code:      int
    pt_name:      str
    soc_name:     str
    confidence:   float
    rationale:    str
    raw_response: str
    flagged:      bool = False  # set to True if confidence < CONFIDENCE_THRESHOLD


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
        ollama_url: str = OLLAMA_DEFAULT_URL,
        model: str = OLLAMA_MODEL,
        confidence_threshold: float = 0.5,
        temperature: float = 0.1,
        use_groq: bool = False,
        groq_api_key: Optional[str] = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        # Groq backend (experimental, external API -- see WARNING above)
        self.use_groq    = use_groq
        self.groq_api_key = groq_api_key
        # Verify the backend is reachable when the coder is initialised
        self._check_connection()

    def _check_connection(self) -> None:
        """
        Verify that the configured backend is reachable.

        Groq: checks that GROQ_API_KEY is set (no network call at init).
        Ollama: checks that the server is reachable and the model is loaded.

        Soft check -- warnings only, no exceptions raised.
        """
        if self.use_groq:
            if not self.groq_api_key:
                print(
                    "WARNING: --groq requested but GROQ_API_KEY is not set. "
                    "Export GROQ_API_KEY=<your_key> before starting the worker."
                )
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
                print(
                    f"WARNING: model '{self.model}' not found in Ollama. "
                    f"Available models: {models}. "
                    f"Run 'ollama pull {self.model}' on the server."
                )
        except requests.RequestException as e:
            print(
                f"WARNING: Cannot reach Ollama at {self.ollama_url}: {e}. "
                f"Stage 3 will fall back to CrossEncoder results if the server remains unreachable."
            )

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
            # Graceful fallback: use the top CrossEncoder result with a low confidence.
            # flagged=True ensures this case appears in the human review queue.
            print(f"LLM coding failed ({e}). Falling back to top CrossEncoder candidate.")
            top = candidates[0]
            return CodingResult(
                pt_code      = top.pt_code,
                pt_name      = top.pt_name,
                soc_name     = top.soc_name,
                confidence   = 0.3,    # low but non-zero fallback confidence
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
