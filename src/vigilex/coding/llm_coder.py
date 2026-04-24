"""
llm_coder.py -- LLM-based final MedDRA PT selection via Ollama.

Takes the Top-5 reranked candidates from CrossEncoderReranker and asks
an LLM (llama3.2 via Ollama) to select the single best PT with a
structured JSON output including confidence score and justification.

Why LLM as final stage?
  The CrossEncoder gives us a ranked list but no semantic reasoning.
  The LLM can:
    - Read the full narrative (not just the PT name)
    - Understand clinical context and negation
    - Produce an interpretable confidence score
    - Flag low-confidence cases for human review
    - Output a structured, auditable decision

Output schema:
  {
    "pt_code":     10018429,
    "pt_name":     "Hypoglycaemia",
    "confidence":  0.92,          -- float 0.0-1.0
    "soc_name":    "Metabolism and nutrition disorders",
    "rationale":   "The narrative describes classic hypoglycaemic symptoms..."
  }

Usage:
  from vigilex.coding.llm_coder import LLMCoder
  coder = LLMCoder(ollama_url="http://localhost:11434")
  result = coder.code(narrative, reranked_candidates)
"""

import json
import sys
import re
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not installed. Run: pip3 install requests --break-system-packages")

from vigilex.coding.reranker import RerankedResult


OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_MODEL       = "llama3.2"

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


def _build_user_prompt(narrative: str, candidates: list[RerankedResult]) -> str:
    """Format the user message with narrative and candidate list."""
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


@dataclass
class CodingResult:
    pt_code:    int
    pt_name:    str
    soc_name:   str
    confidence: float          # 0.0 - 1.0
    rationale:  str
    raw_response: str          # full LLM output for debugging
    flagged:    bool = False   # True if confidence < threshold


class LLMCoder:
    """
    Final MedDRA coding stage using Ollama LLM.

    Parameters
    ----------
    ollama_url : str
        Base URL of the Ollama server. Default: http://localhost:11434
    model : str
        Ollama model name. Default: llama3.2
    confidence_threshold : float
        Cases below this confidence are flagged for human review. Default: 0.5
    temperature : float
        LLM temperature. Low values (0.1-0.2) give more deterministic output.
        For structured coding tasks, low temperature is preferred. Default: 0.1
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_DEFAULT_URL,
        model: str = OLLAMA_MODEL,
        confidence_threshold: float = 0.5,
        temperature: float = 0.1,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify Ollama is reachable and the model is available."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                print(f"WARNING: model '{self.model}' not found in Ollama. "
                      f"Available: {models}. Run: ollama pull {self.model}")
        except requests.RequestException as e:
            print(f"WARNING: Cannot reach Ollama at {self.ollama_url}: {e}")

    def _call_ollama(self, user_prompt: str) -> str:
        """Send a chat request to Ollama and return the response text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,   # enough for JSON output
            },
        }
        r = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _parse_response(self, raw: str, candidates: list[RerankedResult]) -> dict:
        """
        Extract JSON from LLM response.

        LLMs sometimes wrap JSON in markdown code blocks -- strip those first.
        Falls back to the top CrossEncoder candidate if parsing fails.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Find JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

        parsed = json.loads(match.group())

        # Validate required fields
        required = {"pt_code", "pt_name", "soc_name", "confidence", "rationale"}
        missing = required - set(parsed.keys())
        if missing:
            raise ValueError(f"JSON missing fields: {missing}")

        # Clamp confidence to [0.0, 1.0]
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))

        return parsed

    def code(
        self,
        narrative: str,
        candidates: list[RerankedResult],
    ) -> CodingResult:
        """
        Select the best MedDRA PT for an adverse event narrative.

        Parameters
        ----------
        narrative : str
            Free-text adverse event description (MAUDE mdr_text excerpt).
        candidates : list[RerankedResult]
            Top-5 candidates from CrossEncoderReranker.rerank().

        Returns
        -------
        CodingResult with pt_code, pt_name, confidence, rationale, flagged.
        """
        if not candidates:
            raise ValueError("No candidates provided -- run hybrid search + reranker first.")

        user_prompt = _build_user_prompt(narrative, candidates)

        try:
            raw = self._call_ollama(user_prompt)
            parsed = self._parse_response(raw, candidates)
        except Exception as e:
            # Fallback: return top CrossEncoder candidate with low confidence
            print(f"LLM coding failed ({e}), falling back to top CE candidate.")
            top = candidates[0]
            return CodingResult(
                pt_code      = top.pt_code,
                pt_name      = top.pt_name,
                soc_name     = top.soc_name,
                confidence   = 0.3,
                rationale    = f"LLM fallback -- CrossEncoder top candidate. Error: {e}",
                raw_response = str(e),
                flagged      = True,
            )

        result = CodingResult(
            pt_code      = int(parsed["pt_code"]),
            pt_name      = parsed["pt_name"],
            soc_name     = parsed["soc_name"],
            confidence   = parsed["confidence"],
            rationale    = parsed["rationale"],
            raw_response = raw,
            flagged      = parsed["confidence"] < self.confidence_threshold,
        )
        return result
