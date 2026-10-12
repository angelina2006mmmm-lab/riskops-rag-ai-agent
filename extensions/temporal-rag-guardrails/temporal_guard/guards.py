"""Heuristic demo safeguards, not comprehensive model security."""
import math,re
PATTERNS = (
 r"ignore (?:all |any )?(?:previous|prior|system) instructions",
 r"reveal (?:the |your )?(?:secret|api key|system prompt)",
 r"execute (?:shell|python|bash) (?:code|command)",
 r"(?:curl|wget) https?://",
)
def looks_like_injection(text: str) -> bool:
    return any(re.search(p,text,re.I) for p in PATTERNS)
def valid_number(value) -> bool:
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value >= 0
def redact_text(text: str) -> str:
    return re.sub(r"(?i)sk-[A-Za-z0-9_-]{10,}","[REDACTED_KEY]",text)[:250]
