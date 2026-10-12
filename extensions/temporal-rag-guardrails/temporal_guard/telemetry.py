"""Local demo counters; no prompts, raw inputs, or personal data stored."""
from collections import Counter
from threading import Lock
_lock=Lock()
_outcomes=Counter()
_requests=0
_latency_ms_sum=0.0
_latency_ms_max=0.0
def record(status:str,latency_ms:float)->None:
 global _requests,_latency_ms_sum,_latency_ms_max
 with _lock:
  _requests+=1
  _outcomes[status]+=1
  _latency_ms_sum+=latency_ms
  _latency_ms_max=max(_latency_ms_max,latency_ms)
def snapshot()->dict:
 with _lock:
  return {"requests":_requests,"outcomes":dict(_outcomes),
          "avg_latency_ms":round(_latency_ms_sum/_requests,3) if _requests else 0.0,
          "max_latency_ms":round(_latency_ms_max,3),
          "limitations":"Process-local synthetic demo counters; reset on restart, not production observability."}
