"""Select effective policy versions: never retrieve a future revision."""
import json
from datetime import date
from pathlib import Path
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "policies.json"
def select_policies(as_of: str, filename=DATA_FILE) -> dict:
    day = date.fromisoformat(as_of)
    policies = json.loads(Path(filename).read_text(encoding="utf-8"))
    result = {}
    for item in policies:
        start = date.fromisoformat(item["effective_from"])
        if start > day: continue
        prior = result.get(item["id"])
        if prior is None or (start,item["version"]) > (date.fromisoformat(prior["effective_from"]),prior["version"]):
            result[item["id"]] = item
    return result
