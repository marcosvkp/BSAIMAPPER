import json
import os
from datetime import datetime


class GenerationLogger:
    def __init__(self, log_dir="logs", session_name="generation"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{session_name}_{ts}.jsonl")
        self._events = []

    def log(self, event_type, **payload):
        data = {
            "ts_utc": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            **payload,
        }
        self._events.append(data)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def summary(self):
        return self.path, len(self._events)
