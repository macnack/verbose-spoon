import os, json, gzip, atexit, hashlib, time
from pathlib import Path
from typing import Optional

class SampleRecorder:
    def __init__(self, path: str, gzip_enabled: bool = True, header: Optional[dict] = None):
        self.path = Path(path)
        self.gzip_enabled = gzip_enabled
        self._fh = None
        self.header = header or {}
        self._open()
        atexit.register(self.close)

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "at"  # append text
        if self.gzip_enabled:
            self._fh = gzip.open(self.path, mode)  # .jsonl.gz
        else:
            self._fh = open(self.path, mode, encoding="utf-8")
        # write a header record once
        hdr = {"__header__": True, "ts": time.time(), **self.header}
        self._fh.write(json.dumps(hdr) + "\n")
        self._fh.flush()

    def write(self, record: dict):
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(self):
        self._fh.flush()

    def close(self):
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

def short_hash(s: str, n=10) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]
