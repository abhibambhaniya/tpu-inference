"""
Forward-pass logger for vLLM profiling.
Runs in the EngineCore subprocess (GPU worker process).

Activated by env vars:
  VLLM_PROF_MODEL      — model name (used in output filename)
    VLLM_PROF_OUTPUT_DIR — directory to write forward pass CSV

Each row written:
    fwd_id,start_ts,end_ts,duration_ms,req_ids,num_tokens,total_tokens,requests

Complex fields (req_ids, num_tokens, requests) are JSON-encoded strings.
"""
import csv
import json
import os
import threading
from typing import Dict, List, Optional

_fp_logger: Optional["ForwardPassLogger"] = None
_fp_logger_lock = threading.Lock()


class ForwardPassLogger:
    def __init__(self, model: str, output_dir: str, filetype: str = "mist", csv_filename: Optional[str] = None):
        os.makedirs(output_dir, exist_ok=True)
        safe_model = model.replace("/", "_")
                
        if csv_filename:
            self.path = os.path.join(output_dir, csv_filename)
        else:
            self.path = os.path.join(output_dir, f"{safe_model}_{filetype}_fwd.csv")
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self._file = open(self.path, "a", newline="")
        self.filetype = filetype
        if self.filetype == "mist":
            self._fieldnames = ["Prefill", "Context", "Decode", "Time (ms)"]
        else:
            self._fieldnames = [
                "fwd_id",
                "start_ts",
                "end_ts",
                "duration_ms",
                "req_ids",
                "num_tokens",
                "total_tokens",
                "requests[num_scheduled_tokens, past_kv_cache_size]",
            ]
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, delimiter=";")
        if needs_header:
            self._writer.writeheader()
            self._file.flush()
        self._lock = threading.Lock()
        self._counter = 0
        print(f"[profiling] ForwardPassLogger initialized: {self.path}", flush=True)
        self.request_id_dict = {}

    def _get_or_create_req_num(self, req_id: str) -> int:
        req_num = self.request_id_dict.get(req_id)
        if req_num is None:
            req_num = len(self.request_id_dict)
            self.request_id_dict[req_id] = req_num
        return req_num
        
    def record(
        self,
        start_ts: float,
        end_ts: float,
        duration_ms: float,
        req_ids: List[str],
        num_tokens: Dict[str, int],
        requests: Optional[Dict[str, dict]] = None,
    ):
        with self._lock:
            req_nums = [self._get_or_create_req_num(req_id) for req_id in req_ids]
            fwd_id = self._counter
            self._counter += 1
            if self.filetype == "mist":
                # Mist format: Prefill, Context, Decode, Time (ms)
                prefills = []
                contexts = []
                decodes = []
                for req_id, req_details in requests.items():
                    if req_details.get("prefill"):
                        prefills.append(req_details["prefill"])
                    else:
                        contexts.append(req_details["context"])
                        decodes.append(req_details["decode"])
                entry = {
                    "Prefill": str(prefills),
                    "Context": str(contexts),
                    "Decode": str(decodes),
                    "Time (ms)": duration_ms,
                }
            else:
                entry = {
                "fwd_id": fwd_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": duration_ms,
                "req_ids": json.dumps(req_nums),
                "num_tokens": json.dumps({self._get_or_create_req_num(req_id): num for req_id, num in num_tokens.items()}),
                "total_tokens": sum(num_tokens.values()),
                "requests[num_scheduled_tokens, past_kv_cache_size]": "",
            }
                if requests is not None:
                    entry["requests[num_scheduled_tokens, past_kv_cache_size]"] = json.dumps({ self._get_or_create_req_num(req_id): req for req_id, req in requests.items()})
            # print(f"[profiling] Logging forward pass: {entry}", flush=True)
            self._writer.writerow(entry)
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()


def get_fp_logger(csv_filename: str = None) -> Optional["ForwardPassLogger"]:
    global _fp_logger
    if _fp_logger is not None:
        return _fp_logger
    # Lazy init from env vars (EngineCore subprocess inherits parent env)
    model = os.environ.get("VLLM_PROF_MODEL", "")
    output_dir = os.environ.get("VLLM_PROF_OUTPUT_DIR", "")
    # print(f"[profiling] get_fp_logger: model={model}, output_dir={output_dir}", flush=True)
    if not model or not output_dir:
        return None
    with _fp_logger_lock:
        if _fp_logger is None:
            _fp_logger = ForwardPassLogger(model, output_dir, filetype=os.environ.get("VLLM_PROFILE_TYPE", "default"), csv_filename=csv_filename)
    return _fp_logger
