"""A tiny stand-in for Ollama AND for OpenAI-compatible APIs (Groq/Gemini/OpenAI) used for testing.

It speaks just enough of the Ollama HTTP API (/api/tags, /api/version,
/api/chat) and answers with schema-valid JSON.  For column/dataset prompts it
"accepts" the candidate actions listed in the prompt (like a well-behaved
model would) and adds a thinking string, so the whole pipeline can be
exercised end-to-end.

Run:  python tests/mock_ollama.py 11435
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "llama3.2:3b"


def _candidate_actions(prompt: str, column: str | None):
    actions = []
    in_cands = False
    for line in prompt.splitlines():
        if line.startswith("CANDIDATE ACTIONS"):
            in_cands = True
            continue
        if in_cands and line.startswith("- "):
            m = re.match(r"- (\w+) (\{.*?\})\s*(?:#\s*(.*))?$", line)
            if not m:
                continue
            name, params, reason = m.group(1), m.group(2), m.group(3) or "candidate"
            try:
                p = json.loads(params)
            except json.JSONDecodeError:
                p = {}
            if "mapping" in p and isinstance(p["mapping"], str):
                p.pop("mapping")
            actions.append({"action": name, "column": column or "", "params": p, "reason": reason, "confidence": 0.8})
        elif in_cands and line.strip() == "":
            in_cands = in_cands and not actions
    return actions


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path.startswith("/openai/v1/models"):
            return self._json({"object": "list", "data": [{"id": MODEL}, {"id": "llama-3.3-70b-versatile"}, {"id": "gemini-2.5-flash"}]})
        if self.path.startswith("/api/tags"):
            return self._json({"models": [{"name": MODEL}, {"name": "qwen2.5:1.5b"}]})
        if self.path.startswith("/api/version"):
            return self._json({"version": "0.5.7-mock"})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        openai_style = "/chat/completions" in self.path
        if not (self.path.startswith("/api/chat") or openai_style):
            return self._json({"error": "not found"}, 404)
        if openai_style and not self.headers.get("Authorization", "").startswith("Bearer "):
            return self._json({"error": {"message": "missing key"}}, 401)
        if not openai_style and req.get("model") != MODEL:
            return self._json({"error": f"model '{req.get('model')}' not found"}, 404)
        user = next((m["content"] for m in req.get("messages", []) if m["role"] == "user"), "")
        system = next((m["content"] for m in req.get("messages", []) if m["role"] == "system"), "")
        if "Summarise the cleaning plan" in system:
            content = {"summary": "The dataset had placeholder nulls, numbers and dates stored as text, inconsistent "
                                  "category spellings and duplicate rows. The plan converts types, merges spellings, "
                                  "blanks impossible values, imputes gaps and removes duplicate or empty rows. The main "
                                  "judgement calls are clipping salary outliers and the date day/month order."}
        elif user.startswith("FILE:"):
            content = {"thinking": "The table has empty rows, a fully empty column, a constant column and an exact duplicate column; "
                                   "duplicate records exist, including ones that only differ in case. Column names need snake_case.",
                       "actions": _candidate_actions(user, None)}
            # also drop the empty column based on the prompt text
        else:
            # one or several "COLUMN: '<name>' (position n)" sections
            sections = re.split(r"(?m)^(?==+ COLUMN \d+ of \d+ =+$)", user)
            acts, cols = [], []
            for sec in sections:
                m = re.search(r"COLUMN: '(.+?)' \(position", sec) or re.search(r'COLUMN: "(.+?)" \(position', sec)
                if not m:
                    continue
                col = m.group(1)
                cols.append(col)
                acts += _candidate_actions(sec, col)
                # pretend the model knows a synonym / uses an alias
                if col == "City":
                    acts.append({"action": "normalize_categories", "column": col,
                                 "params": {"mapping": [{"from": "Bengaluru", "to": "Bangalore"}, {"from": "Not A City", "to": "X"}]},
                                 "reason": "Bengaluru is the same city as Bangalore.", "confidence": 0.7})
                if col == "Age":
                    acts.append({"action": "fill_na", "column": col, "params": {"strategy": "average"}, "reason": "alias test", "confidence": 0.5})
            thinking = f"{cols} look like they need {len(acts)} fix(es) based on the evidence; keeping the sensible defaults."
            content = {"thinking": thinking, "actions": acts}
        if openai_style:
            return self._json({"id": "chatcmpl-mock", "object": "chat.completion", "model": req.get("model"),
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 456, "completion_tokens": 123}})
        self._json({"model": MODEL, "message": {"role": "assistant", "content": json.dumps(content)},
                    "done": True, "eval_count": 123, "prompt_eval_count": 456})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11435
    print(f"mock ollama on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
