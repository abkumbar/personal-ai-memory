#!/usr/bin/env python3
"""Read-only MCP gateway over published memories (Streamable HTTP, JSON responses).

  python3 mcp_gateway.py token create NAME     prints a new bearer token once
  python3 mcp_gateway.py token revoke NAME     takes effect on the next request
  python3 mcp_gateway.py token list
  python3 mcp_gateway.py serve [--port 8765]   listens on 127.0.0.1 only by default

Tools: search_memory(query, project=None, limit=5), get_memory(memory_id),
get_project_context(project_id).

Guarantees:
- Reads only the published directory (default ~/GrokBotContext/memories.jsonl), never the
  vault. It refuses to start if that directory is inside the vault.
- The published file is re-read whenever it changes, so an unpublished memory stops being
  served immediately.
- Every request needs `Authorization: Bearer <token>`. Tokens are stored as SHA-256 hashes.
- Requests with an Origin header from an unexpected site are rejected (DNS rebinding).
- Every tool call is logged to state/mcp_access.jsonl with token name, tool, arguments and
  the memory IDs returned.

Standard library only.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import sys
import threading

import ingest

PUBLISH_DIR = Path.home() / "GrokBotContext"
VAULT = Path.home() / "personal_chats"
TOKENS_PATH = ingest.STATE_DIR / "mcp_tokens.json"
ACCESS_LOG = ingest.STATE_DIR / "mcp_access.jsonl"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "personal-memory", "version": "1.0.0"}
MAX_BODY = 64 * 1024
MAX_LIMIT = 20
MAX_QUERY = 500
STOPWORDS = set("a an and are as at be by for from has have i in is it of on or that the this to was "
                "were will with what which who how why when user users".split())

INSTRUCTIONS = ("Read-only access to memories the user explicitly approved for sharing. "
                "Records with attribution ai_suggested are AI advice, not the user's decisions. "
                "Each record carries source IDs for traceability; the original conversations are not available.")

TOOLS = [
    {"name": "search_memory",
     "description": "Search the user's approved memories. Returns the best matches with their statements.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look for."},
         "project": {"type": "string", "description": "Optional project name to restrict the search."},
         "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": 5}},
         "required": ["query"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "get_memory",
     "description": "Fetch one approved memory by its id (for example mem-0123456789abcdef).",
     "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}},
                     "required": ["memory_id"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "get_project_context",
     "description": "All approved memories for one project, grouped by category, oldest first. "
                    "An unknown project returns the list of available projects.",
     "inputSchema": {"type": "object", "properties": {"project_id": {"type": "string"}},
                     "required": ["project_id"]},
     "annotations": {"readOnlyHint": True}},
]


def now():
    return ingest.iso(datetime.now(timezone.utc))


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).casefold()) if w not in STOPWORDS]


def digest(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --- Data -------------------------------------------------------------------

class MemoryStore:
    """Published memories, reloaded whenever memories.jsonl changes."""

    def __init__(self, publish_dir):
        self.path = publish_dir / "memories.jsonl"
        self.lock = threading.Lock()
        self.stamp = None
        self.records, self.by_id, self.index = [], {}, None

    def refresh(self):
        stamp = self.path.stat().st_mtime_ns if self.path.exists() else None
        with self.lock:
            if stamp == self.stamp:
                return
            text = self.path.read_text(encoding="utf-8") if stamp else ""
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            docs = [Counter(words(" ".join([r["title"], r["title"], r["statement"], r.get("project", "")])))
                    for r in records]
            df = Counter(w for d in docs for w in d)
            avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
            self.records, self.by_id = records, {r["id"]: r for r in records}
            self.index = (docs, df, avg)
            self.stamp = stamp

    def search(self, query, project=None, limit=5):
        self.refresh()
        docs, df, avg = self.index
        terms = words(query)
        n = len(self.records)
        results = []
        for record, doc in zip(self.records, docs):
            if project and norm(record.get("project", "")) != norm(project):
                continue
            length = sum(doc.values()) or 1
            score = 0.0
            for term in terms:
                if term in doc:
                    idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    tf = doc[term]
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / (avg or 1)))
            if score > 0:
                results.append((round(score, 3), record))
        results.sort(key=lambda item: -item[0])
        return [dict(public(record), score=score) for score, record in results[:limit]]

    def get(self, memory_id):
        self.refresh()
        record = self.by_id.get(str(memory_id).strip())
        return public(record) if record else None

    def project(self, project_id):
        self.refresh()
        wanted = norm(project_id)
        matches = [r for r in self.records if norm(r.get("project", "")) == wanted]
        if not matches:
            return None, sorted(Counter(r.get("project") or "(none)" for r in self.records).items())
        grouped = defaultdict(list)
        for record in sorted(matches, key=lambda r: (r.get("created", ""), r["id"])):
            grouped[record["category"]].append(public(record))
        return {"project": matches[0]["project"], "memory_count": len(matches), "by_category": grouped}, None


def public(record):
    keys = ("id", "title", "statement", "project", "category", "attribution", "created",
            "source", "source_conversation_id", "source_message_ids")
    return {k: record.get(k) for k in keys}


# --- Tokens -----------------------------------------------------------------

class Tokens:
    def __init__(self, path=TOKENS_PATH):
        self.path = path
        self.stamp, self.active = None, {}

    def load(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"tokens": []}

    def save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ingest.write_atomic(self.path, json.dumps(data, indent=1) + "\n")
        self.path.chmod(0o600)

    def check(self, header):
        """Return the token name for a valid bearer header, else None. Reloads on change."""
        stamp = self.path.stat().st_mtime_ns if self.path.exists() else None
        if stamp != self.stamp:
            self.active = {t["sha256"]: t["name"] for t in self.load()["tokens"] if not t.get("revoked")}
            self.stamp = stamp
        # xAI's `authorization` field may arrive with or without the "Bearer " prefix.
        match = re.fullmatch(r"(?:Bearer\s+)?(\S+)", (header or "").strip())
        if not match:
            return None
        presented = digest(match.group(1))
        for stored, name in self.active.items():
            if hmac.compare_digest(stored, presented):
                return name
        return None

    def create(self, name):
        data = self.load()
        if any(t["name"] == name and not t.get("revoked") for t in data["tokens"]):
            raise SystemExit(f"An active token named {name!r} already exists. Revoke it first.")
        token = "pmem_" + secrets.token_urlsafe(32)
        data["tokens"].append({"name": name, "sha256": digest(token), "created": now(), "revoked": None})
        self.save(data)
        return token

    def revoke(self, name):
        data = self.load()
        hits = [t for t in data["tokens"] if t["name"] == name and not t.get("revoked")]
        for token in hits:
            token["revoked"] = now()
        self.save(data)
        return len(hits)


# --- MCP --------------------------------------------------------------------

def call_tool(store, name, args):
    """Return (payload, is_error, returned ids)."""
    if name == "search_memory":
        query = str(args.get("query", ""))[:MAX_QUERY]
        if not query.strip():
            return {"error": "query is required"}, True, []
        try:
            limit = max(1, min(int(args.get("limit") or 5), MAX_LIMIT))
        except (TypeError, ValueError):
            limit = 5
        results = store.search(query, args.get("project") or None, limit)
        return {"results": results}, False, [r["id"] for r in results]
    if name == "get_memory":
        record = store.get(args.get("memory_id", ""))
        if record is None:
            return {"error": "No approved memory with that id."}, True, []
        return record, False, [record["id"]]
    if name == "get_project_context":
        context, available = store.project(args.get("project_id", ""))
        if context is None:
            return {"error": "Unknown project.",
                    "available_projects": [{"project": p, "memories": n} for p, n in available]}, True, []
        ids = [r["id"] for group in context["by_category"].values() for r in group]
        return context, False, ids
    return {"error": f"Unknown tool {name}"}, True, []


def handle_rpc(message, store, token_name, client, log):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    method, params, msg_id = message["method"], message.get("params") or {}, message.get("id")
    if method != "tools/call":  # Tool calls are logged below with their results.
        client_info = params.get("clientInfo") if method == "initialize" else None
        log({"at": now(), "token": token_name, "client": client, "method": method,
             **({"client_info": client_info} if client_info else {})})
    if msg_id is None:  # Notification: nothing to send back.
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        result = {"protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        payload, is_error, ids = call_tool(store, name, args if isinstance(args, dict) else {})
        log({"at": now(), "token": token_name, "client": client, "method": method, "tool": name,
             "arguments": args, "returned": ids, "error": is_error})
        result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                  "structuredContent": payload, "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def make_handler(store, tokens, access_log, allowed_origins):
    log_lock = threading.Lock()

    def log(entry):
        with log_lock:
            access_log.parent.mkdir(parents=True, exist_ok=True)
            with access_log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    class Handler(BaseHTTPRequestHandler):
        server_version = "personal-memory-mcp"
        sys_version = ""

        def log_message(self, *args):
            pass

        def reply(self, status, body=None, headers=()):
            data = b"" if body is None else json.dumps(body).encode("utf-8")
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def gate(self):
            origin = self.headers.get("Origin")
            if origin and origin not in allowed_origins:
                self.reply(403, {"error": "origin not allowed"})
                return None
            name = tokens.check(self.headers.get("Authorization"))
            if name is None:
                log({"at": now(), "token": None, "client": self.client_address[0], "http": self.command,
                     "user_agent": self.headers.get("User-Agent"), "tool": None, "error": "unauthorized"})
                self.reply(401, {"error": "unauthorized"}, [("WWW-Authenticate", "Bearer")])
            return name

        def do_POST(self):
            if self.path.split("?")[0] != "/mcp":
                return self.reply(404, {"error": "not found"})
            name = self.gate()
            if name is None:
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self.reply(413 if length > MAX_BODY else 400, {"error": "bad body size"})
            try:
                message = json.loads(self.rfile.read(length))
            except ValueError:
                return self.reply(400, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32700, "message": "Parse error"}})
            client = self.client_address[0]
            if isinstance(message, list):
                responses = [r for r in (handle_rpc(m, store, name, client, log) for m in message) if r]
                return self.reply(200, responses) if responses else self.reply(202)
            response = handle_rpc(message, store, name, client, log)
            return self.reply(200, response) if response else self.reply(202)

        def do_GET(self):
            if self.gate() is not None:
                self.reply(405, {"error": "this server does not stream; use POST"}, [("Allow", "POST")])

        def do_DELETE(self):
            if self.gate() is not None:
                self.reply(405, {"error": "sessions are not used"}, [("Allow", "POST")])

    return Handler


def make_server(host, port, publish_dir=PUBLISH_DIR, tokens=None, access_log=ACCESS_LOG,
                vault=VAULT, allowed_origins=()):
    resolved, vault_root = publish_dir.resolve(), vault.resolve()
    if resolved == vault_root or vault_root in resolved.parents:
        raise SystemExit(f"Refusing to serve {publish_dir}: it is inside the vault.")
    store = MemoryStore(publish_dir)
    store.refresh()
    origins = set(allowed_origins) | {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    handler = make_handler(store, tokens or Tokens(), access_log, origins)
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    token = sub.add_parser("token")
    token.add_argument("action", choices=["create", "revoke", "list"])
    token.add_argument("name", nargs="?")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--publish-dir", type=Path, default=PUBLISH_DIR)
    serve.add_argument("--allow-origin", action="append", default=[],
                       help="extra allowed Origin, for example your HTTPS tunnel URL")
    args = parser.parse_args()

    if args.command == "token":
        tokens = Tokens()
        if args.action == "list":
            for t in tokens.load()["tokens"]:
                print(f"{t['name']}: created {t['created']}, " + (f"revoked {t['revoked']}" if t.get("revoked") else "active"))
            return 0
        if not args.name:
            parser.error("token create/revoke needs a NAME")
        if args.action == "create":
            print(tokens.create(args.name))
            print("Store this now; it is not shown again.", file=sys.stderr)
        else:
            print(f"revoked {tokens.revoke(args.name)} token(s) named {args.name!r}")
        return 0

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Only loopback hosts are allowed. Reach it remotely through an HTTPS tunnel.")
    server = make_server(args.host, args.port, args.publish_dir, allowed_origins=args.allow_origin)
    print(f"serving MCP on http://{args.host}:{args.port}/mcp from {args.publish_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
