#!/usr/bin/env python3
"""Ask Grok one question that it can only answer through the personal memory gateway.

  XAI_API_KEY=... python3 try_grok.py "What have I decided about the sleep app?"

Reads your xAI key from the XAI_API_KEY environment variable and the gateway token from
state/mcp_token_grok-bot.txt. Neither is printed. The gateway URL comes from --url, or
the MEMORY_GATEWAY_URL environment variable. Prints Grok's answer, the MCP calls xAI reports, and the new
entries in the gateway access log, so you can see Grok actually used the gateway.

Standard library only.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
API = "https://api.x.ai/v1"


def request(method, path, key, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(API + path, data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"xAI API error {error.code}: {error.read().decode('utf-8', 'replace')[:600]}")


GATEWAY_URL = os.environ.get("MEMORY_GATEWAY_URL", "")  # e.g. your Tailscale Funnel https URL.


def tunnel_url():
    return GATEWAY_URL


def pick_model(key):
    ids = [m["id"] for m in request("GET", "/models", key).get("data", [])]
    preferred = [i for i in ids if i.startswith("grok-4")] or [i for i in ids if i.startswith("grok")]
    if not preferred:
        sys.exit(f"No Grok model found. Available: {ids}")
    return sorted(preferred)[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default="What do you know about my food photo device project?")
    parser.add_argument("--url", help="public gateway base URL, e.g. https://your-machine.your-tailnet.ts.net (or set MEMORY_GATEWAY_URL)")
    parser.add_argument("--model", help="Grok model id (default: newest grok-4 model listed)")
    args = parser.parse_args()

    key = os.environ.get("XAI_API_KEY")
    if not key:
        sys.exit("Set XAI_API_KEY in your environment first.")
    token = (HERE / "state" / "mcp_token_grok-bot.txt").read_text().strip()
    base = (args.url or tunnel_url() or "").rstrip("/")
    if not base.startswith("https://"):
        sys.exit("No public HTTPS gateway URL. Pass --url or start the tunnel.")
    model = args.model or pick_model(key)

    log_path = HERE / "state" / "mcp_access.jsonl"
    seen = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    print(f"model: {model}\ngateway: {base}/mcp\n", flush=True)

    result = request("POST", "/responses", key, {
        "model": model,
        "input": [{"role": "system", "content": "Use the personal_memory tools to answer from the user's "
                                                "approved memories. Say which memories you used."},
                  {"role": "user", "content": args.question}],
        "tools": [{"type": "mcp", "server_url": f"{base}/mcp", "server_label": "personal_memory",
                   "authorization": token,
                   "allowed_tools": ["search_memory", "get_memory", "get_project_context"]}],
    })

    answer, calls = [], []
    for item in result.get("output", []):
        if item.get("type") == "message":
            answer += [c.get("text", "") for c in item.get("content", []) if c.get("type") in ("output_text", "text")]
        elif str(item.get("type", "")).startswith("mcp"):
            calls.append({k: item.get(k) for k in ("type", "name", "arguments", "error") if item.get(k)})
    print("Grok's answer:\n" + ("\n".join(answer) or json.dumps(result)[:2000]))
    print("\nMCP activity reported by xAI:")
    for call in calls or [{"note": "none reported"}]:
        print(" ", json.dumps(call)[:300])
    print("\nNew gateway access-log entries:")
    new = log_path.read_text().splitlines()[seen:] if log_path.exists() else []
    for line in new or ["  none: Grok did not reach the gateway"]:
        entry = json.loads(line) if line.startswith("{") else None
        print(" ", f"{entry['at']} token={entry['token']} tool={entry['tool']} returned={len(entry.get('returned') or [])} "
                   f"error={entry.get('error')}" if entry else line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
