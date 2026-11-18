import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import mcp_gateway as G


def record(memory_id, title, statement, project="Northwind", category="technical_decision",
           attribution="user_stated", created="2026-01-02"):
    return {"id": memory_id, "title": title, "statement": statement, "project": project,
            "category": category, "attribution": attribution, "created": created, "source": "chatgpt",
            "source_conversation_id": "chatgpt-c1", "source_message_ids": ["m1"],
            "approved_by": "manual", "approved_hash": "abc"}


RECORDS = [
    record("mem-1", "Postgres for meal logs", "The user chose Postgres for meal logs because queries are relational."),
    record("mem-2", "Eval lab", "The user is building an evaluation lab for agents.", category="project_context",
           created="2026-01-01"),
    record("mem-3", "Sleep app pricing", "The user plans a freemium price for the sleep app.", project="sleep-app"),
]


class GatewayTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.publish = root / "GrokBotContext"
        self.publish.mkdir()
        self.write(RECORDS)
        self.log = root / "access.jsonl"
        self.tokens = G.Tokens(root / "tokens.json")
        self.token = self.tokens.create("grok")
        self.server = G.make_server("127.0.0.1", 0, self.publish, self.tokens, self.log, vault=root / "vault")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def write(self, records):
        (self.publish / "memories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    def post(self, body, token=None, headers=None):
        request = urllib.request.Request(self.url, json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            **({"Authorization": f"Bearer {token or self.token}"} if token != "" else {}), **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            return error.code, None

    def call(self, tool, **args):
        status, body = self.post({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                  "params": {"name": tool, "arguments": args}})
        self.assertEqual(status, 200)
        return body["result"]

    def test_handshake_and_tool_list(self):
        status, body = self.post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
        self.assertEqual((status, body["result"]["protocolVersion"]), (200, "2025-06-18"))
        self.assertEqual(self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})[0], 202)
        names = [t["name"] for t in self.post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})[1]["result"]["tools"]]
        self.assertEqual(names, ["search_memory", "get_memory", "get_project_context"])

    def test_auth_required_and_revocation_immediate(self):
        self.assertEqual(self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="")[0], 401)
        self.assertEqual(self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="pmem_wrong")[0], 401)
        self.assertEqual(self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"})[0], 200)
        bare = self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="",
                         headers={"Authorization": self.token})
        self.assertEqual(bare[0], 200)
        self.tokens.revoke("grok")
        self.assertEqual(self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"})[0], 401)

    def test_bad_origin_rejected(self):
        status, _ = self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_search_ranks_filters_and_caps(self):
        result = self.call("search_memory", query="postgres database meal")
        self.assertEqual(result["structuredContent"]["results"][0]["id"], "mem-1")
        self.assertNotIn("approved_hash", result["structuredContent"]["results"][0])
        filtered = self.call("search_memory", query="price sleep postgres", project="Sleep App")
        self.assertEqual([r["id"] for r in filtered["structuredContent"]["results"]], ["mem-3"])
        self.assertLessEqual(len(self.call("search_memory", query="the user", limit=999)["structuredContent"]["results"]), 20)

    def test_get_memory_and_project_context(self):
        self.assertEqual(self.call("get_memory", memory_id="mem-2")["structuredContent"]["title"], "Eval lab")
        self.assertTrue(self.call("get_memory", memory_id="mem-404")["isError"])
        context = self.call("get_project_context", project_id="northwind")["structuredContent"]
        self.assertEqual(context["memory_count"], 2)
        self.assertEqual(set(context["by_category"]), {"technical_decision", "project_context"})
        unknown = self.call("get_project_context", project_id="nope")
        self.assertTrue(unknown["isError"])
        self.assertIn("sleep-app", json.dumps(unknown["structuredContent"]["available_projects"]))

    def test_unpublished_memory_disappears_immediately(self):
        self.assertFalse(self.call("get_memory", memory_id="mem-1")["isError"])
        self.write(RECORDS[1:])
        self.assertTrue(self.call("get_memory", memory_id="mem-1")["isError"])
        self.assertEqual(self.call("search_memory", query="postgres")["structuredContent"]["results"], [])

    def test_access_log(self):
        self.post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "grok", "version": "1"}}})
        self.call("search_memory", query="postgres")
        self.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="pmem_wrong")
        entries = [json.loads(l) for l in self.log.read_text().splitlines()]
        self.assertEqual((entries[0]["method"], entries[0]["client_info"]["name"]), ("initialize", "grok"))
        self.assertEqual((entries[1]["token"], entries[1]["tool"], entries[1]["returned"]), ("grok", "search_memory", ["mem-1"]))
        self.assertEqual(entries[2]["error"], "unauthorized")

    def test_get_is_not_allowed_and_unknown_method_errors(self):
        request = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {self.token}"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 405)
        self.assertEqual(self.post({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})[1]["error"]["code"], -32601)

    def test_refuses_publish_dir_inside_vault(self):
        with self.assertRaises(SystemExit):
            G.make_server("127.0.0.1", 0, self.publish, self.tokens, self.log, vault=self.publish.parent)


if __name__ == "__main__":
    unittest.main()
