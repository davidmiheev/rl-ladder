"""Cancel a running Kaggle notebook session through Kaggle's MCP endpoint (token from env)."""
import json, os, ssl, sys, urllib.request
TOKEN = os.environ["KAGGLE_API_TOKEN"].strip()
ctx = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt" if os.path.exists("/root/.ccr/ca-bundle.crt") else None)  # Claude cloud proxy CA
op = urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
H = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": f"Bearer {TOKEN}"}
def rpc(m, p, i=1):
    b = json.dumps({"jsonrpc": "2.0", "id": i, "method": m, "params": p}).encode()
    d = op.open(urllib.request.Request("https://www.kaggle.com/mcp", b, H), timeout=120).read().decode()
    return json.loads([l for l in d.splitlines() if l.startswith("data:")][0][5:])
rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "c13", "version": "0"}})
tools = {t["name"]: t for t in rpc("tools/list", {}, 2)["result"]["tools"]}
if sys.argv[1] == "--schema":
    print(json.dumps(tools["cancel_notebook_session"]["inputSchema"])[:800]); sys.exit()
res = rpc("tools/call", {"name": "cancel_notebook_session", "arguments": json.loads(sys.argv[1])}, 3)
print(json.dumps(res)[:400])
