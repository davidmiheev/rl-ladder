"""Save and run a Kaggle script notebook through Kaggle's MCP endpoint (token from env).

    python scripts/kaggle/kpost.py <slug> <notebook.py> [gpu]     # "gpu": two T4s
"""
import json, os, ssl, sys, urllib.request
slug, path = sys.argv[1], sys.argv[2]
TOKEN = os.environ["KAGGLE_API_TOKEN"].strip()
ctx = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt" if os.path.exists("/root/.ccr/ca-bundle.crt") else None)  # Claude cloud proxy CA
opener = urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
H = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": f"Bearer {TOKEN}"}
def rpc(method, params, id_=1):
    body = json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params}).encode()
    data = opener.open(urllib.request.Request("https://www.kaggle.com/mcp", body, H), timeout=180).read().decode()
    return json.loads([l for l in data.splitlines() if l.startswith("data:")][0][5:])
rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "c10", "version": "0"}})
req = {"slugNullable": f"davidmiheev/{slug}", "newTitleNullable": slug, "textNullable": open(path).read(),
       "kernelTypeNullable": "script", "languageNullable": "python", "isPrivateNullable": True,
       "enableInternetNullable": True, "enableGpuNullable": len(sys.argv) > 3 and sys.argv[3] == "gpu", "kernelExecutionTypeNullable": "SaveAndRunAll",
       **({"machineShapeNullable": "NvidiaTeslaT4"} if len(sys.argv) > 3 and sys.argv[3] == "gpu" else {})}
res = rpc("tools/call", {"name": "save_notebook", "arguments": {"request": req}}, 2)
print(res["result"]["content"][0]["text"].replace("\n", " ")[:200])
