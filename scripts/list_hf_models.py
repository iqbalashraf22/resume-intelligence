import json
import os
import sys
import urllib.request

from dotenv import load_dotenv

load_dotenv()
req = urllib.request.Request(
    "https://router.huggingface.co/v1/models",
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
)
data = json.load(urllib.request.urlopen(req))
ids = sorted(m["id"] for m in data.get("data", []))

keyword = sys.argv[1].lower() if len(sys.argv) > 1 else ""
for model_id in ids:
    if keyword in model_id.lower():
        print(model_id)
print(f"\n{len(ids)} models total")