import json, urllib.request

# Import the actual system prompt from llm_extractor
import sys; sys.path.insert(0, ".")
from llm_extractor import _SYSTEM_PROMPT as sys_prompt

turns = [
    "John: I visited Tokyo in March 2022 and loved the food.",
    "Alice: Yeah, totally.",
    "Sam: I don't own a car — I take the subway everywhere.",
]

for turn in turns:
    payload = {
        "model": "qwen2.5:1.5b",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": 42, "num_predict": 256},
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": turn},
        ],
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
        c = body["message"]["content"]
        print(f"Turn  : {turn!r}")
        print(f"Raw   : {c[:400]}")
        try:
            d = json.loads(c)
            print(f"Parsed: {d}")
        except Exception as e:
            print(f"Parse error: {e}")
        print()
