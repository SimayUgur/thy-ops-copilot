#!/usr/bin/env bash
set -e

API=${API:-http://127.0.0.1:8000}

echo "[1/3] Health check"
curl -fsS "$API/healthz" && echo -e "\nOK\n"

echo "[2/3] Cache/graph basic ask"
curl -fsS -X POST "$API/ask_unified" \
  -H "Content-Type: application/json" \
  -d '{"question":"Son 7 günde ortalama gecikme nedir?","preview_rows":50,"return_rows":true,"return_chart":false,"top_k":8}' \
  | python - <<'PY'
import sys, json
d=json.load(sys.stdin)
print("used_cache:", d.get("used_cache"), "source:", d.get("source"))
PY

echo "[3/3] Policy prompt"
curl -fsS -X POST "$API/ask_unified" \
  -H "Content-Type: application/json" \
  -d '{"question":"Fazla bagaj politikası nedir?","preview_rows":0,"return_rows":false,"return_chart":false,"top_k":8}' \
  | python - <<'PY'
import sys, json
d=json.load(sys.stdin)
print("len(answer)=", len((d.get("final_answer") or "")))
PY
