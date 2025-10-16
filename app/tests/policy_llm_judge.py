# app/tests/policy_llm_judge.py
from __future__ import annotations
import sys, pathlib
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # .../thy_agentic_sql
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os, json, argparse, sys, re
from typing import Dict, Any, List

# LLM
from langchain_openai import ChatOpenAI

# Değerlendirilecek agent
from app.agents.policy_web.policy_agent import answer_policy, rebuild_policy_index

RUBRIC: Dict[str, Any] = {
    "dimensions": {
        # Kaynakla tutarlılık ve uydurma yokluğu
        "faithfulness": {"weight": 0.40, "desc": "Cevap kaynaklarla tutarlı, uydurma yok"},
        # Sorunun çekirdeğini ve önemli istisnaları kapsama
        "coverage":     {"weight": 0.30, "desc": "Kritik noktaları ve istisnaları kapsıyor"},
        # Hedef biçim: paragraf + alıntılar (blockquote) + tekil Kaynakça
        "formatting":   {"weight": 0.20, "desc": "Paragraf + blockquote + Kaynaklar akışı"},
        # Yeterli ve tekilleştirilmiş atıf
        "citations":    {"weight": 0.10, "desc": "Yeterli, tekil, ilgili kaynakça"},
    },
    "thresholds": {"pass": 0.75}
}

SYSTEM = (
    "You are a strict evaluator for airline policy Q&A. "
    "Judge ONLY the provided answer text against the rubric. "
    "Penalize hallucinations or irrelevant quotes. "
    "Prefer answers that present a detailed paragraph first, then quotes (as blockquotes), then a single 'Kaynaklar' list."
)

PROMPT = """Question:
{q}

Answer:
{a}

Rubric (JSON):
{rubric}

Score each dimension in 0..1 and compute a weighted 'overall' per weights in rubric.
Return STRICT JSON (no backticks) like:
{{
  "faithfulness": 0.0,
  "coverage": 0.0,
  "formatting": 0.0,
  "citations": 0.0,
  "overall": 0.0,
  "notes": "one short paragraph of feedback"
}}
"""

# Varsayılan soru seti (senin son testindeki liste)
DEFAULT_QUESTIONS = [
    "Uçuş kuponları sırasıyla kullanılmak zorunda mı? Sıra bozulursa bilet ne olur?",
    "Uygulanabilir ücret nasıl belirlenir? İlk uçuş kuponunun tarihine göre mi? Aradaki fark kime aittir?",
    "Bilet ücreti neleri kapsamaz? Vergi ve harçları kim öder?",
    "Biletleme süresi içinde ödeme yapılmazsa taşıyıcı ne yapabilir?",
    "Rezervasyonumu kullanmazsam takip eden bacak rezervasyonlarım ne olur? Tazminat var mı?",
    "Ücretli koltuk seçimi garanti midir?",
    "Taşıma reddedilirse tazminat ve iade ne olur?",
]

def _has_env() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env değişkeni yok. Lütfen ayarla ve tekrar dene.", file=sys.stderr)
        sys.exit(2)

def judge_one(llm: ChatOpenAI, q: str, a: str) -> Dict[str, Any]:
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(q=q, a=a, rubric=json.dumps(RUBRIC, ensure_ascii=False))},
    ]
    resp = llm.invoke(msg)
    raw = (resp.content or "").strip()

    # Sıkı JSON parse
    try:
        data = json.loads(raw)
    except Exception:
        # JSON gömülü ise yakalamaya çalış
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))

    # Ağırlıklı overall yoksa hesapla
    if "overall" not in data:
        w = RUBRIC["dimensions"]
        overall = (
            data.get("faithfulness", 0) * w["faithfulness"]["weight"] +
            data.get("coverage", 0)     * w["coverage"]["weight"] +
            data.get("formatting", 0)   * w["formatting"]["weight"] +
            data.get("citations", 0)    * w["citations"]["weight"]
        )
        data["overall"] = round(float(overall), 4)
    return data

def run(questions: List[str], model: str, temperature: float, dump_answers: bool) -> Dict[str, Any]:
    # İndeksi garantiye almak için
    rebuild_policy_index()

    llm = ChatOpenAI(model=model, temperature=temperature)
    out: List[Dict[str, Any]] = []

    for q in questions:
        a = answer_policy(q)["answer"]
        score = judge_one(llm, q, a)
        out.append({
            "question": q,
            "answer": a if dump_answers else "(omitted)",
            "score": score
        })

    # Özet
    overall_scores = [x["score"]["overall"] for x in out]
    avg_overall = round(sum(overall_scores) / max(1, len(overall_scores)), 4)
    passed = sum(1 for x in overall_scores if x >= RUBRIC["thresholds"]["pass"])
    summary = {"avg_overall": avg_overall, "pass": passed, "total": len(out)}

    return {"summary": summary, "results": out, "rubric": RUBRIC, "model": model}

def main():
    _has_env()
    parser = argparse.ArgumentParser(description="LLM as judge for policy_web answers")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
                        help="Judge modeli (vars: gpt-4o-mini)")
    parser.add_argument("--temp", type=float, default=0.0, help="Judge sıcaklığı (vars: 0.0)")
    parser.add_argument("--questions", type=str, default="",
                        help="Virgülle ayrılmış özel sorular; boş ise default kullanılır")
    parser.add_argument("--dump-answers", action="store_true",
                        help="JSON çıktı içinde tam cevap metinlerini de göster")
    parser.add_argument("--out", type=str, default="",
                        help="JSON çıktıyı bir dosyaya yaz (örn. tests/policy_llm_judge.json)")
    args = parser.parse_args()

    qs = [q.strip() for q in args.questions.split(",") if q.strip()] if args.questions else DEFAULT_QUESTIONS
    report = run(qs, model=args.model, temperature=args.temp, dump_answers=args.dump_answers)

    js = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"Wrote: {args.out}")
    else:
        print(js)

if __name__ == "__main__":
    main()
