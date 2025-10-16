# app/agents/sql/sql_langraph.py
from __future__ import annotations

import ast
import os
import re
import json
from typing import TypedDict, List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# i18n normalize (tek giriş noktası)
from app.i18n.locale import map_terms_to_schema

# Router
from app.agents.sql.router import route_agents

# Tek tablo sub-agent graph (subquestion + column selector)
from app.agents.sql.customer_agent import graph_final

# THY-özel OpenAI zincirleri (NL→SQL)
from app.agents.sql.sql_agent_thy import (
    chain_filter_extractor,
    chain_range_date_extractor,
    chain_query_extractor,
    chain_query_validator,
)

# Birleşik sentezleyici (SQL-only / Policy-only / Hybrid)
from app.agents.synthesis.synthesizer import synthesize_unified

# Fuzzy (DB DISTINCT’e map) + normalize edilmiş yes-list
from app.agents.sql.fuzzy_wuzzy import call_match, normalize_filters

# Web/Policy router (LLM-first, timeout→regex fallback)
from app.agents.dispatcher import classify_intent

# Web/Policy agent (RAG + Wikipedia fallback)
from app.agents.policy_web.policy_agent import answer_policy

# Yalnızca ham SQL’i ayıklamak için
from app.utils.text import extract_sql_only


# ---------------- config / db ----------------
DB_URL = os.getenv("DB_URL", "sqlite:///app/data/thy_ops.db")
engine = create_engine(DB_URL, future=True)

checkpointer = MemorySaver()
USE_CKPT = os.getenv("LANGGRAPH_DISABLE_CHECKPOINT", "0") != "1"

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_llm_split = ChatOpenAI(model=_OPENAI_MODEL, temperature=0)


# ---------------- helpers ----------------
def _dedup_column_extract(outputs: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """
    flights_out / complaints_out / refunds_out / weather_out -> column_extract dedup
    """
    seen = set()
    final: List[List[str]] = []
    for v in outputs.values():
        if not v:
            continue
        for item in v.get("column_extract", []):
            key = tuple(item)
            if key not in seen:
                seen.add(key)
                final.append(item)
    return final


def _is_no_filter(val) -> bool:
    """["no"] / [] / hatalı yapı → True"""
    return (
        not isinstance(val, list)
        or len(val) == 0
        or (len(val) == 1 and str(val[0]).lower() == "no")
    )


# ---------------- state ----------------
class FinalState(TypedDict, total=False):
    # raw & normalized
    raw_query: str              # UI’dan gelen orijinal soru
    user_query: str             # normalize edilmiş soru
    term_map: Dict[str, str]    # {"rötar":"delay", ...} audit için

    # WEB/POLICY routing meta
    use_web: bool
    want_sql: bool
    time_window_days: int
    routing_reason: str
    policy_score: float
    sql_score: float

    # web/policy answer
    web_answer: str
    policy_citations: List[Dict[str, str]]

    # router (SQL domain içi)
    router_out: List[str]

    # single-table agents output
    flights_out: Dict[str, Any]
    complaints_out: Dict[str, Any]
    refunds_out: Dict[str, Any]
    weather_out: Dict[str, Any]

    # filters
    filtered_col: str           # str(list-of-lists)
    filter_extractor: list      # ["yes", ["table","col","values"], ...] | ["no"]
    fuzz_match: list            # [["table name:..","column_name:..","filter_value:.."], ...]
    fuzz_norm: list             # ["yes", ["table","col","v1, v2"], ...]
    range_filters: list         # ["ranges", ...] | ["dates", ...] | ["none"]

    # sql + exec + synthesis
    sql_query: str
    final_query: str
    preview_rows: int
    rows: List[Dict[str, Any]]
    columns: List[str]

    # ⬇️ Synth çıktılarını state'te tut
    final_answer: str
    final_sql: str
    rows_preview: List[Dict[str, Any]]
    citations: List[Dict[str, str]]

    analysis_text: str
    headline_metrics: List[Dict[str, Any]]
    vega_lite_spec: Dict[str, Any]
    want_analysis: bool
    want_chart: bool

    # hybrid split
    policy_query_raw: str
    sql_query_text: str

    # join/barrier kontrol
    policy_done: bool
    sql_done: bool
    arrived: List[str]
    policy_status: str  # "ok" | "not_found" | "error"
    sql_status: str     # "ok" | "no_table" | "no_rows" | "error"


# ---------------- nodes ----------------
def normalize_node(state: FinalState):
    raw = state.get("raw_query", "") or state.get("user_query", "")
    q_norm, applied = map_terms_to_schema(raw)
    return {"user_query": q_norm, "raw_query": raw, "term_map": applied}


def policy_gate(state: FinalState):
    q = state["user_query"]
    res = classify_intent(q)

    POLICY_KWS = [
        r"\bpolitika\w*\b", r"\bpolicy\b", r"\bkoşul\w*\b", r"\bsart\w*\b", r"\bşart\w*\b",
        r"\biade\w*\b", r"\biptal\w*\b", r"\bücret\w*\b", r"\bfare\w*\b", r"\brefund\w*\b",
        r"\bkur(?:a|u)l\w*\b",
        r"\bbagaj\w*\b", r"\bbaggage\b", r"\bfazla\s*bagaj\w*\b", r"\bel\s*bagaj\w*\b",
        r"\bkupon\w*\b", r"\buçuş\w*\s*kupon\w*\b", r"\bsıra\w*\b", r"\bkullan\w*\s*kupon\w*\b",
        r"\bstopover\w*\b", r"\bduraklama\w*\b",
        r"\bcodeshare\w*\b", r"\bkod\s*paylaş\w*\b",
        r"\bcheck[- ]?in\b"
    ]
    SQL_KWS = [
        r"\bortalama\w*\b", r"\bavg\b", r"\btoplam\b", r"\bsum\b",
        r"\boran\w*\b", r"\btrend\w*\b", r"\bgecik\w*\b|\brötar\b|\bdelay\b",
        r"\bsay[ıi]s[ıi]\b", r"\ben\schok|en çok\b", r"\bmax\b", r"\bmin\b"
    ]

    p_hit = any(re.search(p, q, flags=re.IGNORECASE | re.UNICODE) for p in POLICY_KWS)
    s_hit = any(re.search(p, q, flags=re.IGNORECASE | re.UNICODE) for p in SQL_KWS)

    # LLM kararı + keyword sinyali
    use_web  = bool(res.get("want_policy") or p_hit)
    want_sql = bool(res.get("want_sql")   or s_hit)

    # # Saf politika → SQL'i kapat (mevcut davranış değişmesin)
    # policy_only = p_hit and not s_hit
    # if policy_only:
    #     want_sql = False

    if p_hit and not s_hit:
        want_sql = False
        use_web  = True

    # ✅ Sert kural 2: Saf SQL ise policy'yi kapat (kırılan davranışı geri getirir)
    if s_hit and not p_hit:
        want_sql = True
        use_web  = False
        state["force_sql"] = True  # audit/debug için

    return {
        "use_web": use_web,
        "want_sql": want_sql,
        "time_window_days": res.get("window_days", 60),
        "routing_reason": res.get("reason", ""),
        "policy_score": float(res.get("policy_score", 0.0)) + (0.2 if p_hit else 0.0),
        "sql_score": float(res.get("sql_score", 0.0)) + (0.2 if s_hit else 0.0),
    }


def policy_condition(state: FinalState):
    use_web = bool(state.get("use_web"))
    want_sql = bool(state.get("want_sql"))
    if use_web and want_sql:
        return "hybrid"
    if use_web:
        return "web"
    return "sql"


# -------- HYBRID SPLIT: cümleleri policy/sql olarak ayır --------
_split_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Kullanıcının Türkçe sorusunu kısa parçalara böl ve her parçayı 'policy' ya da 'sql' olarak etiketle. "
     "Sadece JSON döndür. Şema alanları uydurma. "
     'JSON formatı: {"policy_parts": ["..."], "sql_parts": ["..."]}'),
    ("human", "{question}")
])
_split_chain = _split_prompt | _llm_split | StrOutputParser()

def hybrid_split_node(state: FinalState):
    q = state.get("user_query") or state.get("raw_query") or ""
    try:
        out = _split_chain.invoke({"question": q}).strip()
        data = json.loads(out)
        pol_parts = [p.strip() for p in (data.get("policy_parts") or []) if p.strip()]
        sql_parts = [p.strip() for p in (data.get("sql_parts") or []) if p.strip()]
    except Exception:
        pol_parts, sql_parts = [], []

    # Heuristik takviye (LLM kaçırırsa)
    if not pol_parts and re.search(r"(policy|politika|iade|iptal|bagaj|refund|kural)", q, re.I):
        pol_parts = [q]
    if not sql_parts and re.search(r"(ortalama|toplam|oran|trend|gecik|rötar|count|avg|sum|max|min)", q, re.I):
        sql_parts = [q]

    return {
        "policy_query_raw": " ve ".join(pol_parts) if pol_parts else "",
        "sql_query_text":   " ve ".join(sql_parts) if sql_parts else "",
        "hybrid_split_done": True,
    }


# -------- POLICY PATH --------
def policy_rewrite(state: FinalState):
    """
    Hibrit/policy isteklerde, policy tarafı için soruyu temizle.
    """
    if not state.get("use_web"):
        return {}

    q = (state.get("policy_query_raw") or state.get("user_query") or "").strip()
    if not q:
        return {}

    # Parçalama (bağlaçlar)
    parts = re.split(r"\b(?:ve|ile|,|;|/|&)\b", q, flags=re.IGNORECASE)
    POLICY_KWS = r"(politika|policy|iade|ücret|kural|kur(a|u)l|bagaj|baggage|refund)"
    policy_parts = [p.strip() for p in parts if re.search(POLICY_KWS, p, re.IGNORECASE)]
    cand = policy_parts[0] if policy_parts else q

    # Zaman & analitik metinleri çıkar
    cand = re.sub(r"\bson\s+\d+\s+(gün|hafta|ay|yıl)\b", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\bge(çen|çtiğimiz)\s+(gün|hafta|ay|yıl)\b", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\b\d+\s*(dk|dakika|saat|gün)\b", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\b(ortalama|avg|toplam|sum|oran|trend|gecikme|rötar|delay|sayısı?)\b", "", cand, flags=re.IGNORECASE)

    cand = re.sub(r"\s+", " ", cand).strip()
    if cand and not cand.endswith("?"):
        cand += "?"

    return {"policy_query": cand}


def web_policy_node(state: FinalState):
    q = state.get("policy_query") or state["user_query"]
    out = answer_policy(q)
    web = out.get("answer", "") or ""
    cits = out.get("citations", []) or []

    status = "ok" if web.strip() else "not_found"

    base = {
        "web_answer": web,
        "policy_citations": cits,
        "citations": cits,       # synth’e de taşı
        "policy_done": True,
        "policy_status": status,
    }

    # POLICY-ONLY ise final'ı şimdiden doldur
    if state.get("use_web") and not state.get("want_sql"):
        base["final_answer"]  = web or "Politika tarafında uygun doküman bulunamadı."
        base["analysis_text"] = base["final_answer"]

    return base


# ------ SQL PATH ------
def router(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    return {"router_out": route_agents(q)}


def route_request(state: FinalState):
    valid = {"flights", "complaints", "refunds", "weather"}
    targets = [r for r in state.get("router_out", []) if r in valid]
    # Eğer hiç hedef yoksa işaretle
    if not targets:
        state["sql_status"] = state.get("sql_status") or "no_table"
    return targets
    #return [r for r in state.get("router_out", []) if r in valid]


def flights_agent(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    sub = graph_final.invoke({"user_query": q, "table_lst": ["flights"]})
    return {"flights_out": sub}


def complaints_agent(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    sub = graph_final.invoke({"user_query": q, "table_lst": ["complaints"]})
    return {"complaints_out": sub}


def refunds_agent(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    sub = graph_final.invoke({"user_query": q, "table_lst": ["refunds"]})
    return {"refunds_out": sub}


def weather_agent(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    sub = graph_final.invoke({"user_query": q, "table_lst": ["weather"]})
    return {"weather_out": sub}


def filter_check(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    collected = {
        "flights_out": state.get("flights_out", {}),
        "complaints_out": state.get("complaints_out", {}),
        "refunds_out": state.get("refunds_out", {}),
        "weather_out": state.get("weather_out", {}),
    }
    col_details = _dedup_column_extract(collected)
    resp = chain_filter_extractor.invoke({"columns": str(col_details), "query": q})
    resp = resp.replace("```", "").replace("\n", "").strip()
    try:
        filt = ast.literal_eval(resp)
    except Exception:
        filt = ["no"]
    return {"filter_extractor": filt, "filtered_col": str(col_details)}


def filter_condition(state: FinalState):
    return "no" if _is_no_filter(state.get("filter_extractor")) else "yes"


def fuzz_filter(state: FinalState):
    val = state.get("filter_extractor", ["no"])
    triples = call_match(val)
    norm = normalize_filters(val)
    return {"fuzz_match": triples, "fuzz_norm": norm}


def query_generation(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    col_str = state.get("filtered_col", "[]")

    cat_filters = state.get("fuzz_norm", ["no"])
    if _is_no_filter(cat_filters):
        cat_filters = state.get("filter_extractor", ["no"])

    rng_raw = chain_range_date_extractor.invoke({"columns": col_str, "query": q})
    rng_raw = rng_raw.replace("```", "").replace("\n", "").strip()
    try:
        range_filters = ast.literal_eval(rng_raw)
    except Exception:
        range_filters = ["none"]

    sql_raw = chain_query_extractor.invoke({
        "columns": col_str,
        "query": q,
        "filters": cat_filters,
        "range_filters": range_filters
    })
    sql_only = extract_sql_only(sql_raw)
    return {"sql_query": sql_only, "range_filters": range_filters}


def query_validation(state: FinalState):
    q = state.get("sql_query_text") or state["user_query"]
    col_str = state.get("filtered_col", "[]")
    sql_in = state.get("sql_query", "")

    cat_filters = state.get("fuzz_norm", ["no"])
    if _is_no_filter(cat_filters):
        cat_filters = state.get("filter_extractor", ["no"])

    rng = state.get("range_filters", ["none"])

    sql_final_raw = chain_query_validator.invoke({
        "columns": col_str,
        "query": q,
        "filters": cat_filters,
        "range_filters": rng,
        "sql_query": sql_in
    })
    return {"final_query": extract_sql_only(sql_final_raw) or sql_in}


def execute_sql(state: FinalState):
    sql = state.get("final_query") or state.get("sql_query") or ""
    pr = int(state.get("preview_rows", 100) or 0)

    rows: List[Dict[str, Any]] = []
    cols: List[str] = []
    status = state.get("sql_status") or None  # önceki işaret korunabilir

    if not sql or pr <= 0:
        # SQL metni yok → tablo bulunamadı ya da üretilemedi
        return {
            "rows": rows, "columns": cols, "sql_done": True,
            "executed_sql": sql, "sql_status": status or "no_table"
        }

    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            rows = [dict(r._mapping) for r in res.fetchmany(pr)]
            cols = list(rows[0].keys()) if rows else []
            if not rows:
                cnt = conn.execute(text(f"SELECT COUNT(*) AS c FROM ({sql}) t")).scalar() or 0
            else:
                cnt = len(rows)
        # başarı: veri var/yok ayrımı
        status = "ok" if cnt > 0 else "no_rows"

    except SQLAlchemyError:
        rows, cols, cnt = [], [], 0
        status = "error"

    return {
        "rows": rows, "columns": cols, "rowcount": cnt,
        "sql_done": True, "executed_sql": sql, "sql_status": status
    }



def policy_safety(state: FinalState):
    """
    Eğer hibrit/policy sinyali kaçmışsa ama soru policy anahtarları içeriyorsa,
    tek seferlik policy_agent çalıştır.
    """
    if state.get("web_answer") or state.get("policy_done"):
        return {}
    q = state.get("user_query", "") or ""
    POLICY_KWS = [r"\bpolitika\w*\b", r"\biade\w*\b", r"\bücret\w*\b", r"\brefund\b", r"\bpolicy\b", r"\bfazla\s*bagaj\b", r"\bbagaj\b"]
    if any(re.search(p, q, flags=re.IGNORECASE) for p in POLICY_KWS):
        out = answer_policy(q)
        return {
            "web_answer": out.get("answer", ""),
            "policy_citations": out.get("citations", []),
            "citations": out.get("citations", []),
            "policy_done": True,
            "use_web": True,
        }
    return {}


# --------- JOIN / BARRIER (tek synthesizer tetiklemesi) ----------
def join_node(state: FinalState):
    arrived = set(state.get("arrived", []))
    if state.get("policy_done"):
        arrived.add("policy")
    if state.get("sql_done"):
        arrived.add("sql")
    return {"arrived": list(arrived)}


def join_condition(state: FinalState):
    hybrid = bool(state.get("use_web") and state.get("want_sql"))
    if not hybrid:
        return "go"
    arrived = set(state.get("arrived", []))
    return "go" if arrived.issuperset({"sql", "policy"}) else "wait"


def synthesize_node(state: FinalState):
    q = state["user_query"]
    sql_for_payload = (
        state.get("executed_sql")
        or state.get("final_query")
        or state.get("sql_query")
        or ""
    )
    rows = state.get("rows", []) or []
    web  = state.get("web_answer", "") or ""
    citations = state.get("policy_citations", []) or state.get("citations", []) or []
    want_chart = bool(state.get("want_chart", False))

    # POLICY-ONLY güvenli geri çağrı
    if state.get("use_web") and not state.get("want_sql") and not web.strip():
        try:
            _out = answer_policy(q)
            web = (_out.get("answer") or "").strip()
            citations = _out.get("citations", []) or []
            state["web_answer"] = web
            state["policy_citations"] = citations
            state["policy_done"] = True
            state["citations"] = citations
        except Exception:
            pass

    # POLICY-ONLY: direkt dön
    if state.get("use_web") and not state.get("want_sql"):
        final_answer = web.strip() or "Politika cevabı üretilemedi."
        return {
            "final_answer": final_answer,
            "analysis_text": final_answer,
            "headline_metrics": [],
            "vega_lite_spec": None,
            "citations": citations,
            "final_sql": "",
            "rows_preview": [],
        }

    # SQL tarafı istenmiyorsa temizle
    if not state.get("want_sql", False):
        sql_for_payload = ""
        rows = []

    # HYBRID / SQL-ONLY birleştirme
    policy_status = state.get("policy_status") or ("ok" if (state.get("web_answer") or "").strip() else "not_found")
    sql_status    = state.get("sql_status") or ("ok" if rows else ("no_rows" if (state.get("executed_sql") or state.get("final_query")) else "no_table"))
    syn = synthesize_unified(
        question=q,
        sql=sql_for_payload,
        rows=rows,
        web_answer=web,
        web_citations=citations,
        want_chart=want_chart,
        llm_merge=True,
        return_sql=True,
    )

    # Güvenli fallback zinciri
    final_answer = (syn.get("final_answer") or "").strip() \
        or (syn.get("analysis_text") or "").strip() \
        or web.strip() \
        or ("Sorgu çalıştı ve satırlar döndü; kısa özet üretilemedi." if rows else
            ("SQL hazır; önizleme satırı bulunamadı." if sql_for_payload else "Politika / SQL cevabı üretilemedi."))


     # --- Eksik taraf uyarıları (kısa, tek satır) ---
    notes = []
    if state.get("use_web") and policy_status != "ok":
        notes.append("*(Policy notu: uygun doküman bulunamadı.)*")
    if state.get("want_sql") and sql_status != "ok":
        if sql_status == "no_table":
            notes.append("*(SQL notu: ilgili tablo/kolon tespit edilemedi.)*")
        elif sql_status == "no_rows":
            notes.append("*(SQL notu: sorgu çalıştı ancak veri dönmedi.)*")
        elif sql_status == "error":
            notes.append("*(SQL notu: sorgu yürütme hatası.)*")

    if notes:
        final_answer = (final_answer + "\n\n" + "\n".join(notes)).strip()

    final_sql    = syn.get("final_sql") or sql_for_payload or ""
    rows_preview = syn.get("rows_preview") or (rows[:10] if isinstance(rows, list) else [])


    return {
        "final_answer": final_answer,
        "analysis_text": syn.get("analysis_text") or final_answer,
        "headline_metrics": syn.get("headline_metrics") or [],
        "vega_lite_spec": syn.get("vega_lite_spec"),
        "citations": syn.get("citations", []) or citations,
        "final_sql": final_sql,
        "sql": final_sql,     # <<< UI geriye dönük alan
        "rows_preview": rows_preview,
        "policy_status": policy_status,
        "sql_status": sql_status,
    }
    


def policy_only_or_join(state: FinalState):
    # Policy-only ise direkt synthesize; aksi halde join’e
    return "synthesize" if (state.get("use_web") and not state.get("want_sql")) else "join"


# ---------------- graph build ----------------
builder = StateGraph(FinalState)

# nodes
builder.add_node("normalize", normalize_node)
builder.add_node("policy_gate", policy_gate)
builder.add_node("hybrid_split", hybrid_split_node)

builder.add_node("policy_rewrite", policy_rewrite)
builder.add_node("web_policy", web_policy_node)

builder.add_node("router", router)
builder.add_node("flights", flights_agent)
builder.add_node("complaints", complaints_agent)
builder.add_node("refunds", refunds_agent)
builder.add_node("weather", weather_agent)

builder.add_node("filter_check", filter_check)
builder.add_node("fuzz_filter", fuzz_filter)
builder.add_node("query_generator", query_generation)
builder.add_node("query_validation", query_validation)
builder.add_node("execute_sql", execute_sql)
builder.add_node("policy_safety", policy_safety)

# join & synth
builder.add_node("join", join_node)
builder.add_node("synthesize", synthesize_node)

# edges
builder.add_edge(START, "normalize")
builder.add_edge("normalize", "policy_gate")

builder.add_conditional_edges(
    "policy_gate",
    policy_condition,
    # web ve hybrid: önce hybrid_split → sonra iki kol da başlasın
    {"web": "hybrid_split", "sql": "router", "hybrid": "hybrid_split"}
)

# hybrid_split'ten policy ve sql kollarını başlat
builder.add_edge("hybrid_split", "policy_rewrite")
builder.add_edge("hybrid_split", "router")

# policy kolu
builder.add_edge("policy_rewrite", "web_policy")
builder.add_edge("web_policy", "join")

# SQL path
builder.add_conditional_edges(
    "router",
    route_request,
    ["flights", "complaints", "refunds", "weather"]
)
builder.add_edge("flights", "filter_check")
builder.add_edge("complaints", "filter_check")
builder.add_edge("refunds", "filter_check")
builder.add_edge("weather", "filter_check")

builder.add_conditional_edges(
    "filter_check",
    filter_condition,
    {"no": "query_generator", "yes": "fuzz_filter"}
)
builder.add_edge("fuzz_filter", "query_generator")
builder.add_edge("query_generator", "query_validation")
builder.add_edge("query_validation", "execute_sql")

builder.add_edge("execute_sql", "policy_safety")
builder.add_edge("policy_safety", "join")

# Join kararı
builder.add_conditional_edges(
    "join",
    join_condition,
    {"go": "synthesize", "wait": "join"}
)

builder.add_edge("synthesize", END)

if USE_CKPT:
    graph_main = builder.compile(checkpointer=checkpointer)
else:
    graph_main = builder.compile()
