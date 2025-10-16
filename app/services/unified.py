# app/services/unified.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, List

from app.services.cache_service import try_cache, write_through_if_needed
from app.services.orchestrator_service import try_graph


# ==============================
# Helpers & debug
# ==============================
def _dbg(*args):
    """Set UNIFIED_DBG=1 to see debug prints."""
    if os.getenv("UNIFIED_DBG", "0") == "1":
        try:
            print("[UNIFIED]", *args)
        except Exception:
            pass


def _answer_policy_safe(question: str) -> Dict[str, Any]:
    """Call policy RAG safely; never raise."""
    try:
        from app.agents.policy_web.policy_agent import answer_policy
        return answer_policy(question) or {}
    except Exception:
        return {}


def _coalesce_final_answer(graph_out: Dict[str, Any]) -> str:
    """Pick the best final text from graph output."""
    for key in ("final_answer", "analysis_text", "web_answer"):
        val = (graph_out.get(key) or "").strip()
        if val:
            return val
    return "Sorgudan veri veya politika cevabı üretilemedi."


def _coalesce_citations(graph_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    cits = graph_out.get("citations")
    if not cits:
        cits = graph_out.get("policy_citations", [])
    return cits or []


def _coalesce_sql(graph_out: Dict[str, Any]) -> str:
    return (
        (graph_out.get("final_sql") or "")
        or (graph_out.get("executed_sql") or "")
        or (graph_out.get("final_query") or "")
        or (graph_out.get("sql_query") or "")
        or (graph_out.get("sql") or "")   # <<< eklendi
        or ""
    )


def _coalesce_rows(graph_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (graph_out.get("rows_preview") or graph_out.get("rows") or []) or []


def _coalesce_columns(graph_out: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    if graph_out.get("columns"):
        return list(graph_out["columns"])
    if rows and isinstance(rows, list) and isinstance(rows[0], dict):
        return list(rows[0].keys())
    return []


def _looks_like_policy(q: str) -> bool:
    """Heuristic to detect policy-style questions (very recall-heavy)."""
    POLICY_HINTS = [
        r"\bpolitika\w*\b", r"\bpolicy\b", r"\bkoşul\w*\b", r"\bsart\w*\b", r"\bşart\w*\b",
        r"\biade\w*\b", r"\biptal\w*\b", r"\bücret\w*\b", r"\brefund\w*\b", r"\bfare\w*\b",
        r"\bkur(?:a|u)l\w*\b",
        r"\bbagaj\w*\b", r"\bbaggage\b", r"\bfazla\s*bagaj\w*\b", r"\bel\s*bagaj\w*\b",
        r"\bkupon\w*\b", r"\buçuş\w*\s*kupon\w*\b", r"\bsıra\w*\b", r"\bkullan\w*\s*kupon\w*\b",
        r"\bstopover\w*\b", r"\bduraklama\w*\b",
        r"\bcodeshare\w*\b", r"\bkod\s*paylaş\w*\b",
        r"\bcheck[- ]?in\b",
    ]
    t = q or ""
    return any(re.search(p, t, flags=re.IGNORECASE | re.UNICODE) for p in POLICY_HINTS)


# ==============================
# Public entrypoint
# ==============================
async def answer_unified(
    question: str,
    preview_rows: int,
    return_rows: bool,
    return_chart: bool,
    top_k: int,
    threshold: float = 0.85,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified entry:
    1) Try cache (fast SQL-only answers)
    2) Run LangGraph (policy + SQL orchestrator)
    3) Strong packaging with safe fallbacks (policy-first for policy-ish questions)
    """

    # ---------- 1) CACHE FAST PATH ----------
    cache_out = await try_cache(
        question, top_k, threshold, preview_rows, return_rows, return_chart, session_id=session_id
    )
    if cache_out:
        cache_out["source"] = "cache"
        cache_out.setdefault("used_cache", True)
        # ensure final_answer is present
        cache_out["final_answer"] = (
            cache_out.get("final_answer") or cache_out.get("analysis_text") or ""
        ).strip()
        return cache_out

    # ---------- 2) GRAPH PATH ----------
    try:
        graph_out = await try_graph(
            question,
            preview_rows,
            return_chart,
            return_rows,
            session_id=session_id,
        )
        graph_out["source"] = "graph"
        graph_out.setdefault("used_cache", False)

        final_answer = _coalesce_final_answer(graph_out)
        rows = _coalesce_rows(graph_out) if return_rows else []
        columns = _coalesce_columns(graph_out, rows)
        citations = _coalesce_citations(graph_out)
        sql_text = _coalesce_sql(graph_out)

        # ---------- 2.a) Last-ditch policy patch ----------
        # If it *looks like* a policy question, but graph decided use_web=False (or produced an empty/weak policy),
        # call policy RAG directly and prefer its answer.
        looks_policy = _looks_like_policy(question)
        use_web_from_graph = bool(graph_out.get("use_web", False))
        policy_missing = (
            (not final_answer)
            or final_answer.strip().lower().startswith("sorgudan veri")
            or not citations
        )
        if looks_policy and (not use_web_from_graph or policy_missing):
            pol = _answer_policy_safe(question)
            web = (pol.get("answer") or "").strip()
            cits = pol.get("citations") or []
            if web:
                final_answer = web
                citations = cits

        payload = {
            "used_cache": False,
            "use_web": bool(use_web_from_graph or looks_policy),
            "want_sql": bool(graph_out.get("want_sql", False)),
            "final_answer": final_answer or "Sorgudan veri veya politika cevabı üretilemedi.",
            "analysis_text": (graph_out.get("analysis_text") or "").strip() or final_answer,
            "headline_metrics": graph_out.get("headline_metrics") or [],
            "vega_lite_spec": graph_out.get("vega_lite_spec"),
            "sql": sql_text,
            "final_sql": graph_out.get("final_sql"),
            "executed_sql": graph_out.get("executed_sql"),
            "final_query": graph_out.get("final_query"),
            "sql_query": graph_out.get("sql_query"),
            "sql_status": graph_out.get("sql_status"),
            "policy_status": graph_out.get("policy_status"),
            "rows": rows,
            "columns": columns,
            "citations": citations,
            "source": "graph",
        }

        _dbg("OUT", {
            "use_web": payload["use_web"],
            "want_sql": payload["want_sql"],
            "final_len": len(payload["final_answer"] or ""),
            "cit#": len(payload["citations"] or []),
            "rows#": len(payload["rows"] or []),
        })

        # ---------- 3) WRITE-THROUGH (optional) ----------
        try:
            await write_through_if_needed(
                question,
                {"sql": payload["sql"], "rows": payload["rows"], "dialect": "sqlite"},
                session_id=session_id,
            )
        except Exception:
            pass

        return payload

    except Exception as e:
        # ---------- 2.b) GRAPH FAILED → POLICY-ONLY BACKSTOP ----------
        _dbg("GRAPH_ERROR", str(e))
        pol = _answer_policy_safe(question)
        web = (pol.get("answer") or "").strip()
        cits = pol.get("citations") or []
        return {
            "used_cache": False,
            "use_web": True if web else False,
            "want_sql": False,
            "final_answer": web or "Sorgudan veri veya politika cevabı üretilemedi.",
            "analysis_text": web or "",
            "headline_metrics": [],
            "vega_lite_spec": None,
            "sql": "",
            "rows": [] if return_rows else [],
            "columns": [],
            "citations": cits,
            "source": "graph-fallback-policy",
        }
