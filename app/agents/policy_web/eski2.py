# THY Policy QA (pro, rerank + structured + versioned) — sentence-level evidence
# • Offline-first RAG: CSV + PDF → FAISS (MMR bulk) → Cross-Encoder Rerank (Cohere/BGE) → top-k
# • Wikipedia yalnızca FALLBACK (Tavily Extract, tek URL whitelist, circuit breaker)
# • Gelişmiş chunking (tiktoken + heading-aware + overlap + tiny-merge)
# • Structured Output: Madde madde + detay paragraf + Son güncelleme
# • Alıntı Kalitesi: Cümle-düzeyi kanıt seçimi + (syf N) + regex boost
# • Kaynaklar: SADECE kullanılan kaynaklar; özgün [idx] korunur (mapping bozmaz)
# • Index versiyonlama: POLICY_INDEX_DIR/vYYYYMMDD, son 2 versiyon sakla
# • Retry/Timeout: tenacity ile; LLM ve Tavily çağrılarında
# ---------------------------------------------------------------------------

from __future__ import annotations

import os, re, asyncio, datetime, hashlib, glob, shutil, math
from typing import List, Dict, Any, Tuple, Optional
import textwrap
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# LLM + embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store + loaders
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Splitters
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import (  # type: ignore
        MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    )

# Rerank (Cohere Rerank v3.5) – varsa kullan
try:
    from langchain_cohere import CohereRerank  # pip install -U langchain-cohere
    _HAS_COHERE = True
except Exception:
    _HAS_COHERE = False

# Alternatif Rerank (BGE Reranker)
try:
    # pip install FlagEmbedding
    from FlagEmbedding import FlagReranker  # type: ignore
    _HAS_BGE = True
except Exception:
    _HAS_BGE = False

# Optional Tavily
try:
    from langchain_tavily import TavilyExtract
    _HAS_TAVILY = True
except Exception:
    _HAS_TAVILY = False

# Retry / circuit breaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# =========================== .env & constants ===========================
env_file = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=env_file, override=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

POLICY_CSV = os.getenv("POLICY_CSV", "app/data/thy_qa_policy.csv")
POLICY_PDFS = [p.strip() for p in os.getenv("POLICY_PDFS", "").split(",") if p.strip()]
POLICY_INDEX_DIR = os.getenv("POLICY_INDEX_DIR", "app/data/thy_policy_index")

RETRIEVE_K = int(os.getenv("POLICY_RETRIEVE_K", "3"))   # final k
FETCH_K = max(RETRIEVE_K * 7, 20)                        # ilk çekiş (MMR toplu)
RERANK_TOP = max(RETRIEVE_K * 3, 12)                     # rerank sonrası kırp

CHUNK_TOKENS = int(os.getenv("POLICY_CHUNK_TOKENS", "700"))
CHUNK_OVERLAP = int(os.getenv("POLICY_CHUNK_OVERLAP", "150"))

WIKI_URL = os.getenv("WIKI_URL", "https://w.wiki/F3kw")
THY_INFO_URL = "https://www.turkishairlines.com/tr-int/bilgi-edin"

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
BGE_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Circuit breaker – wiki
_CB_FAILS = 0
_CB_OPEN = False
_CB_THRESHOLD = 3
_CB_RESET_AFTER_OK = 1


# =========================== helpers ===========================
def _today_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def _normalize_source_url(src: Optional[str]) -> str:
    if not src:
        return ""
    s = str(src).strip()
    if s.startswith(("http", "csv:", "file://")):
        return s
    if s.endswith(".pdf") or s.startswith(("/", "./")) or os.path.exists(s):
        return f"file://{os.path.abspath(s)}"
    return s

def _hash_text(s: str, n: int = 512) -> str:
    return hashlib.sha256((s or "")[:n].encode("utf-8", "ignore")).hexdigest()

def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("chunk_id"), _hash_text(d.page_content))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def _seems_incomplete_sentence(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if t.endswith(("...", "…")):
        return True
    return not t.endswith((".", "!", "?", "”", "’", "\"", "»"))

def _page_from_url(u: str) -> Optional[int]:
    m = re.search(r'#page=(\d+)', u or '')
    return int(m.group(1)) if m else None

def _new_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, max_tokens=900, timeout=60)


# =========================== sentence-level evidence ===========================
_SENT_SPLIT = re.compile(r'(?<=[\.!\?])\s+')

KEY_PATTERNS = [
    r"uçuş kupon", r"kupon sırası|3\.3",
    r"stopover|duraklama|24\s*saat",
    r"codeshare|kod\s*paylaş",
    r"check-?\s*in",
    r"iade|iptal|değişiklik|refund|cancellation|change",
    r"32\s*kg|23\s*kg|bagaj|baggage",
    r"bilet\s*ücret|fare|tarife",
]
_KEY_RE = re.compile("|".join(KEY_PATTERNS), re.I)

_WORD = re.compile(r"\w+", re.I)

def _tokenize(txt: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(txt or "")]

def _idf_like(q_terms: List[str], s_terms: List[str]) -> float:
    # basit idf benzeri ağırlık: kısıtlı bağlamda ters ağırlık üretilir
    # (gerçek korpus idf'i yerine sorgu terim çeşitliliğini gözetir)
    if not q_terms or not s_terms:
        return 0.0
    qset = set(q_terms)
    inter = qset & set(s_terms)
    if not inter:
        return 0.0
    # çeşitlilik ödülü
    return len(inter) / len(qset)

def _score_sentence(q: str, s: str) -> float:
    qt = q.lower()
    st = s.lower()
    # anahtar kelime eşleşmesi
    kw = 1.0 if (_KEY_RE.search(qt) and _KEY_RE.search(st)) else 0.0
    # kelime kesişimi + idf benzeri
    q_terms = _tokenize(qt)
    s_terms = _tokenize(st)
    overlap = len(set(q_terms) & set(s_terms)) / max(1, len(set(q_terms)))
    idf_w = _idf_like(q_terms, s_terms)
    # uzunluk cezası: çok kısa/çok uzun cümleyi kırp
    L = len(st)
    len_ok = 1.0 if 40 <= L <= 350 else 0.6 if 20 <= L <= 500 else 0.2
    # “başlık/section” gibi sinyaller
    heading_boost = 0.1 if re.search(r"\b(bölüm|section|madde|3\.\d+)\b", st) else 0.0
    # BM25-lite (parametreler sabit)
    base = (2.2 * overlap + 1.5 * idf_w) * len_ok + heading_boost
    return base + 0.8 * kw

def _select_top_sentences(question: str, text: str, top_n: int = 2) -> List[str]:
    if not text:
        return []
    # kaba cümle bölümü (noktalama temelli)
    sents = _SENT_SPLIT.split(text)
    # split sonucu: [sent0, sep, sent1, sep, ...] -> düzeltilmiş liste
    fixed = []
    for i in range(0, len(sents)-1, 2):
        fixed.append((sents[i] + sents[i+1]).strip())
    if not fixed and sents:
        fixed = [sents[0].strip()]
    # skorla
    scored = [(s, _score_sentence(question, s)) for s in fixed if s]
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    seen = set()
    for s, _ in scored:
        h = _hash_text(s, n=256)
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
        if len(out) >= top_n:
            break
    return out


# =========================== Versioned index ===========================
def _version_dir(base: str) -> str:
    return os.path.join(base, f"v{datetime.datetime.utcnow():%Y%m%d}")

def _list_versions(base: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(base, "v*")) if os.path.isdir(p)])

def _prune_versions(base: str, keep: int = 2) -> None:
    vers = _list_versions(base)
    if len(vers) <= keep:
        return
    for p in vers[:-keep]:
        shutil.rmtree(p, ignore_errors=True)

def _build_index_csv_pdf(csv_path: str, pdf_paths: List[str], base_dir: str) -> FAISS:
    docs = _docs_from_csv(csv_path) + _docs_from_pdfs(pdf_paths)
    if not docs:
        raise FileNotFoundError("Policy index için CSV/PDF bulunamadı.")
    chunks = _split_documents_advanced(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL))
    verdir = _version_dir(base_dir)
    os.makedirs(verdir, exist_ok=True)
    vs.save_local(verdir)
    _prune_versions(base_dir, keep=2)
    return vs

def _load_latest_index(base_dir: str) -> Optional[FAISS]:
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vers = _list_versions(base_dir)
    for p in reversed(vers):
        try:
            return FAISS.load_local(p, emb, allow_dangerous_deserialization=True)
        except Exception:
            continue
    return None

_VS: Optional[FAISS] = None

def _get_vs() -> FAISS:
    global _VS
    if _VS is None:
        _VS = _load_latest_index(POLICY_INDEX_DIR)
        if _VS is None:
            _VS = _build_index_csv_pdf(POLICY_CSV, POLICY_PDFS, POLICY_INDEX_DIR)
    return _VS

def rebuild_policy_index() -> None:
    if os.path.isdir(POLICY_INDEX_DIR) and not _list_versions(POLICY_INDEX_DIR):
        shutil.rmtree(POLICY_INDEX_DIR, ignore_errors=True)
    _build_index_csv_pdf(POLICY_CSV, POLICY_PDFS, POLICY_INDEX_DIR)


# =========================== Splitters ===========================
def _tiny_merge(parts: List[str], min_chars: int = 300) -> List[str]:
    if not parts:
        return parts
    buf, out = "", []
    for p in parts:
        if not p.strip():
            continue
        if len(buf) < min_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def _split_documents_advanced(docs: List[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        if d.metadata.get("kind") == "csv":
            out.append(d)
    long_docs = [d for d in docs if d.metadata.get("kind") != "csv"]

    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    rc = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        disallowed_special=(),
    )
    for d in long_docs:
        raw = d.page_content or ""
        try:
            splitter = MarkdownHeaderTextSplitter(headers=headers, strip_headers=False)
            sections = splitter.split_text(raw)
        except Exception:
            sections = [d]
        for sec in sections:
            sec_text = (sec.page_content or "").strip()
            if not sec_text:
                continue
            parts = rc.split_text(sec_text)
            parts = _tiny_merge(parts, min_chars=300)
            base = dict(d.metadata)
            meta_sec = getattr(sec, "metadata", {}) or {}
            path = " > ".join([meta_sec.get(k, "") for _, k in headers if meta_sec.get(k)])
            if path:
                base["section_path"] = path
            for idx, p in enumerate(parts):
                out.append(Document(page_content=p, metadata={**base, "chunk_id": idx}))
    return out


# =========================== Loaders ===========================
def _docs_from_csv(csv_path: str) -> List[Document]:
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    norm = {c.strip().lower(): c for c in df.columns}
    if "question" not in norm or "answer" not in norm:
        raise ValueError("CSV must contain 'question' and 'answer' columns")

    docs: List[Document] = []
    for i, row in df.iterrows():
        q = str(row[norm["question"]]).strip()
        a = str(row[norm["answer"]]).strip()
        title = f"THY Politika (CSV): {q[:40]}..."
        src = ""
        if "source_url" in norm:
            src = str(row[norm["source_url"]]).strip()
        src = _normalize_source_url(src) or f"csv:row:{i}"
        last_upd = str(row[norm["last_updated"]]).strip() if "last_updated" in norm else ""
        meta = {"kind": "csv", "title": title, "source": src, "last_updated": last_upd, "row": int(i)}
        docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata=meta))
    return docs

def _docs_from_pdfs(pdf_paths: List[str]) -> List[Document]:
    out: List[Document] = []
    for path in pdf_paths:
        if not path or not os.path.exists(path):
            continue
        loader = PyPDFLoader(path)
        pages = loader.load()
        title = os.path.basename(path)
        for d in pages:
            page_no = d.metadata.get("page", 1)
            src_file = d.metadata.get("source", path)
            meta = {**(d.metadata or {}), "kind": "pdf", "title": title,
                    "source": f"file://{os.path.abspath(src_file)}#page={page_no}"}
            out.append(Document(page_content=d.page_content, metadata=meta))
    return out


# =========================== Prompts & structured output ===========================
class PolicyAnswer(BaseModel):
    bullets: List[str] = Field(default_factory=list, description="Kısa, bilgi yoğun maddeler; her madde 1 cümle.")
    details: str = Field(default="", description="Kaynaklardaki ifadeleri birleştiren daha uzun, akıcı açıklama.")
    last_updated: Optional[str] = Field(default=None, description="YYYY-MM-DD formatında, varsa.")
    source_ids: List[int] = Field(default_factory=list, description="[1..k] biçiminde kullanılan kaynak indeksleri.")

_parser = PydanticOutputParser(pydantic_object=PolicyAnswer)

_QA_PROMPT_BASE = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Sen bir THY havayolu politika asistanısın. SADECE verilen bağlamdaki içeriklere dayanarak cevap ver. "
         "Halüsinasyon yapma; context içinde yani bağlamda olmayan sayısal/koşulsal detayları uydurma. "
         "Zaman penceresi sorulursa genel politikayı açıkla. "
         "ÇIKTIYI SADECE JSON OLARAK ver; 'Kaynaklar' bölümünü YAZMA. "
         "Önce 'details' alanında tek parça, akıcı ve bilgi yoğun BİR paragraf üret; "
         "sonra bullets alanında en fazla 4 madde çıkar. "
         "KULLANDIĞIN HER KANITA KARŞILIK GELEN [n] numarasını 'source_ids' alanına ekle; boş bırakma."
        ),
        ("human",
         "Soru (TR):\n{question}\n\n"
         "Bağlam (her satır bir kaynak, [n] ile):\n{context}\n\n"
         "{format_instructions}"
        ),
    ]
)
_QA_PROMPT = _QA_PROMPT_BASE.partial(format_instructions=_parser.get_format_instructions())

_WIKI_PROMPT_BASE = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Aşağıdaki Vikipedi içeriğini kısa ama bilgi yoğun biçimde özetle; politika/ücret/koşul yoksa resmi sayfaya yönlendir. "
         "SADECE JSON cevabı üret; 'Kaynaklar' yazma."
        ),
        ("human",
         "Soru (TR):\n{question}\n\n"
         "Vikipedi içeriği:\n{content}\n"
         "{format_instructions}"
        ),
    ]
)
_WIKI_PROMPT = _WIKI_PROMPT_BASE.partial(format_instructions=_parser.get_format_instructions())


# =========================== Retry wrappers ===========================
class ExternalCallError(Exception): ...
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.6, min=0.5, max=3),
       retry=retry_if_exception_type(ExternalCallError))
def _safe_llm_invoke(messages: List) -> AIMessage:
    try:
        return _new_llm().invoke(messages)
    except Exception as e:
        raise ExternalCallError(str(e))


# =========================== Retrieval + Rerank ===========================
def _format_context(docs: List[Document]) -> Tuple[str, List[str], List[str]]:
    lines, urls, titles = [], [], []
    for i, d in enumerate(docs, start=1):
        src = _normalize_source_url(d.metadata.get("source"))
        title = d.metadata.get("title") or ("CSV" if d.metadata.get("kind") == "csv" else "PDF")
        last_upd = (d.metadata.get("last_updated") or "").strip()
        snippet = d.page_content.replace("\n", " ").strip()[:800]
        suffix = f" (src: {src})" if not last_upd else f" (src: {src}; günc: {last_upd})"
        lines.append(f"[{i}] {snippet}{suffix}")
        urls.append(src or "")
        titles.append(str(title))
    return "\n".join(lines), urls, titles

def _cohere_rerank(query: str, docs: List[Document]) -> List[Document]:
    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=RERANK_TOP, model="rerank-3.5")
    scored = reranker.compress_documents(docs, query=query)
    return [d for d in scored][:RERANK_TOP]

def _bge_rerank(query: str, docs: List[Document]) -> List[Document]:
    reranker = FlagReranker(BGE_MODEL, use_fp16=True)
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs, normalize=True)
    idxs = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:RERANK_TOP]
    return [docs[i] for i in idxs]

def _mmr_then_rerank(vs: FAISS, question: str) -> List[Document]:
    retriever = vs.as_retriever(search_type="mmr",
                                search_kwargs={"k": FETCH_K, "fetch_k": FETCH_K*2, "lambda_mult": 0.5})
    docs = retriever.invoke(question) or []
    docs = _dedup_docs(docs)
    try:
        if _HAS_COHERE and COHERE_API_KEY:
            docs = _cohere_rerank(question, docs)
        elif _HAS_BGE:
            docs = _bge_rerank(question, docs)
        else:
            docs = docs[:RERANK_TOP]
    except Exception:
        docs = docs[:RERANK_TOP]
    # PDF öncelik
    docs.sort(key=lambda d: (d.metadata.get("weak", False), d.metadata.get("kind") != "pdf"))
    return docs[:RETRIEVE_K]


# =========================== Wiki (fallback, circuit breaker) ===========================
def _wiki_fetch() -> str:
    global _CB_FAILS, _CB_OPEN
    if _CB_OPEN:
        raise ExternalCallError("Circuit open for wiki")
    if not (_HAS_TAVILY and WIKI_URL):
        return ""
    try:
        extractor = TavilyExtract(tavily_api_key=os.getenv("TAVILY_API_KEY"))
        page = extractor.invoke({"urls": [WIKI_URL]})
        if isinstance(page, dict) and "results" in page and page["results"]:
            _CB_FAILS = max(0, _CB_FAILS - _CB_RESET_AFTER_OK)
            return (page["results"][0].get("content") or page["results"][0].get("markdown") or "")[:5000]
        _CB_FAILS += 1
    except Exception:
        _CB_FAILS += 1
    if _CB_FAILS >= _CB_THRESHOLD:
        _CB_OPEN = True
    return ""


# =========================== QA engines ===========================
def _dual_query(question: str) -> List[str]:
    # EN sorular için TR varyantı da çek → recall artışı (opsiyonel)
    try:
        from app.utils.text import detect_lang
        lang = detect_lang(question)
    except Exception:
        lang = "tr"
    if lang == "en":
        out = _new_llm().invoke([SystemMessage(content="Translate to Turkish, only the translation."),
                                 HumanMessage(content=question)])
        tr = (out.content or "").strip()
        return [question, tr] if tr and tr.lower() != question.lower() else [question]
    return [question]

def _policy_json_answer(question: str, context_text: str) -> "PolicyAnswer":
    msg = _QA_PROMPT.invoke({"question": question, "context": context_text}).to_messages()
    resp = _safe_llm_invoke(msg)
    text = resp.content or "{}"
    if _seems_incomplete_sentence(text):
        msg = list(msg) + [AIMessage(content=text), HumanMessage(content="JSON çıktıyı tamamla.")]
        resp2 = _safe_llm_invoke(msg)
        text = resp2.content or text
    try:
        return _parser.parse(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return _parser.parse(m.group(0)) if m else PolicyAnswer()

def _wiki_json_answer(question: str, wiki_content: str) -> "PolicyAnswer":
    msg = _WIKI_PROMPT.invoke({"question": question, "content": wiki_content}).to_messages()
    resp = _safe_llm_invoke(msg)
    text = resp.content or "{}"
    try:
        return _parser.parse(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return _parser.parse(m.group(0)) if m else PolicyAnswer()


# =========================== Rendering with sentence-level quotes ===========================
def _pick_quotes_from_docs(question: str, docs: List[Document], max_quotes_total: int = 4, per_doc: int = 2) -> List[Tuple[int, str, Optional[int]]]:
    """
    Geri dönen: [(source_index_1based, "alıntı cümle", page), ...]
    - her doc için sentence-level seçim (BM25-lite + regex boost)
    - pdf sayfa numarası metadata’dan alınır
    """
    triples: List[Tuple[int, str, Optional[int]]] = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        if not text:
            continue
        sents = _select_top_sentences(question, text, top_n=per_doc)
        if not sents:
            continue
        page = _page_from_url(str(d.metadata.get("source", "")))
        for s in sents:
            triples.append((i, s.strip(), page))
    # küresel seçim: en iyi cümleler zaten local’de sıralı; ilk N
    # (daha agresif global skor istenirse _score_sentence ile yeniden skorlama yapılabilir)
    seen = set()
    out: List[Tuple[int, str, Optional[int]]] = []
    for t in triples:
        sig = (t[0], _hash_text(t[1], 256))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(t)
        if len(out) >= max_quotes_total:
            break
    return out

def _strip_llm_sources_if_any(text: str) -> str:
    m = re.search(r"\nKaynaklar:\s*(?:\n|$)", text)
    return text[:m.start()] if m else text

def _render_with_citations_structured(ans: "PolicyAnswer", citations: List[Dict[str, str]]) -> str:
    lines = []
    if ans.bullets:
        lines.append("• " + "\n• ".join(ans.bullets))
    if ans.details:
        lines.append("\n" + ans.details.strip())
    if ans.last_updated:
        lines.append(f"\nSon güncelleme: {ans.last_updated}")
    if citations:
        lines.append("\nKaynaklar:")
        for c in citations:
            idx = c.get("orig_idx")
            title = (c.get("title") or "").strip()
            url = _normalize_source_url(c.get("url") or "").strip()
            # özgün [idx] korunur ki blockquote referansı bozulmasın
            if idx is not None:
                lines.append(f"[{idx}] {title} {url}".rstrip())
            else:
                lines.append(f"{title} {url}".rstrip())
    out = "\n".join(lines).strip()
    return _strip_llm_sources_if_any(out)

def _render_final(question: str,
                  ans: "PolicyAnswer",
                  docs: List[Document],
                  all_citations: List[Dict[str, str]]) -> Tuple[str, List[Dict[str,str]]]:
    """
    Nihai metni üret:
    1) detaylı paragraf
    2) blockquote: cümle-düzeyi alıntılar ([idx] + (syf N))
    3) Kaynaklar: SADECE kullanılan kaynaklar (özgün [idx] ile)
    """
    blocks = []

    # 1) detaylı paragraf
    if ans.details:
        blocks.append(textwrap.fill(ans.details.strip(), width=100))

    # 2) alıntılar
    quotes = _pick_quotes_from_docs(question, docs, max_quotes_total=4, per_doc=2)
    used_idx_from_quotes = sorted({idx for idx, _, _ in quotes})

    if quotes:
        qlines = []
        for idx, sent, page in quotes:
            tail = f" (syf {page})" if page else ""
            qlines.append(f'> [{idx}] {sent}{tail}')
        blocks.append("\n".join(qlines))

    # 3) source_ids disiplin: LLM’in verdiğini temizle → sadece var olan ve kullanılanlar
    valid_ids = {i for i in range(1, len(docs)+1)}
    llm_ids = [i for i in ans.source_ids if i in valid_ids]
    # boşsa: alıntılardan doldur
    if not llm_ids:
        llm_ids = used_idx_from_quotes[:]
    # nihai kullanılan set: alıntılar ∪ llm_ids
    used_idx = sorted(set(used_idx_from_quotes) | set(llm_ids))

    # 4) last_updated
    if ans.last_updated:
        blocks.append(f"Son güncelleme: {ans.last_updated}")

    # 5) Kaynaklar: yalnızca used_idx; özgün [idx] korunur
    if used_idx:
        # all_citations listesi docs sırasıyla inşa edilmişti → orig_idx ekleyelim, filtreleyelim
        final_cits: List[Dict[str, str]] = []
        for orig_i, c in enumerate(all_citations, start=1):
            if orig_i in used_idx:
                final_cits.append({"title": c.get("title",""),
                                   "url": c.get("url",""),
                                   "orig_idx": orig_i})
        if final_cits:
            lines = ["Kaynaklar:"]
            seen = set()
            for c in final_cits:
                key = (c.get("title","").strip(), c.get("url","").strip(), c.get("orig_idx"))
                if key in seen:
                    continue
                seen.add(key)
                title = key[0] or "Kaynak"
                url = _normalize_source_url(key[1]) if key[1] else ""
                lines.append(f"[{key[2]}] {title} {url}".rstrip())
            blocks.append("\n".join(lines))
    else:
        final_cits = []

    text = "\n\n".join([b for b in blocks if b]).strip()
    return text, final_cits


# =========================== Public API ===========================
def answer_policy(query: str) -> Dict[str, Any]:
    """
    Single-call senaryo için sync API.
    Guardrails:
      - En az 1 blockquote + 1 kaynak yoksa, resmi sayfaya yönlendir.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass

    vs = _get_vs()

    # Kuruluş/genel bilgi -> wiki gate
    wiki_first_patterns = [
        r"\bkurul\w*\b", r"\btarihçe\b|\bhistory\b", r"\bgenel\s*merkez\b|\bmerkez[iı]\b",
        r"\bceo\b|\bgenel\s*müdür\b|\byönetim\b", r"\bfilo\b|\buçak\s*sayısı\b|\bhub\b|\bmerkez\s*havaalan[ıi]\b",
        r"\bstar\s*alliance\b|\bittifak\b|\büyeli[kğ]\b", r"\bsahip\b|\bortaklık\b|\bhisse\b", r"\bslogan\b|\blogo\b"
    ]
    if any(re.search(p, query, re.I) for p in wiki_first_patterns):
        content = _wiki_fetch()
        if content:
            pa = _wiki_json_answer(query, content)
            final_text = _render_with_citations_structured(
                pa, [{"title": "Wikipedia - Türk Hava Yolları", "url": WIKI_URL, "orig_idx": 1}]
            )
            return {"answer": final_text + f"\n\n(Son kontrol: {_today_str()})",
                    "citations": [{"title": "Wikipedia - Türk Hava Yolları", "url": WIKI_URL, "orig_idx": 1}],
                    "last_checked": _today_str(), "source_channel": "wiki"}

    # Offline-first RAG
    all_docs: List[Document] = []
    for q in _dual_query(query):
        all_docs.extend(_mmr_then_rerank(vs, q))
    docs = _dedup_docs(all_docs)[:RETRIEVE_K]

    if not docs:
        # Fallback wiki → yoksa yönlendirme
        content = _wiki_fetch()
        pa = _wiki_json_answer(query, content) if content else PolicyAnswer(
            bullets=[], details="Güncel ve resmi bilgi için THY Bilgi Edin sayfasına bakınız.",
            last_updated=None, source_ids=[]
        )
        cits = [{"title": "Wikipedia - Türk Hava Yolları", "url": WIKI_URL, "orig_idx": 1}] if content else \
               [{"title": "THY - Bilgi Edin", "url": THY_INFO_URL, "orig_idx": 1}]
        final_text = _render_with_citations_structured(pa, cits)
        return {"answer": final_text + f"\n\n(Son kontrol: {_today_str()})",
                "citations": cits, "last_checked": _today_str(),
                "source_channel": "wiki" if content else "redirect"}

    context_text, src_urls, src_titles = _format_context(docs)
    pa = _policy_json_answer(query, context_text)

    # Citations’ın temel listesi (docs sıralı)
    base_cits = [{"title": t or "", "url": u} for u, t in zip(src_urls, src_titles)]

    # Nihai render (cümle düzeyi)
    final_text, used_cits = _render_final(query, pa, docs, base_cits)

    # Guardrail: alıntı + kaynak yoksa hard cap → yönlendirme
    has_quote = bool(re.search(r'^\>\s*\[\d+\]\s', final_text, re.M))
    has_cit = "Kaynaklar:" in final_text
    if not (has_quote and has_cit):
        fallback = "\n\nNot: Bu soru için güvenilir kanıt satırı tespit edilemedi. Resmi sayfaya yönlendirme yapıldı."
        return {
            "answer": (f"Güncel ve resmi bilgi için THY Bilgi Edin sayfasına bakınız.\n{THY_INFO_URL}"
                       f"{fallback}\n\n(Son kontrol: {_today_str()})"),
            "citations": [{"title": "THY - Bilgi Edin", "url": THY_INFO_URL, "orig_idx": 1}],
            "last_checked": _today_str(),
            "source_channel": "redirect",
        }

    return {
        "answer": f"{final_text}\n\n(Son kontrol: {_today_str()})",
        "citations": used_cits,
        "last_checked": _today_str(),
        "source_channel": "csv_pdf_rag",
    }


# =========================== Warmup ===========================
if os.getenv("POLICY_WARM", "1") == "1":
    try:
        _ = _get_vs()
    except Exception:
        pass
