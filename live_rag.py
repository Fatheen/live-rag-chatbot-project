# Lightweight live RAG: fetch a few pages from a site, extract text, embed in memory, retrieve, answer with citations.

from __future__ import annotations
import re
import time
from typing import List, Tuple, Dict
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from chatbot_rag import send  

USER_AGENT = "UIUC-LiveRAG/1.0 (educational)"
HTTP_TIMEOUT = 12
MAX_BYTES = 2_500_000  # safety cap per page
EMBED_MODEL = "all-MiniLM-L6-v2"


_embedder = SentenceTransformer(EMBED_MODEL)


_page_cache: Dict[str, str] = {}

def _normalize_site(site: str) -> str:
    s = site.strip()
    if not s.startswith("http"):
        s = "https://" + s
    return s.rstrip("/")

def _same_host(url: str, root: str) -> bool:
    return urlparse(url).netloc.endswith(urlparse(root).netloc)

def _fetch(url: str) -> str:
    if url in _page_cache:
        return _page_cache[url]

    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        # drop huge binaries/pdfs/whatever
        return ""
    html = r.text

  
    text = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
    if not text or len(text.split()) < 50:
     
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    _page_cache[url] = text or ""
    return _page_cache[url]

def _discover_links(site_root: str, query_hint: str, max_pages: int) -> List[str]:
    import re, time
    from urllib.parse import urljoin, urlparse, urlunparse

    def _clean(u: str) -> str:
       
        p = urlparse(u)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))

    def _is_allowed(u: str) -> bool:
  
        if not _same_host(u, site_root):
            return False
        banned_ext = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv",
                      ".ppt", ".pptx", ".zip", ".rar", ".jpg", ".jpeg",
                      ".png", ".gif", ".svg", ".mp4", ".mov", ".txt")
        path = urlparse(u).path.lower()
        return not any(path.endswith(ext) for ext in banned_ext)

    def _extract_links(html: str, base: str) -> List[Tuple[str, str]]:
        soup = BeautifulSoup(html, "lxml")
        links: List[Tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            u = urljoin(base, a["href"])
            if _is_allowed(u):
                links.append((u, (a.get_text(" ", strip=True) or "")))
        return links

    # 1) Try sitemap.xml (best signal, broadest coverage)
    try:
        sm = requests.get(
            urljoin(site_root, "/sitemap.xml"),
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        if sm.status_code == 200 and "<loc>" in sm.text:
            sm_urls = re.findall(r"<loc>(.*?)</loc>", sm.text)
            sm_urls = [_clean(u) for u in sm_urls if _is_allowed(u)]
            sm_urls = list(dict.fromkeys(sm_urls))  # dedupe
            # score and take top max_pages
            return _rank_and_trim(sm_urls, ["" for _ in sm_urls], query_hint, max_pages)
    except Exception:
        pass  # silently fall back

    # 2) Fallback: crawl from root + common hub paths with a shallow BFS
    hub_paths = [
        "/", "/forms", "/policies", "/policy", "/travel", "/travel_and_expense_management",
        "/procurement", "/purchasing", "/accounts_payable", "/payroll", "/budget",
        "/accounting", "/grants", "/sponsored_programs", "/rates", "/resources"
    ]
    seeds = [urljoin(site_root, p) for p in hub_paths]
    seen_pages: set[str] = set()
    candidate_links: List[Tuple[str, str]] = []

    frontier = []
   
    try:
        r = requests.get(site_root, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        frontier.append((site_root, r.text))
    except Exception:
        pass

    for s in seeds:
        if s not in {u for u, _ in frontier}:
            try:
                rr = requests.get(s, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
                if rr.status_code == 200 and "text/html" in rr.headers.get("content-type", ""):
                    frontier.append((s, rr.text))
                time.sleep(0.05)
            except Exception:
                continue

  
    MAX_VISIT = min(60, max_pages * 6)  
    for base_url, html in frontier:
        if len(candidate_links) > MAX_VISIT:
            break
        for u, anchor_text in _extract_links(html, base_url):
            cu = _clean(u)
            if cu in seen_pages:
                continue
            seen_pages.add(cu)
            candidate_links.append((cu, anchor_text))
        time.sleep(0.05)

   
    urls = [u for u, _ in candidate_links]
    anchors = [a for _, a in candidate_links]
    ranked = _rank_and_trim(urls, anchors, query_hint, max_pages)
    return ranked or [site_root]


def _rank_and_trim(urls: List[str], anchors: List[str], query_hint: str, max_pages: int) -> List[str]:
    """
    Score by overlap of query terms with URL and anchor text, plus boosts
    for finance/travel keywords. Shorter, cleaner paths get a slight bonus.
    """
    import math

   
    terms = set(t for t in re.findall(r"[a-zA-Z]{3,}", (query_hint or "").lower()))

    
    boosts = {
        "travel": 2.5, "reimburse": 2.5, "reimbursement": 2.5, "expense": 2.0, "emburse": 2.0,
        "tcard": 2.0, "p-card": 2.0, "card": 1.0, "procurement": 2.0, "purchasing": 2.0,
        "payroll": 2.0, "vendor": 1.8, "supplier": 1.8, "invoice": 1.8,
        "form": 1.5, "forms": 1.5, "policy": 1.5, "policies": 1.5,
        "budget": 1.2, "accounting": 1.2, "grant": 1.2, "sponsored": 1.0, "rates": 1.0
    }

    scored = []
    for u, a in zip(urls, anchors):
        hay_url = u.lower()
        hay_anchor = (a or "").lower()

        
        base = sum(1 for t in terms if t in hay_url) + 0.8 * sum(1 for t in terms if t in hay_anchor)

        
        boost = 0.0
        for k, w in boosts.items():
            if k in hay_url or k in hay_anchor:
                boost += w

        
        path = urlparse(u).path
        depth_penalty = 0.15 * max(0, path.count("/") - 2)

        score = base + boost - depth_penalty
        scored.append((score, u))

   
    best: dict[str, float] = {}
    for sc, u in scored:
        if u not in best or sc > best[u]:
            best[u] = sc

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [u for u, _ in ranked[:max_pages]]

def _chunk(text: str, size: int = 600, overlap: int = 175) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
        i += max(step, 1)
    return chunks

def _embed(texts: List[str]) -> np.ndarray:
    
    X = _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return X.astype("float32")

def _retrieve(query: str, chunks: List[str], k: int) -> List[int]:
    if not chunks:
        return []
    Q = _embed([query])  
    X = _embed(chunks)   
    sims = cosine_similarity(Q, X)[0]
    order = np.argsort(sims)[-k:][::-1]
    return order.tolist()

def build_prompt(question: str, contexts: List[str], urls: List[str]) -> str:
    
    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); ordered.append(u)
    src_block = "\n".join(f"- {u}" for u in ordered[:5])

    return (
        "Answer strictly using the website excerpts below. "
        "If the answer isn't present, say you can't find it on the site.\n\n"
        f"Question:\n{question}\n\n"
        "Website excerpts:\n"
        + "\n\n---\n\n".join(contexts) +
        "\n\nWhen you answer:\n"
        "- Be concise and specific.\n"
        "- Quote exact numbers/dates if present.\n"
        "- End with a short 'Sources' list of the provided URLs.\n\n"
        f"Sources:\n{src_block}\n"
    )

def answer_live(question: str, site: str, max_pages: int = 8, k: int = 6) -> Dict[str, List[str] | str]:
    root = _normalize_site(site)
    # 1) discover a handful of likely pages
    urls = _discover_links(root, question, max_pages=max_pages)

    # 2) fetch + extract
    pages: List[Tuple[str, str]] = [] 
    for u in urls:
        try:
            txt = _fetch(u)
        except Exception:
            txt = ""
        if len(txt.split()) >= 80:
            pages.append((u, txt))
        
        time.sleep(0.1)

    # 3) chunk + map url->chunks
    chunk_texts: List[str] = []
    chunk_urls: List[str] = []
    for u, txt in pages:
        cs = _chunk(txt)
        if not cs:
            continue
        chunk_texts.extend(cs)
        chunk_urls.extend([u] * len(cs))

    if not chunk_texts:
        return {"answer": "I couldn't find relevant content on the site.", "sources": []}

    # 4) retrieve top-k
    idxs = _retrieve(question, chunk_texts, k=k)
    picked = [chunk_texts[i] for i in idxs]
    picked_urls = [chunk_urls[i] for i in idxs]

    # 5) build prompt + call your LLM
    prompt = build_prompt(question, picked, picked_urls)
    answer_text = send(prompt)  


    seen, ordered = set(), []
    for u in picked_urls:
        if u not in seen:
            seen.add(u); ordered.append(u)

    return {"answer": answer_text, "sources": ordered[:5]}
