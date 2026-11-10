"""Live paper discovery using public arXiv API and optional Semantic Scholar API."""
from __future__ import annotations
import urllib.parse, urllib.request, xml.etree.ElementTree as ET, json, re

ARXIV="https://export.arxiv.org/api/query"
def search_arxiv(query, max_results=20, *, submitted_after=None):
    params=urllib.parse.urlencode({"search_query":f"all:{query}","start":0,
                                  "max_results":max_results,"sortBy":"submittedDate","sortOrder":"descending"})
    req=urllib.request.Request(ARXIV+"?"+params,headers={"User-Agent":"ATREA/0.1 research-agent"})
    with urllib.request.urlopen(req,timeout=25) as r: xml_data=r.read()
    ns={"a":"http://www.w3.org/2005/Atom"}
    root=ET.fromstring(xml_data); out=[]
    for e in root.findall("a:entry",ns):
        title=" ".join((e.findtext("a:title",default="",namespaces=ns)).split())
        abstract=" ".join((e.findtext("a:summary",default="",namespaces=ns)).split())
        url=e.findtext("a:id",default="",namespaces=ns)
        published=e.findtext("a:published",default="",namespaces=ns)
        if submitted_after and published and published[:10] <= str(submitted_after)[:10]:
            continue
        out.append({"paper_id":url.rsplit("/",1)[-1],"title":title,"abstract":abstract,
                    "url":url,"published":published,"source":"arxiv",
                    "extracted_methods":extract_actionable_methods(title+" "+abstract)})
    return out

def search_semantic_scholar(query, limit=20, api_key=None):
    params=urllib.parse.urlencode({"query":query,"limit":limit,
        "fields":"title,abstract,url,year,publicationDate,externalIds"})
    headers={"User-Agent":"ATREA/0.1"}
    if api_key: headers["x-api-key"]=api_key
    req=urllib.request.Request("https://api.semanticscholar.org/graph/v1/paper/search?"+params,headers=headers)
    with urllib.request.urlopen(req,timeout=25) as r: data=json.load(r)
    out=[]
    for p in data.get("data",[]):
        url=p.get("url") or ""
        out.append({"paper_id":str(p.get("paperId")),"title":p.get("title") or "",
          "abstract":p.get("abstract") or "","url":url,"published":p.get("publicationDate") or p.get("year"),
          "source":"semantic_scholar","extracted_methods":extract_actionable_methods((p.get("title") or "")+" "+(p.get("abstract") or ""))})
    return out

def extract_actionable_methods(text):
    """Transparent keyword extraction; use an LLM afterwards for structured review."""
    patterns=[r"\b(Transformer|PatchTST|TimesNet|Chronos|TimesFM|Moirai|iTransformer|DLinear)\b",
              r"\b(LoRA|fine[- ]tuning|pretraining|self[- ]supervised|zero[- ]shot|few[- ]shot)\b",
              r"\b(decomposition|patching|channel[- ]independent|probabilistic forecasting|domain adaptation)\b"]
    found=[]
    for pat in patterns: found += re.findall(pat,text,re.I)
    return sorted(set(x.lower() for x in found))


class PaperSearchResults(list):
    """List-compatible search result with provider diagnostics."""
    def __init__(self, items=(), provider_errors=None):
        super().__init__(items)
        self.provider_errors = provider_errors or []


def search_latest_papers(query, max_results=20, *, submitted_after=None, semantic_scholar_key=None):
    """Search two scholarly indexes and rank candidates transparently.

    This is a discovery shortlist, not a claim that the top result is the globally
    best paper/model. Venue quality, reproducibility and task fit still require review.
    """
    from datetime import date, datetime
    import os
    semantic_scholar_key = semantic_scholar_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    candidates=[]; errors=[]
    try:
        candidates.extend(search_arxiv(query, max_results=max_results,
                                       submitted_after=submitted_after))
    except Exception as exc:
        errors.append({"provider":"arxiv","error":f"{type(exc).__name__}: {exc}"})
    try:
        candidates.extend(search_semantic_scholar(query, limit=max_results,
                                                  api_key=semantic_scholar_key))
    except Exception as exc:
        errors.append({"provider":"semantic_scholar","error":f"{type(exc).__name__}: {exc}"})

    def date_key(value):
        if not value: return None
        try: return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except Exception:
            try: return date(int(str(value)[:4]), 1, 1)
            except Exception: return None

    q_terms={x.lower() for x in re.findall(r"[A-Za-z0-9-]+", query) if len(x)>2}
    unique={}
    for p in candidates:
        key=(p.get("paper_id") or p.get("url") or p.get("title","")).lower()
        if not key: continue
        text=(p.get("title","")+" "+p.get("abstract","")).lower()
        lexical=(sum(1 for t in q_terms if t in text)/max(1,len(q_terms)))
        methods=p.get("extracted_methods") or []
        pub=date_key(p.get("published"))
        if submitted_after and pub and pub <= date_key(submitted_after):
            continue
        age_days=max(0,(date.today()-pub).days) if pub else 3650
        recency=max(0.0,1.0-min(age_days,730)/730)
        score=0.65*lexical+0.25*recency+0.10*min(len(methods)/4,1.0)
        p["discovery_score"]=round(score,4)
        p["discovery_signals"]={"lexical_match":round(lexical,4),
                                "recency":round(recency,4),
                                "method_terms":len(methods)}
        p["discovery_limitations"]="该分数仅用于检索排序，不代表论文质量或模型最优性。"
        if key not in unique or score>unique[key]["discovery_score"]:
            unique[key]=p
    ranked=sorted(unique.values(),key=lambda x:(x["discovery_score"],x.get("published") or ""),reverse=True)
    result=ranked[:max_results]
    # Keep provider errors even when every provider failed or returned no matches.
    return PaperSearchResults(result, provider_errors=errors)
