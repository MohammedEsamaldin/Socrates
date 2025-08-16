"""
External Factuality Checker - Real-world fact verification
Implements RAG, web search, and Wikipedia API for comprehensive fact checking with LLM-based verdicts
"""
import requests
import json
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import time
from urllib.parse import quote
import wikipedia
import os
from pathlib import Path

from ..utils.logger import setup_logger
from ..config import WIKIPEDIA_API_URL, CONFIDENCE_THRESHOLD
from .llm_manager import LLMManager

logger = setup_logger(__name__)

# Load factuality verdict prompt template
PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"
FACTUALITY_VERDICT_PROMPT = (PROMPT_TEMPLATES_DIR / "factuality_verdict.txt").read_text(encoding="utf-8")
# ================================
# Unified External API Client Layer
# ================================

class ExternalAPIClient:
    """Base client with unified request, retries, and response normalization.

    Extension guide:
    - Subclass this client and implement `_build_requests(claim)` to return a list
      of (method, url, params, headers) you want to try for the given claim.
    - Implement `_interpret(claim, payload)` to map raw API payload to the
      standardized result dict: {source, status, confidence, content, evidence, sources}.
    - Register your client in ExternalFactualityChecker by adding it to `self.clients`.
    """

    NAME = "external"

    def __init__(self, session: Optional[requests.Session] = None,
                 max_retries: int = 2, backoff_sec: float = 0.5, timeout: float = 6.0):
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.timeout = timeout

    def _build_requests(self, claim: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _interpret(self, claim: str, payload: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def query(self, claim: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for req in self._build_requests(claim):
            method = req.get("method", "GET").upper()
            url = req["url"]
            params = req.get("params")
            headers = req.get("headers")
            data = req.get("data")

            for attempt in range(self.max_retries + 1):
                try:
                    resp = self.session.request(method, url, params=params, headers=headers, data=data, timeout=self.timeout)
                    if resp.status_code == 200:
                        try:
                            payload = resp.json()
                        except ValueError:
                            payload = resp.text
                        interpreted = self._interpret(claim, payload)
                        for item in interpreted:
                            # ensure required fields
                            item.setdefault("source", self.NAME)
                            item.setdefault("status", "INCONCLUSIVE")
                            item.setdefault("confidence", 0.0)
                            item.setdefault("content", "")
                            item.setdefault("evidence", [])
                            item.setdefault("sources", [])
                        results.extend(interpreted)
                        break
                    else:
                        logger.debug(f"{self.NAME} HTTP {resp.status_code} for {url}")
                except Exception as e:
                    logger.debug(f"{self.NAME} attempt {attempt+1} error: {e}")
                # backoff before retry if not last attempt
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2 ** attempt))
        return results


class WikipediaClient(ExternalAPIClient):
    NAME = "Wikipedia"

    def __init__(self, api_url: str = WIKIPEDIA_API_URL, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url.rstrip("/") + "/"

    def _build_requests(self, claim: str) -> List[Dict[str, Any]]:
        # Extract key terms from claim for Wikipedia search
        search_terms = self._extract_search_terms(claim)
        if not search_terms:
            return []
        
        # Build proper Wikipedia API URLs for search terms
        queries = [search_terms.strip()]
        urls = []
        
        for query in queries:
            if query:
                # Use Wikipedia search API endpoint
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json&srlimit=3"
                urls.append(search_url)
        
        return [{"method": "GET", "url": url} for url in urls]

    def _interpret(self, claim: str, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            # Handle Wikipedia search API response
            query_results = payload.get("query", {})
            search_results = query_results.get("search", [])
            
            for result in search_results[:2]:  # Top 2 results
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                # Clean HTML tags from snippet
                import re
                snippet = re.sub(r'<[^>]+>', '', snippet)
                
                if title and snippet:
                    content = f"{title}: {snippet}"
                    items.append({
                        "source": self.NAME,
                        "status": "EVIDENCE_FOUND",
                        "confidence": 0.6,
                        "content": content,
                        "evidence": [snippet],
                        "sources": [f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"]
                    })
        return items
    
    def _extract_search_terms(self, claim: str) -> str:
        """Extract key search terms from claim"""
        # Simple keyword extraction - remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'was', 'were', 'is', 'are'}
        words = claim.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(key_words[:5])  # Top 5 key words


class GoogleFactCheckClient(ExternalAPIClient):
    NAME = "GoogleFactCheck"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def _build_requests(self, claim: str) -> List[Dict[str, Any]]:
        params = {
            "key": self.api_key,
            "query": claim[:200],
            "languageCode": "en",
        }
        return [{"method": "GET", "url": self.endpoint, "params": params}]

    def _interpret(self, claim: str, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            claims = payload.get("claims", []) or []
            for c in claims:
                reviews = c.get("claimReview", []) or []
                evidence_texts: List[str] = []
                sources: List[str] = []
                for r in reviews:
                    txt = r.get("textualRating") or r.get("title") or ""
                    if txt:
                        evidence_texts.append(txt)
                    url = r.get("url")
                    if url:
                        sources.append(url)
                content = c.get("text") or ""
                status = "SUPPORTED" if evidence_texts else "INCONCLUSIVE"
                confidence = 0.6 if evidence_texts else 0.25
                items.append({
                    "source": self.NAME,
                    "status": status,
                    "confidence": confidence,
                    "content": content,
                    "evidence": evidence_texts,
                    "sources": list(set(sources)),
                })
        return items


class WikidataClient(ExternalAPIClient):
    NAME = "Wikidata"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We'll use the wbsearchentities endpoint (no API key required)
        self.endpoint = "https://www.wikidata.org/w/api.php"

    def _build_requests(self, claim: str) -> List[Dict[str, Any]]:
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": claim[:150]
        }
        return [{"method": "GET", "url": self.endpoint, "params": params}]

    def _interpret(self, claim: str, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            for ent in payload.get("search", [])[:3]:
                label = ent.get("label")
                desc = ent.get("description")
                qid = ent.get("id")
                url = f"https://www.wikidata.org/wiki/{qid}" if qid else None
                content = f"{label}: {desc}" if label else (desc or "")
                items.append({
                    "source": self.NAME,
                    "status": "SUPPORTED" if content else "INCONCLUSIVE",
                    "confidence": 0.5 if content else 0.2,
                    "content": content,
                    "evidence": [desc] if desc else [],
                    "sources": [url] if url else []
                })
        return items


class ConceptNetClient(ExternalAPIClient):
    NAME = "ConceptNet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = "https://api.conceptnet.io/query"

    def _build_requests(self, claim: str) -> List[Dict[str, Any]]:
        # very light heuristic: query for edges mentioning first two keywords
        tokens = [t for t in claim.split() if t.isalpha()]
        q = " ".join(tokens[:2]) if tokens else claim
        params = {"start": "/c/en/" + q.split()[0].lower()} if q else {}
        return [{"method": "GET", "url": self.endpoint, "params": params}] if params else []

    def _interpret(self, claim: str, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            edges = payload.get("edges", [])[:3]
            for e in edges:
                rel = e.get("rel", {}).get("label")
                start = e.get("start", {}).get("label")
                end = e.get("end", {}).get("label")
                url = e.get("@id")
                snippet = f"{start} -[{rel}]-> {end}" if rel and start and end else (rel or start or end or "")
                items.append({
                    "source": self.NAME,
                    "status": "SUPPORTED" if snippet else "INCONCLUSIVE",
                    "confidence": 0.45 if snippet else 0.2,
                    "content": snippet,
                    "evidence": [snippet] if snippet else [],
                    "sources": ["https://api.conceptnet.io" + url] if url else []
                })
        return items

@dataclass
class FactCheckResult:
    """Result of external fact checking"""
    status: str  # PASS, FAIL, UNCERTAIN
    confidence: float
    external_facts: List[str]
    contradictions: List[str]
    evidence: List[str]
    sources: List[str]
    reasoning: str

class ExternalFactualityChecker:
    """
    Advanced external factuality checker using multiple sources
    Implements RAG-like functionality with Wikipedia, web search, and knowledge bases
    """
    
    def __init__(self,
                 enable_clients: Optional[bool] = None,
                 max_retries: Optional[int] = None,
                 timeout: Optional[float] = None,
                 backoff_sec: Optional[float] = None):
        logger.info("Initializing External Factuality Checker...")

        try:
            # Config via env with sensible defaults
            enable_clients = enable_clients if enable_clients is not None else (os.getenv("FACTUALITY_ENABLED", "true").lower() == "true")
            self.max_retries = int(os.getenv("FACTUALITY_MAX_RETRIES", str(max_retries if max_retries is not None else 2)))
            self.timeout = float(os.getenv("FACTUALITY_TIMEOUT", str(timeout if timeout is not None else 6.0)))
            self.backoff_sec = float(os.getenv("FACTUALITY_BACKOFF", str(backoff_sec if backoff_sec is not None else 0.5)))

            # Initialize LLM manager for factuality verdicts
            self.llm_manager = LLMManager()

            # Initialize Wikipedia helper
            wikipedia.set_lang("en")
            wikipedia.set_rate_limiting(True)

            # Register external clients (can be extended by the user)
            self.clients: List[ExternalAPIClient] = []
            if enable_clients is None:
                enable_clients = True
            if enable_clients:
                session = requests.Session()
                # Free-tier clients
                self.clients = [
                    WikipediaClient(session=session, max_retries=self.max_retries, backoff_sec=self.backoff_sec, timeout=self.timeout),
                    WikidataClient(session=session, max_retries=self.max_retries, backoff_sec=self.backoff_sec, timeout=self.timeout),
                    #ConceptNetClient(session=session, max_retries=self.max_retries, backoff_sec=self.backoff_sec, timeout=self.timeout),
                ]
                # Optional Google Fact Check Tools API
                try:
                    google_key = self._load_google_factcheck_key()
                    if google_key:
                        self.clients.append(
                            GoogleFactCheckClient(api_key=google_key, session=session, max_retries=self.max_retries, backoff_sec=self.backoff_sec, timeout=self.timeout)
                        )
                        logger.info("Google Fact Check client enabled")
                    else:
                        logger.info("Google Fact Check API key not found; skipping this client")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Fact Check client: {e}")

            # Internal knowledge base intentionally disabled for this module
            
            logger.info("External Factuality Checker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing External Factuality Checker: {str(e)}")
            raise
    
    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verify a claim against external sources
        
        Args:
            claim: The claim to verify
            
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Verifying claim externally: {claim}")
        results: List[Dict[str, Any]] = []

        # NOTE: Internal knowledge base intentionally disabled for this module

        # 2) Unified external clients (free-tier APIs)
        for client in getattr(self, "clients", []):
            try:
                client_results = client.query(claim)
                results.extend(client_results)
                logger.debug(f"{client.NAME} returned {len(client_results)} items")
            except Exception as e:
                logger.warning(f"{client.NAME} query failed: {e}")

        # 3) Legacy Wikipedia helper (kept for compatibility)
        wiki_results = self._verify_with_wikipedia(claim)
        if wiki_results:
            # _verify_with_wikipedia returns a single dict; append it
            results.append(wiki_results)

        # 4) Optional simplified web search placeholder
        web_results = self._verify_with_web_search(claim)
        if web_results:
            # _verify_with_web_search returns a single dict; append it
            results.append(web_results)

        # 5) Fallbacks (only if free-tier yielded nothing useful)
        if not results:
            tv_result = self._verify_with_tavily(claim)
            if tv_result:
                results.append(tv_result)
        if not results:
            oi_result = self._verify_with_openai(claim)
            if oi_result:
                results.append(oi_result)

        # Aggregate results
        aggregated = self._aggregate_verification_results(claim, results)
        logger.info(f"External verification status: {aggregated['status']} (conf {aggregated['confidence']:.2f})")
        return aggregated

    # ---- Optional key loading helpers ----
    def _load_google_factcheck_key(self) -> Optional[str]:
        """Load Google Fact Check Tools API key from env or local file.
        Env: GOOGLE_FACTCHECK_API_KEY
        File (optional): socrates_system/google_API_key.txt with a line like: google_API_key = "YOUR_KEY"
        """
        key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
        if key:
            return key
        # Fallback to local file if exists
        try:
            here = os.path.dirname(os.path.dirname(__file__))  # socrates_system/
            key_path = os.path.join(here, "google_API_key.txt")
            if os.path.exists(key_path):
                with open(key_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            _, val = line.split("=", 1)
                            val = val.strip().strip('"').strip("'").strip('.')
                            if val:
                                return val
        except Exception:
            pass
        return None

    # ---- Fallbacks ----
    def _verify_with_tavily(self, claim: str) -> Optional[Dict[str, Any]]:
        """Use Tavily search as a fallback if free-tier yields nothing.
        Requires TAVILY_API_KEY in environment. If library not installed or key missing, returns None.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.info("Tavily API key not set; skipping Tavily fallback")
            return None
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception:
            logger.info("tavily-python not installed; skipping Tavily fallback")
            return None
        try:
            client = TavilyClient(api_key)
            resp = client.search(query=claim, topic="news", search_depth="advanced")
            items = resp.get("results") if isinstance(resp, dict) else None
            if items:
                top = items[0]
                title = top.get("title", "")
                snippet = top.get("content", "") or top.get("snippet", "")
                url = top.get("url")
                return {
                    "source": "Tavily",
                    "status": "SUPPORTED" if snippet else "INCONCLUSIVE",
                    "confidence": 0.5 if snippet else 0.25,
                    "content": f"{title}: {snippet}".strip(": "),
                    "evidence": [snippet] if snippet else [],
                    "sources": [url] if url else [],
                }
        except Exception as e:
            logger.warning(f"Tavily fallback error: {e}")
        return None

    def _verify_with_openai(self, claim: str) -> Optional[Dict[str, Any]]:
        """Ask OpenAI to fact-check as a last-resort fallback.
        Requires OPENAI_API_KEY in environment. Uses chat completions API.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info("OpenAI API key not set; skipping OpenAI fallback")
            return None
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            prompt = (
                "You are a fact-checking assistant. Determine if the following claim is true, false, or uncertain based on reliable knowledge. "
                "Return only a JSON object with keys: status in [SUPPORTED, CONTRADICTED, INCONCLUSIVE], confidence (0-1), evidence (list of short strings), sources (list of URLs).\nClaim: "
                + claim
            )
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(body), timeout=self.timeout)
            if resp.status_code != 200:
                logger.debug(f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            # Attempt to parse JSON
            try:
                parsed = json.loads(content)
                llm_verdict = parsed.get("verdict", "INSUFFICIENT_EVIDENCE")
                if llm_verdict == "TRUE":
                    status = "SUPPORTED"
                elif llm_verdict == "FALSE":
                    status = "CONTRADICTED"
                else:
                    status = "INCONCLUSIVE"
                return {
                    "source": "OpenAI",
                    "status": status,
                    "confidence": parsed.get("confidence", 0.5),
                    "content": claim,
                    "evidence": parsed.get("evidence", []) or [],
                    "sources": parsed.get("sources", []) or [],
                }
            except Exception:
                logger.debug("OpenAI fallback returned non-JSON content")
                return None
        except Exception as e:
            logger.warning(f"OpenAI fallback error: {e}")
            return None
    
    def _verify_with_wikipedia(self, claim: str) -> Optional[Dict[str, Any]]:
        """Verify claim using Wikipedia search and page content"""
        try:
            # Extract key terms from claim for search
            search_terms = self._extract_search_terms(claim)
            search_results = wikipedia.search(search_terms, results=3)
            
            if not search_results:
                return None
            
            evidence = []
            sources = []
            
            for title in search_results[:2]:  # Check top 2 results
                try:
                    page = wikipedia.page(title)
                    summary = page.summary[:500]  # First 500 chars
                    
                    # Always collect evidence for LLM analysis (no semantic similarity filtering)
                    evidence.append(f"Wikipedia ({title}): {summary[:300]}...")
                    sources.append(f"Wikipedia: {title}")
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try first disambiguation option
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = page.summary[:500]
                        
                        evidence.append(f"Wikipedia ({e.options[0]}): {summary[:300]}...")
                        sources.append(f"Wikipedia: {e.options[0]}")
                    except:
                        continue
                        
                except:
                    continue
            
            if evidence:
                # Return evidence for LLM analysis (no pre-filtering)
                confidence = min(len(evidence) * 0.4, 0.8)
                
                return {
                    "source": "Wikipedia",
                    "status": "EVIDENCE_FOUND",  # Let LLM decide if it supports or contradicts
                    "confidence": confidence,
                    "evidence": evidence,
                    "sources": sources,
                    "content": " ".join(evidence)
                }
            
        except Exception as e:
            logger.warning(f"Wikipedia verification failed: {str(e)}")
        
        return None
    
    def _verify_with_web_search(self, claim: str) -> Optional[Dict[str, Any]]:
        """Verify claim using web search (simplified implementation)"""
        try:
            # For MVP, we'll use a simplified approach
            # In production, you'd integrate with search APIs like Google, Bing, or DuckDuckGo
            
            # Extract entities and create search query
            search_query = self._create_search_query(claim)
            
            # Simulate web search results (in production, use actual API)
            simulated_results = self._simulate_web_search(claim, search_query)
            
            if simulated_results:
                return {
                    "source": "Web Search",
                    "status": simulated_results["status"],
                    "confidence": simulated_results["confidence"],
                    "evidence": simulated_results["evidence"],
                    "sources": simulated_results["sources"],
                    "content": " ".join(simulated_results["evidence"])
                }
            
        except Exception as e:
            logger.warning(f"Web search verification failed: {str(e)}")
        
        return None
    
    def _extract_search_terms(self, claim: str) -> str:
        """Extract key terms from claim for search"""
        # Simple extraction - in production, use NER
        words = claim.split()
        
        # Filter out common words
        stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return " ".join(key_words[:5])  # Top 5 key words
    
    def _calculate_support_score(self, claim: str, text: str) -> float:
        """Calculate how much the text supports the claim"""
        # Removed SentenceTransformer initialization
        
        return 0.0
    
    def _create_search_query(self, claim: str) -> str:
        """Create optimized search query from claim"""
        # Add fact-checking keywords
        search_terms = self._extract_search_terms(claim)
        return f"{search_terms} facts verification"
    
    def _simulate_web_search(self, claim: str, search_query: str) -> Optional[Dict[str, Any]]:
        """Simulate web search results (placeholder for actual implementation)"""
        # This is a simplified simulation
        # In production, integrate with actual search APIs
        
        claim_lower = claim.lower()
        
        # Simulate some common fact patterns
        if "capital" in claim_lower and "france" in claim_lower and "paris" in claim_lower:
            return {
                "status": "SUPPORTED",
                "confidence": 0.9,
                "evidence": ["Multiple authoritative sources confirm Paris as France's capital"],
                "sources": ["Government websites", "Encyclopedia sources"]
            }
        elif "eiffel tower" in claim_lower and "rome" in claim_lower:
            return {
                "status": "CONTRADICTED",
                "confidence": 0.9,
                "evidence": ["The Eiffel Tower is located in Paris, France, not Rome"],
                "sources": ["Tourism websites", "Geographic databases"]
            }
        elif "water" in claim_lower and "boil" in claim_lower and "100" in claim_lower:
            return {
                "status": "SUPPORTED",
                "confidence": 0.95,
                "evidence": ["Scientific sources confirm water boils at 100°C at standard pressure"],
                "sources": ["Scientific databases", "Educational resources"]
            }
        
        return None
    
    def _aggregate_verification_results(self, claim: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to aggregate and analyze results from multiple sources"""
        if not results:
            return {
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "external_facts": ["No external sources found"],
            "contradictions": [],
            "evidence": [],
            "sources": [],
            "reasoning": "No external sources available for verification"
        }
    
        # Use LLM to make the final factuality determination
        return self._get_llm_factuality_verdict(claim, results)
    """Aggregate verification results from multiple sources"""      
                    
                    
    # def _aggregate_verification_results(self, claim: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Aggregate results from multiple sources"""
    #     if not results:
    #         return {
    #             "status": "UNCERTAIN",
    #             "confidence": 0.0,
    #             "external_facts": ["No external sources found"],
    #             "contradictions": [],
    #             "evidence": [],
    #             "sources": [],
    #             "reasoning": "No external sources available for verification"
    #         }
        
    #     # Analyze results
    #     supported_results = [r for r in results if r.get("status") in ["TRUE", "SUPPORTED"]]
    #     contradicted_results = [r for r in results if r.get("status") in ["FALSE", "CONTRADICTED"]]
        
    #     external_facts = []
    #     evidence = []
    #     sources = []
    #     contradictions = []
        
    #     # Collect evidence and sources
    #     for result in results:
    #         if result.get("evidence"):
    #             evidence.extend(result["evidence"])
    #         if result.get("sources"):
    #             sources.extend(result["sources"])
    #         if result.get("content"):
    #             external_facts.append(result["content"])
        
    #     # Determine overall status
    #     if contradicted_results:
    #         status = "FAIL"
    #         for result in contradicted_results:
    #             contradictions.append(f"Source {result['source']}: {result.get('content', 'Contradiction found')}")
    #     elif supported_results:
    #         status = "PASS"
    #     else:
    #         status = "UNCERTAIN"
        
    #     # Calculate overall confidence
    #     if results:
    #         confidence = sum(r.get("confidence", 0) for r in results) / len(results)
    #     else:
    #         confidence = 0.0
        
    #     # Generate reasoning
    #     reasoning = self._generate_verification_reasoning(claim, results, status, confidence)
        
    #     return {
    #         "status": status,
    #         "confidence": confidence,
    #         "external_facts": external_facts,
    #         "contradictions": contradictions,
    #         "evidence": evidence,
    #         "sources": list(set(sources)),  # Remove duplicates
    #         "reasoning": reasoning
    #     }
    
    def _generate_verification_reasoning(self, claim: str, results: List[Dict[str, Any]], 
                                       status: str, confidence: float) -> str:
        """Generate reasoning for verification decision"""
        reasoning_parts = []
        
        # Source summary
        sources = [r.get("source", "Unknown") for r in results]
        reasoning_parts.append(f"Consulted {len(results)} external sources: {', '.join(set(sources))}")
        
        # Status explanation
        if status == "PASS":
            reasoning_parts.append(f"External sources support the claim with {confidence:.2f} confidence")
        elif status == "FAIL":
            reasoning_parts.append(f"External sources contradict the claim with {confidence:.2f} confidence")
        else:
            reasoning_parts.append(f"External sources provide insufficient evidence ({confidence:.2f} confidence)")
        
        # Evidence summary
        total_evidence = sum(len(r.get("evidence", [])) for r in results)
        if total_evidence > 0:
            reasoning_parts.append(f"Found {total_evidence} pieces of supporting evidence")
        
        return ". ".join(reasoning_parts)
    
    def batch_verify_claims(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Verify multiple claims in batch"""
        logger.info(f"Batch verifying {len(claims)} claims")
        
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Verifying claim {i+1}/{len(claims)}")
            result = self.verify_claim(claim)
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    def _get_llm_factuality_verdict(self, claim: str, evidence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to determine factuality verdict based on claim and evidence"""
        
        # Prepare evidence and sources for LLM analysis
        evidence_text = []
        sources_list = []
        
        for result in evidence_results:
            if result.get("content"):
                evidence_text.append(f"- {result['content']}")
            if result.get("evidence"):
                evidence_text.extend([f"- {ev}" for ev in result["evidence"]])
            if result.get("sources"):
                sources_list.extend(result["sources"])
        
        evidence_combined = "\n".join(evidence_text) if evidence_text else "No specific evidence found"
        sources_combined = "\n".join(f"- {source}" for source in set(sources_list)) if sources_list else "No sources available"
        
        # Format prompt with claim, evidence, and sources
        prompt = FACTUALITY_VERDICT_PROMPT.format(
            claim=claim,
            evidence=evidence_combined,
            sources=sources_combined
        )
        
        try:
            # Get LLM response
            llm_response = self.llm_manager.generate_text(prompt, max_tokens=1024)
            
            # Clean and parse JSON response
            if not llm_response or not llm_response.strip():
                logger.warning("Empty LLM response for factuality verdict")
                return self._create_fallback_verdict(claim, evidence_text, sources_list)
            
            # Remove markdown code blocks if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON response
            verdict_data = json.loads(cleaned_response)
            
            # Map LLM verdict to our status format
            llm_verdict = verdict_data.get("verdict", "INSUFFICIENT_EVIDENCE")
            if llm_verdict == "TRUE":
                status = "PASS"
            elif llm_verdict == "FALSE":
                status = "FAIL"
            else:
                status = "UNCERTAIN"
            
            return {
                "status": status,
                "confidence": float(verdict_data.get("confidence", 0.5)),
                "external_facts": evidence_text,
                "contradictions": verdict_data.get("contradicting_evidence", []),
                "evidence": verdict_data.get("supporting_evidence", []),
                "sources": list(set(sources_list)),
                "reasoning": verdict_data.get("reasoning", "LLM-based factuality analysis"),
                "llm_verdict": verdict_data  # Store full LLM response
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM factuality verdict: {e}")
            return self._create_fallback_verdict(claim, evidence_text, sources_list)

    def _create_fallback_verdict(self, claim: str, evidence_text: List[str], sources_list: List[str]) -> Dict[str, Any]:
        """Create fallback verdict when LLM parsing fails"""
        return {
            "status": "UNCERTAIN",
            "confidence": 0.3,
            "external_facts": evidence_text,
            "contradictions": [],
            "evidence": evidence_text,
            "sources": list(set(sources_list)),
            "reasoning": "LLM verdict parsing failed - using fallback analysis"
        }
