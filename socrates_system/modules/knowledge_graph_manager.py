"""
Knowledge Graph Manager - Advanced knowledge representation and management
Implements entity/relation extraction, graph construction, and querying capabilities
"""
import sqlite3
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import asyncio

from ..utils.logger import setup_logger
from ..config import KG_DATABASE_PATH, ENTITY_MODEL_NAME, NLP_MODEL_NAME

logger = setup_logger(__name__)

@dataclass
class Entity:
    """Represents a knowledge graph entity"""
    id: str
    text: str
    label: str
    confidence: float
    attributes: Dict[str, Any]
    session_id: str
    timestamp: datetime

@dataclass
class Relation:
    """Represents a relationship between entities"""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    evidence: List[str]
    session_id: str
    timestamp: datetime

@dataclass
class Claim:
    """Represents a verified claim in the knowledge graph"""
    id: str
    text: str
    entities: List[str]
    relations: List[str]
    confidence: float
    evidence: List[str]
    session_id: str
    timestamp: datetime

class StableId:
    @staticmethod
    def _sha(prefix: str, content: str) -> str:
        return f"{prefix}_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    @staticmethod
    def entity(text: str, label: str) -> str:
        base = f"{(text or '').lower().strip()}|{label}"
        return StableId._sha("ent", base)

    @staticmethod
    def relation(subject_id: str, predicate: str, object_id: str) -> str:
        base = f"{subject_id}|{predicate}|{object_id}"
        return StableId._sha("rel", base)

    @staticmethod
    def canonical(normalized_text: str, entity_type: str) -> str:
        base = f"{(normalized_text or '').lower().strip()}|{entity_type}"
        return StableId._sha("canonical", base)

class KnowledgeGraphManager:
    """
    Advanced Knowledge Graph Manager for the Socrates Agent
    Handles entity extraction, relation extraction, graph construction, and querying
    """
    
    def __init__(self, llm_manager: Any = None):
        logger.info("Initializing Knowledge Graph Manager...")
        
        try:
            # Load NLP models
            self.nlp = spacy.load(ENTITY_MODEL_NAME)
            self.sentence_model = SentenceTransformer(NLP_MODEL_NAME)
            
            # Optional LLM for future LLM-first extraction
            self.llm = llm_manager

            # Embedding cache to avoid recomputation
            self._embedding_cache: Dict[str, np.ndarray] = {}
            
            # Initialize database
            self._init_database()
            
            # In-memory graph for fast operations
            self.session_graphs = {}
            
            # Relation extraction patterns
            self.relation_patterns = {
                'is_a': [
                    r'(.+) is a (.+)',
                    r'(.+) is an (.+)',
                    r'(.+) are (.+)'
                ],
                'located_in': [
                    r'(.+) is located in (.+)',
                    r'(.+) is in (.+)',
                    r'(.+) is situated in (.+)'
                ],
                'part_of': [
                    r'(.+) is part of (.+)',
                    r'(.+) belongs to (.+)',
                    r'(.+) is a component of (.+)'
                ],
                'has_property': [
                    r'(.+) has (.+)',
                    r'(.+) contains (.+)',
                    r'(.+) possesses (.+)'
                ],
                'temporal': [
                    r'(.+) happened in (.+)',
                    r'(.+) occurred in (.+)',
                    r'(.+) was built in (.+)',
                    r'(.+) was founded in (.+)'
                ],
                'causal': [
                    r'(.+) causes (.+)',
                    r'(.+) leads to (.+)',
                    r'(.+) results in (.+)'
                ]
            }
            # Predicate aliases for normalization
            self.predicate_aliases = {
                'is a': 'is_a',
                'is an': 'is_a',
                'are': 'is_a',
                'is_a': 'is_a',
                'located in': 'located_in',
                'in': 'located_in',
                'situated in': 'located_in',
                'located_in': 'located_in',
                'part of': 'part_of',
                'belongs to': 'part_of',
                'component of': 'part_of',
                'part_of': 'part_of',
                'has': 'has_property',
                'contains': 'has_property',
                'possesses': 'has_property',
                'has_property': 'has_property',
                'causes': 'causal',
                'leads to': 'causal',
                'results in': 'causal',
                'causal': 'causal',
                'subject_of': 'subject_of',
                'object_of': 'object_of',
                'related_to': 'related_to',
            }
            
            logger.info("Knowledge Graph Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Knowledge Graph Manager: {str(e)}")
            raise
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                attributes TEXT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES entities (id),
                FOREIGN KEY (object_id) REFERENCES entities (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                entities TEXT,
                relations TEXT,
                confidence REAL NOT NULL,
                evidence TEXT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # Canonicalization registry tables (cross-session)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS canonical_entities (
                canonical_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                name TEXT NOT NULL,
                aliases TEXT,
                created_ts TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_canonical_map (
                entity_id TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL,
                score REAL,
                mapped_ts TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities (id),
                FOREIGN KEY (canonical_id) REFERENCES canonical_entities (canonical_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS canonical_relations (
                id TEXT PRIMARY KEY,
                subject_canonical_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_canonical_id TEXT NOT NULL,
                support_count INTEGER DEFAULT 1,
                confidence REAL,
                created_ts TEXT NOT NULL,
                FOREIGN KEY (subject_canonical_id) REFERENCES canonical_entities (canonical_id),
                FOREIGN KEY (object_canonical_id) REFERENCES canonical_entities (canonical_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relation_canonical_map (
                relation_id TEXT PRIMARY KEY,
                canonical_relation_id TEXT NOT NULL,
                mapped_ts TEXT NOT NULL,
                FOREIGN KEY (relation_id) REFERENCES relations (id),
                FOREIGN KEY (canonical_relation_id) REFERENCES canonical_relations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def initialize_session(self, session_id: str):
        """Initialize a new session knowledge graph"""
        logger.info(f"Initializing session KG: {session_id}")
        
        self.session_graphs[session_id] = nx.MultiDiGraph()
        
        # Load existing session data if available
        self._load_session_data(session_id)
    
    def extract_entities_and_relations(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from text using advanced NLP
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (entities, relations)
        """
        logger.info(f"Extracting entities and relations from: {text[:50]}...")

        llm_entities: List[Entity] = []
        llm_relations: List[Relation] = []
        try:
            llm_out = self._llm_first_extract(text) if getattr(self, 'llm', None) else None
            if llm_out:
                llm_entities, llm_relations = llm_out
        except Exception as e:
            logger.debug(f"LLM-first extraction unavailable: {e}")

        # Always run spaCy to enrich and as fallback
        doc = self.nlp(text)
        spacy_entities = self._extract_entities(doc, text)
        extra_entities, extra_relations = self._extract_custom_ordinal_entities_relations(doc, text, spacy_entities)
        if extra_entities:
            spacy_entities.extend(extra_entities)
        spacy_relations = self._extract_relations(doc, text, spacy_entities)
        if extra_relations:
            spacy_relations.extend(extra_relations)

        # Merge: prefer LLM if available, then enrich with spaCy
        entities = (llm_entities or []) + spacy_entities
        base_relations = (llm_relations or []) + spacy_relations

        # Dedupe and refine with LLM relationship extraction hybrid
        entities = self._dedupe_entities(entities)
        base_relations = self._dedupe_relations(base_relations)
        relations = self._refine_relations_hybrid(text, entities, base_relations)

        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations (LLM+pattern hybrid)")
        return entities, relations

    def _create_entity(self, text: str, label: str, confidence: float, attributes: Optional[Dict[str, Any]] = None) -> Entity:
        attributes = attributes or {}
        return Entity(
            id=StableId.entity(text, label),
            text=text,
            label=label,
            confidence=confidence,
            attributes=attributes,
            session_id="",
            timestamp=datetime.now(),
        )

    def _create_relation(self, subject_id: str, predicate: str, object_id: str, confidence: float, evidence: Optional[List[str]] = None) -> Relation:
        return Relation(
            id=StableId.relation(subject_id, predicate, object_id),
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            evidence=evidence or [],
            session_id="",
            timestamp=datetime.now(),
        )

    def _llm_extract_claims_sync(self, text: str):
        """Call LLMManager.extract_claims asynchronously from sync code.
        Returns LLMResponse or None on failure."""
        if not getattr(self, 'llm', None):
            return None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.llm.extract_claims(text))
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"LLM extract_claims failed: {e}")
            return None

    def _llm_extract_relationships_sync(self, text: str, entities: List[str]):
        """Call LLMManager.extract_relationships asynchronously from sync code.
        Returns LLMResponse or None on failure."""
        if not getattr(self, 'llm', None):
            return None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.llm.extract_relationships(text, entities))
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"LLM extract_relationships failed: {e}")
            return None

    def _normalize_predicate(self, pred: str) -> str:
        p = (pred or '').strip().lower().replace('-', ' ')
        p = re.sub(r"\s+", " ", p)
        canon = self.predicate_aliases.get(p)
        if canon:
            return canon
        p = p.replace(' ', '_')
        return self.predicate_aliases.get(p, p)

    def _normalize_existing_relations(self, relations: List[Relation]) -> List[Relation]:
        """Normalize predicate strings in-place and update stable IDs accordingly."""
        for r in relations:
            try:
                new_pred = self._normalize_predicate(r.predicate)
                if new_pred != r.predicate:
                    r.predicate = new_pred
                    r.id = StableId.relation(r.subject_id, new_pred, r.object_id)
            except Exception:
                pass
        return relations

    def _merge_relations_with_scoring(self, base: List[Relation], new: List[Relation]) -> List[Relation]:
        """Merge two relation lists by (subject, normalized predicate, object) with confidence fusion and evidence union."""
        merged: Dict[Tuple[str, str, str], Relation] = {}
        def add_rel(rel: Relation):
            key = (rel.subject_id, self._normalize_predicate(rel.predicate), rel.object_id)
            if key in merged:
                existing = merged[key]
                ev = list(existing.evidence or []) + list(rel.evidence or [])
                seen = set()
                dedup_ev = []
                for e in ev:
                    if e not in seen:
                        seen.add(e)
                        dedup_ev.append(e)
                existing.evidence = dedup_ev
                try:
                    c1 = float(existing.confidence)
                    c2 = float(rel.confidence)
                    fused = 1 - (1 - min(max(c1, 0.0), 1.0)) * (1 - min(max(c2, 0.0), 1.0))
                    existing.confidence = min(1.0, max(existing.confidence, rel.confidence, fused))
                except Exception:
                    existing.confidence = max(existing.confidence, rel.confidence)
                existing.predicate = self._normalize_predicate(existing.predicate)
                existing.id = StableId.relation(existing.subject_id, existing.predicate, existing.object_id)
            else:
                rel.predicate = self._normalize_predicate(rel.predicate)
                rel.id = StableId.relation(rel.subject_id, rel.predicate, rel.object_id)
                merged[key] = rel
        for r in base:
            add_rel(r)
        for r in new:
            add_rel(r)
        return list(merged.values())

    def _refine_relations_hybrid(self, text: str, entities: List[Entity], base_relations: List[Relation]) -> List[Relation]:
        """Hybrid refinement: normalize and merge pattern/dependency relations with LLM-extracted ones.
        If LLM unavailable or fails, returns normalized+deduped base relations.
        """
        self._normalize_existing_relations(base_relations)
        if not getattr(self, 'llm', None):
            return self._dedupe_relations(base_relations)
        name_to_id: Dict[str, str] = {e.text.lower().strip(): e.id for e in entities}
        ent_names = [e.text for e in entities]
        response = self._llm_extract_relationships_sync(text, ent_names)
        llm_relations: List[Relation] = []
        if response and not getattr(response, 'error', None):
            structured = getattr(response, 'structured_output', None) or {}
            rel_items = []
            if isinstance(structured, dict):
                if isinstance(structured.get('relationships'), list):
                    rel_items.extend(structured.get('relationships') or [])
                if isinstance(structured.get('relations'), list):
                    rel_items.extend(structured.get('relations') or [])
            elif isinstance(structured, list):
                rel_items = structured
            for r in rel_items:
                try:
                    subj = r.get('source') or r.get('subject') or r.get('from') or r.get('head') or r.get('entity1')
                    obj = r.get('target') or r.get('object') or r.get('to') or r.get('tail') or r.get('entity2')
                    pred = r.get('relation') or r.get('predicate') or r.get('type') or 'related_to'
                    if not subj or not obj:
                        continue
                    sname = str(subj).strip()
                    oname = str(obj).strip()
                    sid = name_to_id.get(sname.lower())
                    oid = name_to_id.get(oname.lower())
                    if not sid:
                        for k, vid in name_to_id.items():
                            if k in sname.lower() or sname.lower() in k:
                                sid = vid
                                break
                    if not oid:
                        for k, vid in name_to_id.items():
                            if k in oname.lower() or oname.lower() in k:
                                oid = vid
                                break
                    if not sid or not oid:
                        continue
                    try:
                        rconf = float(r.get('confidence', getattr(response, 'confidence', 0.75) or 0.75))
                    except Exception:
                        rconf = getattr(response, 'confidence', 0.75) or 0.75
                    evidence = []
                    ctx = r.get('context') if isinstance(r, dict) else None
                    if isinstance(ctx, str) and ctx.strip():
                        evidence.append(ctx.strip())
                    if text not in evidence:
                        evidence.append(text)
                    norm_pred = self._normalize_predicate(str(pred))
                    rel_obj = self._create_relation(sid, norm_pred, oid, rconf, evidence=evidence)
                    llm_relations.append(rel_obj)
                except Exception as ex:
                    logger.debug(f"Skipping malformed LLM relationship in refine: {r} ({ex})")
        merged = self._merge_relations_with_scoring(base_relations, llm_relations)
        return self._dedupe_relations(merged)

    def _dedupe_entities(self, entities: List[Entity]) -> List[Entity]:
        """Dedupe entities by stable id, keep highest confidence and merge attributes."""
        by_id: Dict[str, Entity] = {}
        for e in entities:
            existing = by_id.get(e.id)
            if not existing or e.confidence > existing.confidence:
                by_id[e.id] = e
            elif existing:
                merged_attrs = dict(existing.attributes or {})
                merged_attrs.update(e.attributes or {})
                existing.attributes = merged_attrs
        return list(by_id.values())

    def _dedupe_relations(self, relations: List[Relation]) -> List[Relation]:
        """Dedupe relations by stable id, keep highest confidence and merge evidence."""
        by_id: Dict[str, Relation] = {}
        for r in relations:
            existing = by_id.get(r.id)
            if not existing or r.confidence > existing.confidence:
                by_id[r.id] = r
            elif existing:
                ev = list(existing.evidence or []) + list(r.evidence or [])
                seen = set()
                merged = []
                for item in ev:
                    if item not in seen:
                        seen.add(item)
                        merged.append(item)
                existing.evidence = merged
        return list(by_id.values())

    def resolve_canonical_id(self, text: str, label: str, session_id: Optional[str] = None) -> Optional[str]:
        """Resolve the canonical entity ID for a given entity text and label.

        Resolution order:
        1) In-memory session graph (exact text+label match) canonical_id attribute.
        2) SQLite entity_canonical_map for the stable entity id.
        3) SQLite canonical_entities by exact name (normalized) and aliases filtered by label.

        Returns the canonical_id if found, otherwise None.
        """
        try:
            norm_text = re.sub(r"\s+", " ", (text or "")).strip().lower()
            norm_label = (label or "").strip()
            if not norm_text or not norm_label:
                return None

            # 1) In-memory session graph lookup
            if session_id and session_id in self.session_graphs:
                try:
                    g = self.session_graphs[session_id]
                    for node_id, data in g.nodes(data=True):
                        try:
                            t = str(data.get("text", "")).strip().lower()
                            l = str(data.get("label", "")).strip()
                            if t == norm_text and l == norm_label:
                                cid = data.get("canonical_id")
                                if cid:
                                    return cid
                                # Fallback to DB mapping for this node
                                try:
                                    conn = sqlite3.connect(KG_DATABASE_PATH)
                                    cur = conn.cursor()
                                    cur.execute("SELECT canonical_id FROM entity_canonical_map WHERE entity_id = ?", (node_id,))
                                    r = cur.fetchone()
                                    conn.close()
                                    if r and r[0]:
                                        return r[0]
                                except Exception:
                                    pass
                        except Exception:
                            continue
                except Exception:
                    pass

            # 2) Direct DB lookup by stable entity id
            stable_eid = StableId.entity(text, norm_label)
            try:
                conn = sqlite3.connect(KG_DATABASE_PATH)
                cur = conn.cursor()
                cur.execute("SELECT canonical_id FROM entity_canonical_map WHERE entity_id = ?", (stable_eid,))
                r = cur.fetchone()
                if r and r[0]:
                    conn.close()
                    return r[0]
                # 3) Search canonical_entities by label, name, aliases
                cur.execute("SELECT canonical_id, name, aliases FROM canonical_entities WHERE label = ?", (norm_label,))
                rows = cur.fetchall()
                conn.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                rows = []

            for row in rows or []:
                try:
                    cid, name, aliases_json = row[0], str(row[1] or "").strip().lower(), row[2]
                    if norm_text == name:
                        return cid
                    if aliases_json:
                        try:
                            aliases = json.loads(aliases_json)
                        except Exception:
                            aliases = []
                        if any(norm_text == str(a or "").strip().lower() for a in (aliases or [])):
                            return cid
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.debug(f"resolve_canonical_id failed for '{text}'/{label}: {e}")
            return None

    def _llm_first_extract(self, text: str) -> Optional[Tuple[List[Entity], List[Relation]]]:
        """LLM-first extraction of entities and relations with robust parsing.
        Returns (entities, relations) or None if the LLM output is unusable."""
        response = self._llm_extract_claims_sync(text)
        if not response or response.error:
            return None
        structured = response.structured_output or {}

        collected_entities: List[Dict[str, Any]] = []
        collected_relationships: List[Dict[str, Any]] = []

        if isinstance(structured, dict):
            if isinstance(structured.get('entities'), list):
                collected_entities.extend(structured.get('entities') or [])
            if isinstance(structured.get('relationships'), list):
                collected_relationships.extend(structured.get('relationships') or [])

        claims = structured.get('claims') if isinstance(structured, dict) else None
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict):
                    ents = claim.get('entities')
                    rels = claim.get('relationships')
                    if isinstance(ents, list):
                        collected_entities.extend(ents)
                    if isinstance(rels, list):
                        collected_relationships.extend(rels)

        if not collected_entities and not collected_relationships:
            return None

        entities: List[Entity] = []
        name_to_entity: Dict[str, Entity] = {}
        for e in collected_entities:
            try:
                name = (e.get('name') or e.get('text') or e.get('label') or '').strip()
                if not name:
                    continue
                etype = (e.get('type') or e.get('label') or 'ENTITY').strip()
                econf = float(e.get('confidence', 0.8)) if isinstance(e.get('confidence'), (int, float, str)) else 0.8
                attrs = dict(e)
                for k in ['name', 'text', 'type', 'label', 'confidence']:
                    attrs.pop(k, None)
                ent_obj = self._create_entity(name, etype, econf, attributes=attrs)
                entities.append(ent_obj)
                name_to_entity[name.lower()] = ent_obj
            except Exception as ex:
                logger.debug(f"Skipping malformed LLM entity: {e} ({ex})")

        relations: List[Relation] = []
        def _resolve_entity_id(ref_name: str) -> str:
            ref = (ref_name or '').strip()
            if not ref:
                return ''
            found = name_to_entity.get(ref.lower())
            if found:
                return found.id
            fallback_ent = self._create_entity(ref, 'ENTITY', 0.6)
            entities.append(fallback_ent)
            name_to_entity[ref.lower()] = fallback_ent
            return fallback_ent.id

        for r in collected_relationships:
            try:
                subj = r.get('source') or r.get('subject') or r.get('from') or r.get('head') or r.get('entity1')
                obj = r.get('target') or r.get('object') or r.get('to') or r.get('tail') or r.get('entity2')
                pred = r.get('relation') or r.get('predicate') or r.get('type') or 'related_to'
                if not subj or not obj:
                    continue
                sid = _resolve_entity_id(str(subj))
                oid = _resolve_entity_id(str(obj))
                if not sid or not oid:
                    continue
                try:
                    rconf = float(r.get('confidence', response.confidence or 0.75))
                except Exception:
                    rconf = response.confidence or 0.75
                evidence = []
                ctx = r.get('context') if isinstance(r, dict) else None
                if isinstance(ctx, str) and ctx.strip():
                    evidence.append(ctx.strip())
                if text not in evidence:
                    evidence.append(text)
                rel_obj = self._create_relation(sid, str(pred), oid, rconf, evidence=evidence)
                relations.append(rel_obj)
            except Exception as ex:
                logger.debug(f"Skipping malformed LLM relation: {r} ({ex})")

        entities = self._dedupe_entities(entities)
        relations = self._dedupe_relations(relations)
        if not entities and not relations:
            return None
        return entities, relations

    def _ordinal_word_to_index(self, word: str) -> Optional[int]:
        w = word.lower().strip()
        mapping = {
            "first": 1, "1st": 1,
            "second": 2, "2nd": 2,
            "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5,
            "sixth": 6, "6th": 6,
            "seventh": 7, "7th": 7,
            "eighth": 8, "8th": 8,
            "ninth": 9, "9th": 9,
            "tenth": 10, "10th": 10,
        }
        if w in mapping:
            return mapping[w]
        # numeric ordinals like 11th, 12th
        m = re.match(r"^(\d+)(st|nd|rd|th)$", w)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _simple_lemma(self, noun: str) -> str:
        n = noun.strip().lower()
        if n.endswith('s') and len(n) > 3:
            return n[:-1]
        return n

    def _detect_colors_in_context(self, text: str) -> List[str]:
        t = f" {text.lower()} "
        colors = ["red", "blue", "green", "yellow", "black", "white", "gray", "grey", "silver", "gold", "orange", "purple", "brown", "beige"]
        found = []
        for c in colors:
            if f" {c} " in t:
                found.append(c)
        return list(dict.fromkeys(found))  # unique, preserve order

    def _extract_custom_ordinal_entities_relations(self, doc, text: str, base_entities: List[Entity]) -> Tuple[List[Entity], List[Relation]]:
        """Detect ordinal entities like 'the first dog' and groups like '2 dogs',
        create linking relations, and attach simple attributes (e.g., color) when possible.
        """
        extra_entities: List[Entity] = []
        extra_relations: List[Relation] = []

        # 1) Detect group phrases like '2 dogs'
        group_matches = list(re.finditer(r"\b(\d+)\s+([A-Za-z]+)s\b", text))
        groups = []  # [(entity, base)]
        for gm in group_matches:
            count = int(gm.group(1))
            noun_plural = gm.group(2) + 's'
            base = self._simple_lemma(gm.group(2))
            group_text = gm.group(0)
            group_ent = self._create_entity(
                text=group_text,
                label="GROUP_ENTITY",
                confidence=0.85,
                attributes={"base": base, "count": count}
            )
            # Avoid duplicates by text+label
            if not any(e.text == group_ent.text and e.label == group_ent.label for e in (base_entities + extra_entities)):
                extra_entities.append(group_ent)
                groups.append((group_ent, base))
            else:
                # Find existing matching entity
                existing = next((e for e in (base_entities + extra_entities) if e.text == group_ent.text and e.label == group_ent.label), None)
                if existing:
                    groups.append((existing, base))

        # 2) Detect ordinal phrases like 'the first dog' or '2nd cat'
        ord_pattern = re.compile(r"\b(?:the\s+)?((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)))\s+([A-Za-z]+)\b", re.IGNORECASE)
        for m in ord_pattern.finditer(text):
            ord_word = m.group(1)
            noun = m.group(2)
            idx = self._ordinal_word_to_index(ord_word)
            if not idx:
                continue
            base = self._simple_lemma(noun)
            phrase_text = m.group(0)
            canon = f"{base}#{idx}"
            ord_ent = self._create_entity(
                text=phrase_text,
                label="ORDINAL_ENTITY",
                confidence=0.9,
                attributes={"base": base, "ordinal_index": idx, "ordinal_label": ord_word.lower(), "canonical": canon}
            )
            # Deduplicate by text+label
            if not any(e.text.lower() == ord_ent.text.lower() and e.label == ord_ent.label for e in (base_entities + extra_entities)):
                extra_entities.append(ord_ent)

            # Also create a canonical entity (e.g., dog#1) and alias the ordinal phrase to it
            canonical_ent = self._create_entity(
                text=canon,
                label="CANONICAL_ENTITY",
                confidence=0.95,
                attributes={"base": base, "ordinal_index": idx}
            )
            existing_canon = next((e for e in (base_entities + extra_entities) if e.text.lower() == canonical_ent.text.lower() and e.label == canonical_ent.label), None)
            if not existing_canon:
                extra_entities.append(canonical_ent)
                canonical_id = canonical_ent.id
            else:
                canonical_id = existing_canon.id
            # alias link: ordinal -> canonical
            extra_relations.append(self._create_relation(ord_ent.id, "alias_of", canonical_id, 0.95, [text]))

            # Link to any detected matching group (same base)
            for g_ent, g_base in groups:
                if g_base == base:
                    extra_relations.append(self._create_relation(ord_ent.id, "member_of", g_ent.id, 0.9, [text]))

            # Additionally, try to link to an existing group entity in the KG by base (across previous claims)
            try:
                existing_group = self._find_group_entity_by_base(base)
                if existing_group is not None:
                    extra_relations.append(self._create_relation(ord_ent.id, "member_of", existing_group.id, 0.9, [f"link-to-existing-group:{existing_group.text}"]))
            except Exception:
                pass

            # 3) Attach attribute: color, if found in local context
            colors = self._detect_colors_in_context(text)
            # Prefer context like 'first dog is yellow' (post-phrase)
            after = text[m.end(): m.end() + 64].lower()
            color_for_ord = None
            for c in colors:
                if re.search(rf"\b(is|looks|appears|seems|being)\s+{re.escape(c)}\b", after):
                    color_for_ord = c
                    break
            # Fallback: color adjective preceding phrase: 'yellow first dog'
            if not color_for_ord:
                before = text[max(0, m.start()-32): m.start()].lower()
                for c in colors:
                    if f" {c} " in before:
                        color_for_ord = c
                        break
            if color_for_ord:
                color_ent = self._create_entity(text=color_for_ord, label="ATTRIBUTE_VALUE", confidence=0.95, attributes={"type": "color"})
                # ensure we have unique color entity in extras
                existing_color = next((e for e in (base_entities + extra_entities) if e.text == color_ent.text and e.label == color_ent.label), None)
                if not existing_color:
                    extra_entities.append(color_ent)
                    target_color_id = color_ent.id
                else:
                    target_color_id = existing_color.id
                # attach color to both the ordinal phrase and the canonical entity for robust retrieval
                extra_relations.append(self._create_relation(ord_ent.id, "has_color", target_color_id, 0.92, [text]))
                extra_relations.append(self._create_relation(canonical_id, "has_color", target_color_id, 0.92, [text]))

        return extra_entities, extra_relations

    def _find_group_entity_by_base(self, base: str) -> Optional[Entity]:
        """Find a previously stored GROUP_ENTITY whose attributes.base matches the provided base.
        Returns the highest-confidence match or None.
        """
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        # simple LIKE search over JSON text
        like_pattern = f'%"base": "{base}"%'
        cursor.execute('''
            SELECT id, text, label, confidence, attributes, session_id, timestamp
            FROM entities
            WHERE label = 'GROUP_ENTITY' AND attributes LIKE ?
            ORDER BY confidence DESC
            LIMIT 1
        ''', (like_pattern,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        try:
            return Entity(
                id=row[0],
                text=row[1],
                label=row[2],
                confidence=row[3],
                attributes=json.loads(row[4]) if row[4] else {},
                session_id=row[5],
                timestamp=datetime.fromisoformat(row[6])
            )
        except Exception:
            return None
    
    def _extract_entities(self, doc, text: str) -> List[Entity]:
        """Extract named entities from spaCy doc"""
        entities = []
        
        for ent in doc.ents:
            # Calculate confidence based on entity type and context
            confidence = self._calculate_entity_confidence(ent, doc)
            
            entity = Entity(
                id=StableId.entity(ent.text, ent.label_),
                text=ent.text,
                label=ent.label_,
                confidence=confidence,
                attributes={
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_) or ent.label_
                },
                session_id="",  # Will be set when added to session
                timestamp=datetime.now()
            )
            entities.append(entity)
        
        return entities
    
    def _extract_relations(self, doc, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using pattern matching and dependency parsing"""
        relations = []
        
        # Pattern-based relation extraction
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                import re
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    subject_text = match.group(1).strip()
                    object_text = match.group(2).strip()
                    
                    # Find corresponding entities
                    subject_entity = self._find_entity_by_text(subject_text, entities)
                    object_entity = self._find_entity_by_text(object_text, entities)
                    
                    if subject_entity and object_entity:
                        relation = Relation(
                            id=StableId.relation(subject_entity.id, relation_type, object_entity.id),
                            subject_id=subject_entity.id,
                            predicate=relation_type,
                            object_id=object_entity.id,
                            confidence=0.8,  # Pattern-based confidence
                            evidence=[text],
                            session_id="",  # Will be set when added to session
                            timestamp=datetime.now()
                        )
                        relations.append(relation)
        
        # Dependency-based relation extraction
        dependency_relations = self._extract_dependency_relations(doc, entities)
        relations.extend(dependency_relations)
        
        return relations
    
    def _extract_dependency_relations(self, doc, entities: List[Entity]) -> List[Relation]:
        """Extract relations based on syntactic dependencies"""
        relations = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                # Find head and dependent entities
                head_entity = self._find_entity_by_token(token.head, entities)
                dep_entity = self._find_entity_by_token(token, entities)
                
                if head_entity and dep_entity:
                    # Determine relation type based on dependency
                    if token.dep_ == 'nsubj':
                        predicate = 'subject_of'
                    elif token.dep_ == 'dobj':
                        predicate = 'object_of'
                    else:
                        predicate = 'related_to'
                    
                    relation = Relation(
                        id=StableId.relation(head_entity.id, predicate, dep_entity.id),
                        subject_id=head_entity.id,
                        predicate=predicate,
                        object_id=dep_entity.id,
                        confidence=0.6,  # Dependency-based confidence
                        evidence=[token.sent.text],
                        session_id="",
                        timestamp=datetime.now()
                    )
                    relations.append(relation)
        
        return relations
    
    def _calculate_entity_confidence(self, ent, doc) -> float:
        """Calculate confidence score for extracted entity"""
        confidence = 0.5  # Base confidence
        
        # Boost for specific entity types
        high_confidence_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']
        if ent.label_ in high_confidence_types:
            confidence += 0.3
        
        # Boost for capitalized entities
        if ent.text[0].isupper():
            confidence += 0.1
        
        # Boost for longer entities
        if len(ent.text) > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by text match"""
        text_lower = text.lower()
        for entity in entities:
            if entity.text.lower() == text_lower or text_lower in entity.text.lower():
                return entity
        return None
    
    def _find_entity_by_token(self, token, entities: List[Entity]) -> Optional[Entity]:
        """Find entity that contains the given token"""
        for entity in entities:
            if token.text in entity.text:
                return entity
        return None
    
    def add_claim(self, claim: str, evidence: List[str], confidence: float, session_id: str):
        """Add a verified claim to the knowledge graph"""
        logger.info(f"Adding claim to KG: {claim[:50]}...")
        
        # Extract entities and relations from claim
        entities, relations = self.extract_entities_and_relations(claim)
        
        # Set session ID for entities and relations
        for entity in entities:
            entity.session_id = session_id
        for relation in relations:
            relation.session_id = session_id
        
        # Store in database
        self._store_entities(entities)
        self._store_relations(relations)
        
        # Create claim object
        claim_obj = Claim(
            id=f"claim_{hash(claim)}_{session_id}",
            text=claim,
            entities=[e.id for e in entities],
            relations=[r.id for r in relations],
            confidence=confidence,
            evidence=evidence,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
        self._store_claim(claim_obj)
        
        # Update in-memory graph
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = nx.MultiDiGraph()
        
        graph = self.session_graphs[session_id]
        
        # Add entities as nodes
        for entity in entities:
            graph.add_node(entity.id, **asdict(entity))
        
        # Add relations as edges
        for relation in relations:
            graph.add_edge(
                relation.subject_id,
                relation.object_id,
                key=relation.predicate,
                **asdict(relation)
            )
        
        logger.info(f"Claim added successfully with {len(entities)} entities and {len(relations)} relations")
    
    def query_knowledge_graph(self, query: str, session_id: str) -> Dict[str, Any]:
        """Query the knowledge graph for information"""
        logger.info(f"Querying KG: {query[:50]}...")
        
        if session_id not in self.session_graphs:
            return {"results": [], "message": "No knowledge graph found for session"}
        
        graph = self.session_graphs[session_id]
        
        # Extract query entities
        query_doc = self.nlp(query)
        query_entities = [ent.text for ent in query_doc.ents]
        
        results = []
        
        # Find matching nodes
        for node_id, node_data in graph.nodes(data=True):
            if any(qe.lower() in node_data.get('text', '').lower() for qe in query_entities):
                # Get connected information
                neighbors = list(graph.neighbors(node_id))
                edges = list(graph.edges(node_id, data=True))
                
                results.append({
                    "entity": node_data,
                    "connections": len(neighbors),
                    "relations": [{"target": edge[1], "relation": edge[2].get('predicate', 'unknown')} 
                                for edge in edges]
                })
        
        return {
            "results": results,
            "total_found": len(results),
            "query_entities": query_entities
        }
    
    def check_contradiction(self, claim: str, session_id: str) -> Dict[str, Any]:
        """Check if claim contradicts existing knowledge"""
        logger.info(f"Checking contradiction for: {claim[:50]}...")
        
        if session_id not in self.session_graphs:
            return {"contradictions": [], "status": "PASS"}
        
        # Extract entities from new claim
        new_entities, new_relations = self.extract_entities_and_relations(claim)
        
        contradictions = []
        
        # Check against existing claims
        existing_claims = self._get_session_claims(session_id)
        
        for existing_claim in existing_claims:
            # Calculate semantic similarity
            similarity = self._calculate_semantic_similarity(claim, existing_claim['text'])
            
            if similarity > 0.7:  # High similarity but different claims
                # Check for contradictory patterns
                if self._detect_contradiction_patterns(claim, existing_claim['text']):
                    contradictions.append({
                        "existing_claim": existing_claim['text'],
                        "contradiction_type": "semantic_contradiction",
                        "confidence": similarity
                    })
        
        status = "FAIL" if contradictions else "PASS"
        
        return {
            "contradictions": contradictions,
            "status": status,
            "conflicting_claims": [c['existing_claim'] for c in contradictions]
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (cached embeddings)"""
        v1 = self._embed_text(text1)
        v2 = self._embed_text(text2)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        return float(np.dot(v1, v2) / denom)

    def _embed_text(self, text: str) -> np.ndarray:
        t = text or ""
        cached = self._embedding_cache.get(t)
        if cached is not None:
            return cached
        emb = self.sentence_model.encode([t])[0]
        self._embedding_cache[t] = emb
        return emb

    def _detect_contradiction_patterns(self, claim1: str, claim2: str) -> bool:
        """Detect contradictory patterns between claims"""
        # Simple contradiction detection
        contradiction_pairs = [
            ('is', 'is not'), ('has', 'does not have'), ('can', 'cannot'),
            ('true', 'false'), ('yes', 'no'), ('present', 'absent')
        ]
        
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        for pos, neg in contradiction_pairs:
            if pos in claim1_lower and neg in claim2_lower:
                return True
            if neg in claim1_lower and pos in claim2_lower:
                return True
        
        return False
    
    def _store_entities(self, entities: List[Entity]):
        """Store entities in database"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        for entity in entities:
            cursor.execute('''
                INSERT OR REPLACE INTO entities 
                (id, text, label, confidence, attributes, session_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity.id, entity.text, entity.label, entity.confidence,
                json.dumps(entity.attributes), entity.session_id,
                entity.timestamp.isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _store_relations(self, relations: List[Relation]):
        """Store relations in database"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        for relation in relations:
            cursor.execute('''
                INSERT OR REPLACE INTO relations 
                (id, subject_id, predicate, object_id, confidence, evidence, session_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                relation.id, relation.subject_id, relation.predicate, relation.object_id,
                relation.confidence, json.dumps(relation.evidence), relation.session_id,
                relation.timestamp.isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _store_claim(self, claim: Claim):
        """Store claim in database"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO claims 
            (id, text, entities, relations, confidence, evidence, session_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            claim.id, claim.text, json.dumps(claim.entities), json.dumps(claim.relations),
            claim.confidence, json.dumps(claim.evidence), claim.session_id,
            claim.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _get_session_claims(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all claims for a session"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT text, confidence, evidence FROM claims WHERE session_id = ?
        ''', (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"text": r[0], "confidence": r[1], "evidence": json.loads(r[2])} for r in results]
    
    def query_entity_knowledge(self, entity_names: List[str], context: str) -> Dict[str, Any]:
        """Query all stored knowledge about specific entities.
        
        Args:
            entity_names: List of entity names to search for ["car", "person", "building"]
            context: Context string to help with disambiguation ("image_description", "conversation", etc.)
        
        Returns:
            Dict with entities, relationships, and related claims about those entities
            {
                'entities': {'car': {'attributes': {...}, 'confidence': 0.9, ...}},
                'relationships': [{'subject': 'car', 'predicate': 'has_color', 'object': 'red', ...}],
                'related_claims': [{'text': 'The car is red', 'confidence': 0.9, ...}]
            }
        """
        logger.info(f"Querying entity knowledge for: {entity_names}")
        
        # Expand search names with canonicalization for ordinal phrases and base/plural variants
        try:
            search_names = self._expand_canonical_search_names(entity_names)
        except Exception:
            search_names = entity_names[:]
        
        results: Dict[str, Any] = {
            'entities': {},
            'relationships': [],
            'related_claims': []
        }
        
        for entity_name in search_names:
            matching_entities = self._find_entities_by_name(entity_name)
            
            for entity in matching_entities:
                # Store entity details keyed by the requested (possibly expanded) name
                results['entities'][entity_name] = {
                    'id': entity.id,
                    'text': entity.text,
                    'label': entity.label,
                    'attributes': entity.attributes,
                    'confidence': entity.confidence,
                    'session_id': entity.session_id,
                    'timestamp': entity.timestamp.isoformat()
                }
                
                # Get relationships involving this entity
                relationships = self._get_entity_relationships(entity.id)
                results['relationships'].extend(relationships)
                
                # Get claims mentioning this entity
                claims = self._get_claims_with_entity(entity.id)
                results['related_claims'].extend(claims)
        
        logger.info(f"Found knowledge for {len(results['entities'])} entities")
        return results

    def _expand_canonical_search_names(self, entity_names: List[str]) -> List[str]:
        """Return an expanded list of names including canonical aliases for ordinal phrases
        and base/plural forms to improve recall when querying the KG.
        """
        expanded: List[str] = []
        seen = set()
        ord_pattern = re.compile(r"\b(?:the\s+)?((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)))\s+([A-Za-z]+)\b", re.IGNORECASE)
        for name in entity_names or []:
            if not name:
                continue
            expanded.append(name)
            seen.add(name.lower())
            m = ord_pattern.search(name)
            if m:
                idx = self._ordinal_word_to_index(m.group(1))
                base = self._simple_lemma(m.group(2))
                if idx:
                    canon = f"{base}#{idx}"
                    if canon.lower() not in seen:
                        expanded.append(canon)
                        seen.add(canon.lower())
                # also add base and plural form
                base_singular = base
                base_plural = base + 's'
                if base_singular not in seen:
                    expanded.append(base_singular)
                    seen.add(base_singular)
                if base_plural not in seen:
                    expanded.append(base_plural)
                    seen.add(base_plural)
        return expanded or (entity_names or [])

    def add_attribute_facts_from_claim(self, claim: str, session_id: str):
        """Persist attribute-level facts (e.g., colors, ordinal aliasing, group membership) from a claim.
        This stores only selected entity types and relations and is intended to be called after
        an external factuality PASS, even if the full claim is not committed.
        """
        try:
            entities, relations = self.extract_entities_and_relations(claim)
        except Exception as e:
            logger.warning(f"Attribute extraction failed: {e}")
            return
        # Filter entities to relevant types
        keep_labels = {"ATTRIBUTE_VALUE", "ORDINAL_ENTITY", "CANONICAL_ENTITY", "GROUP_ENTITY"}
        entities_kept = [e for e in entities if e.label in keep_labels]
        # Filter relations to relevant predicates
        keep_pred = {"has_color", "member_of", "alias_of"}
        relations_kept = [r for r in relations if r.predicate in keep_pred]
        # Set session and persist
        for e in entities_kept:
            e.session_id = session_id
        for r in relations_kept:
            r.session_id = session_id
        self._store_entities(entities_kept)
        self._store_relations(relations_kept)
        # Update in-memory graph
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = nx.MultiDiGraph()
        graph = self.session_graphs[session_id]
        for e in entities_kept:
            graph.add_node(e.id, **asdict(e))
        for r in relations_kept:
            graph.add_edge(r.subject_id, r.object_id, key=r.predicate, **asdict(r))
        logger.info(f"Attribute facts added to KG: {len(entities_kept)} entities, {len(relations_kept)} relations")

    def _find_entities_by_name(self, entity_name: str) -> List[Entity]:
        """Find entities by name from database (case-insensitive partial match)."""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, text, label, confidence, attributes, session_id, timestamp
            FROM entities 
            WHERE LOWER(text) LIKE LOWER(?)
            ORDER BY confidence DESC
            LIMIT 10
        ''', (f'%{entity_name}%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        entities: List[Entity] = []
        for row in rows:
            entity = Entity(
                id=row[0],
                text=row[1],
                label=row[2],
                confidence=row[3],
                attributes=json.loads(row[4]) if row[4] else {},
                session_id=row[5],
                timestamp=datetime.fromisoformat(row[6])
            )
            entities.append(entity)
        
        return entities

    def _get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships where entity is subject or object, including readable entity texts."""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.subject_id, s.text as subject_text, r.predicate, r.object_id, o.text as object_text, r.confidence, r.evidence
            FROM relations r
            LEFT JOIN entities s ON r.subject_id = s.id
            LEFT JOIN entities o ON r.object_id = o.id
            WHERE r.subject_id = ? OR r.object_id = ?
        ''', (entity_id, entity_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        relationships: List[Dict[str, Any]] = []
        for row in rows:
            relationships.append({
                'subject': row[0],
                'subject_text': row[1],
                'predicate': row[2], 
                'object': row[3],
                'object_text': row[4],
                'confidence': row[5],
                'evidence': json.loads(row[6]) if row[6] else []
            })
        
        return relationships

    def _get_claims_with_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all claims that mention this entity"""
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT text, confidence, evidence
            FROM claims 
            WHERE entities LIKE ?
        ''', (f'%"{entity_id}"%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        claims: List[Dict[str, Any]] = []
        for row in rows:
            claims.append({
                'text': row[0],
                'confidence': row[1],
                'evidence': json.loads(row[2]) if row[2] else []
            })
        
        return claims

    def extract_entities_from_claim(self, claim_text: str, context: str = "") -> List[Dict[str, Any]]:
        """Extract entities from claim text and return structured data.
        
        Returns a list of entity dictionaries: 
        [{'name': 'car', 'type': 'VEHICLE', 'confidence': 0.9, 'attributes': {...}}]
        """
        entities, _ = self.extract_entities_and_relations(claim_text)
        return [{
            'name': entity.text,
            'type': entity.label,
            'confidence': entity.confidence,
            'attributes': entity.attributes
        } for entity in entities]

    def _load_session_data(self, session_id: str):
        """Load existing session data into memory"""
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = nx.MultiDiGraph()
        
        # Load entities and relations from database and rebuild graph
        graph = self.session_graphs[session_id]
        # Ensure clean slate for this load (initialize_session already reset graph, but be safe)
        try:
            graph.clear()
        except Exception:
            self.session_graphs[session_id] = nx.MultiDiGraph()
            graph = self.session_graphs[session_id]

        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()

        # Load entities for this session
        cursor.execute(
            '''
            SELECT id, text, label, confidence, attributes, session_id, timestamp
            FROM entities
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        entity_rows = cursor.fetchall()

        # Add entity nodes
        for row in entity_rows:
            try:
                attrs = json.loads(row[4]) if row[4] else {}
            except Exception:
                attrs = {}
            try:
                ts = datetime.fromisoformat(row[6]) if row[6] else datetime.now()
            except Exception:
                ts = datetime.now()
            try:
                e = Entity(
                    id=row[0],
                    text=row[1],
                    label=row[2],
                    confidence=row[3] if row[3] is not None else 0.5,
                    attributes=attrs,
                    session_id=row[5] or session_id,
                    timestamp=ts,
                )
                graph.add_node(e.id, **asdict(e))
            except Exception:
                # Fallback: add minimal node
                graph.add_node(row[0], id=row[0], text=row[1], label=row[2], confidence=row[3] or 0.5, attributes=attrs, session_id=row[5] or session_id, timestamp=str(ts))

        # Load relations for this session
        cursor.execute(
            '''
            SELECT id, subject_id, predicate, object_id, confidence, evidence, session_id, timestamp
            FROM relations
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        rel_rows = cursor.fetchall()

        # Add relation edges
        for row in rel_rows:
            try:
                evidence = json.loads(row[5]) if row[5] else []
            except Exception:
                evidence = []
            try:
                ts = datetime.fromisoformat(row[7]) if row[7] else datetime.now()
            except Exception:
                ts = datetime.now()
            try:
                r = Relation(
                    id=row[0],
                    subject_id=row[1],
                    predicate=row[2],
                    object_id=row[3],
                    confidence=row[4] if row[4] is not None else 0.5,
                    evidence=evidence,
                    session_id=row[6] or session_id,
                    timestamp=ts,
                )
                graph.add_edge(r.subject_id, r.object_id, key=r.predicate, **asdict(r))
            except Exception:
                # Fallback: add minimal edge
                graph.add_edge(row[1], row[3], key=(row[2] or 'related_to'), id=row[0], subject_id=row[1], predicate=row[2] or 'related_to', object_id=row[3], confidence=row[4] or 0.5, evidence=evidence, session_id=row[6] or session_id, timestamp=str(ts))

        conn.close()
        logger.info(f"Loaded session KG '{session_id}': {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    def get_graph_size(self, session_id: str) -> Dict[str, int]:
        """Get size statistics of the knowledge graph"""
        if session_id not in self.session_graphs:
            return {"nodes": 0, "edges": 0}
        
        graph = self.session_graphs[session_id]
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges()
        }
    
    def export_session_graph(self, session_id: str) -> Dict[str, Any]:
        """Export session knowledge graph"""
        if session_id not in self.session_graphs:
            return {}
        
        graph = self.session_graphs[session_id]
        
        return {
            "nodes": dict(graph.nodes(data=True)),
            "edges": list(graph.edges(data=True)),
            "statistics": self.get_graph_size(session_id)
        }

    def run_canonical_consolidation(self, similarity_threshold: float = 0.86, alias_bonus: float = 0.05, batch_limit: int = 5000) -> Dict[str, Any]:
        """Batch community consolidation across sessions.
        - Build a similarity graph (embeddings + alias_of edges) and take connected components as communities.
        - Assign canonical IDs to each community, populate canonical_entities and entity_canonical_map tables.
        - Update in-memory graphs by annotating nodes with canonical_id when available.
        """
        logger.info("Starting canonical consolidation across sessions...")
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        # Fetch a manageable batch of entities
        cursor.execute('''
            SELECT id, text, label, confidence FROM entities
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (batch_limit,))
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            logger.info("No entities found for consolidation.")
            return {"communities": 0, "entities_mapped": 0, "canonical_created": 0}
        eids = [r[0] for r in rows]
        texts = [str(r[1] or "").strip().lower() for r in rows]
        labels = [str(r[2] or "").strip() for r in rows]
        confs = [float(r[3] or 0.5) for r in rows]
        # Embeddings (normalized)
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v / (n + 1e-12)
        uniq_cache: Dict[str, np.ndarray] = {}
        vecs: List[np.ndarray] = []
        for t in texts:
            if t in uniq_cache:
                vecs.append(uniq_cache[t])
            else:
                v = _norm(self._embed_text(t))
                uniq_cache[t] = v
                vecs.append(v)
        V = np.vstack(vecs)
        # alias_of links
        cursor.execute("SELECT subject_id, object_id FROM relations WHERE predicate = 'alias_of'")
        alias_edges = set((a, b) for a, b in cursor.fetchall() if a and b)
        # Build similarity graph
        G = nx.Graph()
        for eid in eids:
            G.add_node(eid)
        for i in range(len(eids)):
            vi = V[i]
            for j in range(i + 1, len(eids)):
                vj = V[j]
                sim = float(np.dot(vi, vj))
                if (eids[i], eids[j]) in alias_edges or (eids[j], eids[i]) in alias_edges:
                    sim += alias_bonus
                if sim >= similarity_threshold:
                    G.add_edge(eids[i], eids[j], weight=sim)
        communities = list(nx.connected_components(G))
        now = datetime.now().isoformat()
        canonical_created = 0
        mapped = 0
        # choose representative by highest confidence
        def _choose_rep(ids: List[str]) -> Tuple[str, str]:
            best_idx = 0
            best_conf = -1.0
            for eid in ids:
                try:
                    k = eids.index(eid)
                except ValueError:
                    continue
                if confs[k] > best_conf:
                    best_conf = confs[k]
                    best_idx = k
            return texts[best_idx], labels[best_idx]
        # Insert canonical entities and mapping
        for comp in communities:
            ids = list(comp)
            rep_text, rep_label = _choose_rep(ids)
            canonical_id = StableId.canonical(rep_text, rep_label)
            aliases = sorted(list({texts[eids.index(e)] for e in ids if e in eids}))
            cursor.execute('''
                INSERT OR IGNORE INTO canonical_entities (canonical_id, label, name, aliases, created_ts)
                VALUES (?, ?, ?, ?, ?)
            ''', (canonical_id, rep_label, rep_text, json.dumps(aliases), now))
            if getattr(cursor, 'rowcount', -1) and cursor.rowcount > 0:
                canonical_created += int(cursor.rowcount)
            # Map all members
            for eid in ids:
                cursor.execute('''
                    INSERT OR REPLACE INTO entity_canonical_map (entity_id, canonical_id, score, mapped_ts)
                    VALUES (?, ?, ?, ?)
                ''', (eid, canonical_id, 1.0, now))
                mapped += 1
        conn.commit()
        # Update in-memory graphs with canonical_id attribute
        for sid, g in (self.session_graphs or {}).items():
            for node_id in list(g.nodes()):
                try:
                    cursor.execute('SELECT canonical_id FROM entity_canonical_map WHERE entity_id = ?', (node_id,))
                    r = cursor.fetchone()
                    if r and r[0]:
                        g.nodes[node_id]['canonical_id'] = r[0]
                except Exception:
                    continue
        conn.commit()
        conn.close()
        logger.info(f"Canonical consolidation complete: communities={len(communities)}, mapped={mapped}")
        return {"communities": len(communities), "entities_mapped": mapped, "canonical_created": canonical_created}

    def consolidate_relations_canonical(self) -> Dict[str, Any]:
        """Aggregate relations in canonical space and populate canonical_relations and relation_canonical_map.
        Groups by (subject_canonical_id, normalized_predicate, object_canonical_id) and computes support/confidence.
        """
        logger.info("Consolidating relations into canonical space...")
        conn = sqlite3.connect(KG_DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.id, r.subject_id, r.predicate, r.object_id,
                   COALESCE(m1.canonical_id, ''), COALESCE(m2.canonical_id, ''), r.confidence
            FROM relations r
            LEFT JOIN entity_canonical_map m1 ON r.subject_id = m1.entity_id
            LEFT JOIN entity_canonical_map m2 ON r.object_id = m2.entity_id
        ''')
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return {"canonical_relations": 0, "mapped": 0}
        def _normalize_pred(p: str) -> str:
            p = (p or '').strip().lower()
            return self.predicate_aliases.get(p, p) if hasattr(self, 'predicate_aliases') else p
        triple_confs: Dict[Tuple[str, str, str], List[float]] = {}
        rel_to_canon_key: Dict[str, Tuple[str, str, str]] = {}
        for rid, sid, pred, oid, scid, ocid, conf in rows:
            if not scid or not ocid:
                continue
            npred = _normalize_pred(pred)
            key = (scid, npred, ocid)
            triple_confs.setdefault(key, []).append(float(conf or 0.5))
            rel_to_canon_key[rid] = key
        now = datetime.now().isoformat()
        created = 0
        mapped = 0
        for (scid, npred, ocid), confs in triple_confs.items():
            cid = StableId.relation(scid, npred, ocid)
            support = len(confs)
            avg_conf = float(sum(confs) / max(1, support))
            cursor.execute('''
                INSERT OR IGNORE INTO canonical_relations (id, subject_canonical_id, predicate, object_canonical_id, support_count, confidence, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cid, scid, npred, ocid, support, avg_conf, now))
            if getattr(cursor, 'rowcount', -1) and cursor.rowcount > 0:
                created += int(cursor.rowcount)
        for rid, (scid, npred, ocid) in rel_to_canon_key.items():
            cid = StableId.relation(scid, npred, ocid)
            cursor.execute('''
                INSERT OR REPLACE INTO relation_canonical_map (relation_id, canonical_relation_id, mapped_ts)
                VALUES (?, ?, ?)
            ''', (rid, cid, now))
            mapped += 1
        conn.commit()
        conn.close()
        logger.info(f"Canonical relation consolidation complete: created={created}, mapped={mapped}")
        return {"canonical_relations": created, "mapped": mapped}
