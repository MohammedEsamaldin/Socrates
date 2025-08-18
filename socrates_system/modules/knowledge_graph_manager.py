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
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.logger import setup_logger
from ..config import KG_DATABASE_PATH, SESSION_KG_PATH, ENTITY_MODEL_NAME, NLP_MODEL_NAME

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

class KnowledgeGraphManager:
    """
    Advanced Knowledge Graph Manager for the Socrates Agent
    Handles entity extraction, relation extraction, graph construction, and querying
    """
    
    def __init__(self):
        logger.info("Initializing Knowledge Graph Manager...")
        
        try:
            # Load NLP models
            self.nlp = spacy.load(ENTITY_MODEL_NAME)
            self.sentence_model = SentenceTransformer(NLP_MODEL_NAME)
            
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
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc, text)
        
        # Extract relations
        relations = self._extract_relations(doc, text, entities)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
        return entities, relations
    
    def _extract_entities(self, doc, text: str) -> List[Entity]:
        """Extract named entities from spaCy doc"""
        entities = []
        
        for ent in doc.ents:
            # Calculate confidence based on entity type and context
            confidence = self._calculate_entity_confidence(ent, doc)
            
            entity = Entity(
                id=f"ent_{hash(ent.text)}_{ent.label_}",
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
                            id=f"rel_{hash(subject_entity.id + relation_type + object_entity.id)}",
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
                        id=f"rel_{hash(head_entity.id + predicate + dep_entity.id)}",
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
        """Calculate semantic similarity between two texts"""
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity
    
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
        
        results: Dict[str, Any] = {
            'entities': {},
            'relationships': [],
            'related_claims': []
        }
        
        for entity_name in entity_names:
            matching_entities = self._find_entities_by_name(entity_name)
            
            for entity in matching_entities:
                # Store entity details keyed by the requested name
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
        
        # Load entities and relations from database
        # Implementation would load from SQLite and rebuild graph
        pass
    
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
