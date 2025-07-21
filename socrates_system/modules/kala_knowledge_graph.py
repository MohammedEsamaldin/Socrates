"""
KALA Knowledge Graph Manager - Knowledge-Augmented Language Model Adaptation inspired
Implements sophisticated knowledge graph with domain adaptation and entity-relation extraction
"""
import logging
import asyncio
import json
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pickle
import os
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_manager import get_llm_manager, LLMTaskType
from modules.advanced_claim_extractor import AdvancedClaim, Entity, Relationship
from utils.logger import setup_logger

logger = setup_logger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge in the graph"""
    FACTUAL = "factual"           # Direct factual knowledge
    RELATIONAL = "relational"     # Relationships between entities
    TEMPORAL = "temporal"         # Time-based knowledge
    CAUSAL = "causal"            # Cause-effect relationships
    CONTEXTUAL = "contextual"     # Context-dependent knowledge
    INFERENTIAL = "inferential"   # Inferred knowledge
    CONTRADICTORY = "contradictory"  # Conflicting information

class ConfidenceLevel(Enum):
    """Confidence levels for knowledge"""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"           # 0.7-0.9
    MEDIUM = "medium"       # 0.5-0.7
    LOW = "low"            # 0.3-0.5
    VERY_LOW = "very_low"  # 0.0-0.3

@dataclass
class KnowledgeNode:
    """Node in the knowledge graph representing an entity or concept"""
    id: str
    label: str
    node_type: str
    attributes: Dict[str, Any]
    confidence: float
    sources: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    domain_context: str = ""
    creation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    verification_status: str = "unverified"
    embedding: Optional[List[float]] = None

@dataclass
class KnowledgeEdge:
    """Edge in the knowledge graph representing a relationship"""
    source_id: str
    target_id: str
    relation_type: str
    attributes: Dict[str, Any]
    confidence: float
    knowledge_type: KnowledgeType
    sources: List[str] = field(default_factory=list)
    temporal_info: Optional[str] = None
    context: str = ""
    creation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DomainKnowledge:
    """Domain-specific knowledge structure following KALA methodology"""
    domain: str
    entities: Dict[str, KnowledgeNode]
    relations: Dict[str, KnowledgeEdge]
    facts: List[str]
    confidence_distribution: Dict[ConfidenceLevel, int]
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class KALAKnowledgeGraph:
    """
    KALA-inspired Knowledge Graph Manager
    Implements Knowledge-Augmented Language Model Adaptation methodology
    with sophisticated entity-relation extraction and domain adaptation
    """
    
    def __init__(self, storage_path: str = "knowledge_graphs"):
        """Initialize KALA Knowledge Graph Manager"""
        self.llm_manager = get_llm_manager()
        self.storage_path = storage_path
        self.graphs = {}  # session_id -> NetworkX graph
        self.domain_knowledge = {}  # domain -> DomainKnowledge
        self.global_graph = nx.MultiDiGraph()
        
        # KALA-specific components
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self.adaptation_cache = {}
        
        # Initialize storage
        self._init_storage()
        
        # Load existing knowledge
        self._load_existing_knowledge()
        
        logger.info("KALAKnowledgeGraph initialized with domain adaptation capabilities")
    
    def _init_storage(self):
        """Initialize storage directories"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "domains"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "embeddings"), exist_ok=True)
    
    def _load_existing_knowledge(self):
        """Load existing knowledge graphs and domain knowledge"""
        try:
            # Load global graph
            global_path = os.path.join(self.storage_path, "global_graph.pkl")
            if os.path.exists(global_path):
                with open(global_path, 'rb') as f:
                    self.global_graph = pickle.load(f)
                logger.info(f"Loaded global graph with {self.global_graph.number_of_nodes()} nodes")
            
            # Load domain knowledge
            domains_path = os.path.join(self.storage_path, "domains")
            for domain_file in os.listdir(domains_path):
                if domain_file.endswith('.pkl'):
                    domain_name = domain_file[:-4]
                    with open(os.path.join(domains_path, domain_file), 'rb') as f:
                        self.domain_knowledge[domain_name] = pickle.load(f)
                    logger.info(f"Loaded domain knowledge for: {domain_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to load existing knowledge: {e}")
    
    def initialize_session(self, session_id: str, domain: str = "general") -> str:
        """Initialize a new session knowledge graph"""
        self.graphs[session_id] = nx.MultiDiGraph()
        
        # Initialize domain knowledge if not exists
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = DomainKnowledge(
                domain=domain,
                entities={},
                relations={},
                facts=[],
                confidence_distribution={level: 0 for level in ConfidenceLevel}
            )
        
        logger.info(f"Initialized session {session_id} for domain: {domain}")
        return session_id
    
    async def integrate_claim_knowledge(self, session_id: str, claim: AdvancedClaim, 
                                     verification_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Integrate claim knowledge using KALA methodology
        Modulates hidden representations with domain knowledge
        """
        logger.info(f"Integrating claim knowledge: {claim.text[:50]}...")
        
        if session_id not in self.graphs:
            self.initialize_session(session_id)
        
        graph = self.graphs[session_id]
        integration_result = {
            "nodes_added": 0,
            "edges_added": 0,
            "conflicts_resolved": 0,
            "knowledge_updated": False,
            "adaptation_applied": False
        }
        
        try:
            # Stage 1: Extract and enhance entities using LLM
            enhanced_entities = await self._extract_enhanced_entities(claim)
            
            # Stage 2: Extract and enhance relationships using LLM
            enhanced_relationships = await self._extract_enhanced_relationships(claim, enhanced_entities)
            
            # Stage 3: Apply KALA domain adaptation
            adapted_knowledge = await self._apply_domain_adaptation(
                enhanced_entities, enhanced_relationships, claim.context
            )
            
            # Stage 4: Integrate into knowledge graph
            integration_result = await self._integrate_into_graph(
                session_id, adapted_knowledge, verification_result
            )
            
            # Stage 5: Update domain knowledge
            await self._update_domain_knowledge(claim.context, adapted_knowledge)
            
            # Stage 6: Resolve conflicts and maintain consistency
            conflicts_resolved = await self._resolve_knowledge_conflicts(session_id, claim)
            integration_result["conflicts_resolved"] = conflicts_resolved
            
            logger.info(f"Knowledge integration complete: {integration_result}")
            return integration_result
            
        except Exception as e:
            logger.error(f"Knowledge integration failed: {e}")
            return integration_result
    
    async def _extract_enhanced_entities(self, claim: AdvancedClaim) -> List[KnowledgeNode]:
        """Extract enhanced entities with LLM-powered attribute extraction"""
        enhanced_entities = []
        
        for entity in claim.entities:
            # Use LLM to extract additional attributes and context
            context = {
                "entity_text": entity.text,
                "entity_type": entity.entity_type,
                "claim_context": claim.text,
                "existing_attributes": entity.attributes
            }
            
            response = await self.llm_manager.process_request({
                "task_type": LLMTaskType.KNOWLEDGE_INTEGRATION,
                "prompt": f"Extract comprehensive attributes and context for entity: {entity.text}",
                "context": context
            })
            
            # Parse enhanced attributes
            enhanced_attributes = entity.attributes.copy()
            if response.structured_output:
                enhanced_attributes.update(response.structured_output.get("attributes", {}))
            
            # Create enhanced knowledge node
            knowledge_node = KnowledgeNode(
                id=self._generate_entity_id(entity.text, entity.entity_type),
                label=entity.text,
                node_type=entity.entity_type,
                attributes=enhanced_attributes,
                confidence=entity.confidence,
                sources=[claim.text],
                aliases=entity.aliases,
                domain_context=claim.context
            )
            
            enhanced_entities.append(knowledge_node)
        
        return enhanced_entities
    
    async def _extract_enhanced_relationships(self, claim: AdvancedClaim, 
                                           entities: List[KnowledgeNode]) -> List[KnowledgeEdge]:
        """Extract enhanced relationships with LLM-powered analysis"""
        enhanced_relationships = []
        
        # Extract relationships from claim relationships
        for relationship in claim.relationships:
            knowledge_edge = KnowledgeEdge(
                source_id=self._find_entity_id(relationship.source, entities),
                target_id=self._find_entity_id(relationship.target, entities),
                relation_type=relationship.relation,
                attributes=relationship.attributes,
                confidence=relationship.confidence,
                knowledge_type=self._classify_knowledge_type(relationship.relation),
                sources=[claim.text],
                temporal_info=relationship.temporal_info,
                context=relationship.context
            )
            enhanced_relationships.append(knowledge_edge)
        
        # Use LLM to discover additional implicit relationships
        if len(entities) > 1:
            entity_pairs = [(e1, e2) for i, e1 in enumerate(entities) 
                           for e2 in entities[i+1:]]
            
            for e1, e2 in entity_pairs:
                implicit_relations = await self._discover_implicit_relationships(e1, e2, claim)
                enhanced_relationships.extend(implicit_relations)
        
        return enhanced_relationships
    
    async def _discover_implicit_relationships(self, entity1: KnowledgeNode, 
                                            entity2: KnowledgeNode, 
                                            claim: AdvancedClaim) -> List[KnowledgeEdge]:
        """Discover implicit relationships between entities using LLM"""
        context = {
            "entity1": {"text": entity1.label, "type": entity1.node_type, "attributes": entity1.attributes},
            "entity2": {"text": entity2.label, "type": entity2.node_type, "attributes": entity2.attributes},
            "claim_context": claim.text,
            "domain": claim.context
        }
        
        response = await self.llm_manager.extract_relationships(
            claim.text, [entity1.label, entity2.label], context
        )
        
        implicit_relations = []
        if response.structured_output and 'relationships' in response.structured_output:
            for rel_data in response.structured_output['relationships']:
                if rel_data.get('confidence', 0) > 0.6:  # Only high-confidence implicit relations
                    knowledge_edge = KnowledgeEdge(
                        source_id=entity1.id,
                        target_id=entity2.id,
                        relation_type=rel_data.get('relation', 'related_to'),
                        attributes={"implicit": True, "discovery_method": "LLM_inference"},
                        confidence=rel_data.get('confidence', 0.7),
                        knowledge_type=KnowledgeType.INFERENTIAL,
                        sources=[claim.text],
                        context=rel_data.get('context', '')
                    )
                    implicit_relations.append(knowledge_edge)
        
        return implicit_relations
    
    async def _apply_domain_adaptation(self, entities: List[KnowledgeNode], 
                                     relationships: List[KnowledgeEdge], 
                                     domain_context: str) -> Dict[str, Any]:
        """Apply KALA domain adaptation to modulate knowledge representations"""
        logger.debug("Applying KALA domain adaptation")
        
        # Identify domain from context
        domain = self._identify_domain(domain_context)
        
        # Get domain-specific knowledge
        domain_knowledge = self.domain_knowledge.get(domain, self.domain_knowledge.get("general"))
        
        # Apply domain adaptation using LLM
        adaptation_context = {
            "domain": domain,
            "entities": [{"id": e.id, "label": e.label, "type": e.node_type} for e in entities],
            "relationships": [{"source": r.source_id, "target": r.target_id, "type": r.relation_type} for r in relationships],
            "domain_knowledge": {
                "existing_entities": list(domain_knowledge.entities.keys()) if domain_knowledge else [],
                "existing_relations": list(domain_knowledge.relations.keys()) if domain_knowledge else [],
                "domain_facts": domain_knowledge.facts if domain_knowledge else []
            }
        }
        
        response = await self.llm_manager.integrate_knowledge(
            {"entities": entities, "relationships": relationships},
            domain_knowledge.__dict__ if domain_knowledge else {},
            adaptation_context
        )
        
        # Parse adaptation results
        adapted_knowledge = {
            "entities": entities,
            "relationships": relationships,
            "domain": domain,
            "adaptation_applied": True,
            "adaptation_reasoning": response.reasoning or "Domain adaptation applied"
        }
        
        if response.structured_output:
            # Update entities with domain-adapted attributes
            adapted_entities = response.structured_output.get("updated_entities", {})
            for entity in entities:
                if entity.id in adapted_entities:
                    entity.attributes.update(adapted_entities[entity.id])
            
            # Update relationships with domain context
            adapted_relations = response.structured_output.get("updated_relationships", {})
            for relationship in relationships:
                rel_key = f"{relationship.source_id}_{relationship.target_id}_{relationship.relation_type}"
                if rel_key in adapted_relations:
                    relationship.attributes.update(adapted_relations[rel_key])
            
            adapted_knowledge["confidence_updates"] = response.structured_output.get("confidence_updates", {})
        
        return adapted_knowledge
    
    def _identify_domain(self, context: str) -> str:
        """Identify domain from context"""
        context_lower = context.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "science": ["research", "study", "experiment", "hypothesis", "theory"],
            "technology": ["software", "hardware", "computer", "algorithm", "data"],
            "medicine": ["health", "medical", "disease", "treatment", "patient"],
            "history": ["historical", "past", "ancient", "century", "era"],
            "politics": ["government", "policy", "election", "political", "law"],
            "economics": ["economic", "market", "financial", "business", "trade"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                return domain
        
        return "general"
    
    async def _integrate_into_graph(self, session_id: str, adapted_knowledge: Dict[str, Any], 
                                  verification_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Integrate adapted knowledge into the session graph"""
        graph = self.graphs[session_id]
        integration_result = {"nodes_added": 0, "edges_added": 0, "knowledge_updated": True}
        
        # Add entities as nodes
        for entity in adapted_knowledge["entities"]:
            if not graph.has_node(entity.id):
                graph.add_node(entity.id, **entity.__dict__)
                integration_result["nodes_added"] += 1
            else:
                # Update existing node
                graph.nodes[entity.id].update(entity.__dict__)
        
        # Add relationships as edges
        for relationship in adapted_knowledge["relationships"]:
            if graph.has_node(relationship.source_id) and graph.has_node(relationship.target_id):
                graph.add_edge(
                    relationship.source_id, 
                    relationship.target_id, 
                    key=relationship.relation_type,
                    **relationship.__dict__
                )
                integration_result["edges_added"] += 1
        
        # Update global graph
        self.global_graph = nx.compose(self.global_graph, graph)
        
        return integration_result
    
    async def _update_domain_knowledge(self, domain_context: str, adapted_knowledge: Dict[str, Any]):
        """Update domain-specific knowledge following KALA methodology"""
        domain = adapted_knowledge["domain"]
        
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = DomainKnowledge(
                domain=domain,
                entities={},
                relations={},
                facts=[],
                confidence_distribution={level: 0 for level in ConfidenceLevel}
            )
        
        domain_knowledge = self.domain_knowledge[domain]
        
        # Update entities
        for entity in adapted_knowledge["entities"]:
            domain_knowledge.entities[entity.id] = entity
        
        # Update relationships
        for relationship in adapted_knowledge["relationships"]:
            rel_key = f"{relationship.source_id}_{relationship.target_id}_{relationship.relation_type}"
            domain_knowledge.relations[rel_key] = relationship
        
        # Update confidence distribution
        for entity in adapted_knowledge["entities"]:
            confidence_level = self._get_confidence_level(entity.confidence)
            domain_knowledge.confidence_distribution[confidence_level] += 1
        
        # Record adaptation history
        domain_knowledge.adaptation_history.append({
            "timestamp": datetime.now(),
            "entities_added": len(adapted_knowledge["entities"]),
            "relationships_added": len(adapted_knowledge["relationships"]),
            "adaptation_reasoning": adapted_knowledge.get("adaptation_reasoning", "")
        })
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _resolve_knowledge_conflicts(self, session_id: str, claim: AdvancedClaim) -> int:
        """Resolve conflicts in knowledge graph using LLM reasoning"""
        graph = self.graphs[session_id]
        conflicts_resolved = 0
        
        # Find potential conflicts
        conflicts = self._detect_conflicts(graph, claim)
        
        for conflict in conflicts:
            # Use LLM to resolve conflict
            resolution = await self._resolve_conflict_with_llm(conflict, claim)
            
            if resolution["action"] == "update":
                # Update conflicting knowledge
                self._apply_conflict_resolution(graph, resolution)
                conflicts_resolved += 1
            elif resolution["action"] == "flag":
                # Flag as contradictory knowledge
                self._flag_contradictory_knowledge(graph, conflict)
        
        return conflicts_resolved
    
    def _detect_conflicts(self, graph: nx.MultiDiGraph, claim: AdvancedClaim) -> List[Dict[str, Any]]:
        """Detect potential conflicts in the knowledge graph"""
        conflicts = []
        
        # Check for contradictory relationships
        for entity in claim.entities:
            entity_id = self._generate_entity_id(entity.text, entity.entity_type)
            if graph.has_node(entity_id):
                # Check for contradictory attributes
                existing_attrs = graph.nodes[entity_id].get("attributes", {})
                new_attrs = entity.attributes
                
                for attr, value in new_attrs.items():
                    if attr in existing_attrs and existing_attrs[attr] != value:
                        conflicts.append({
                            "type": "attribute_conflict",
                            "entity_id": entity_id,
                            "attribute": attr,
                            "existing_value": existing_attrs[attr],
                            "new_value": value,
                            "claim": claim.text
                        })
        
        return conflicts
    
    async def _resolve_conflict_with_llm(self, conflict: Dict[str, Any], claim: AdvancedClaim) -> Dict[str, Any]:
        """Use LLM to resolve knowledge conflicts"""
        context = {
            "conflict_type": conflict["type"],
            "existing_value": conflict.get("existing_value"),
            "new_value": conflict.get("new_value"),
            "claim": claim.text,
            "entity": conflict.get("entity_id")
        }
        
        response = await self.llm_manager.generate_reasoning(
            f"How should we resolve this knowledge conflict: {conflict}",
            [claim.text],
            context
        )
        
        # Parse resolution decision
        if "update" in response.content.lower():
            return {"action": "update", "reasoning": response.content}
        elif "flag" in response.content.lower():
            return {"action": "flag", "reasoning": response.content}
        else:
            return {"action": "ignore", "reasoning": response.content}
    
    def _apply_conflict_resolution(self, graph: nx.MultiDiGraph, resolution: Dict[str, Any]):
        """Apply conflict resolution to the graph"""
        # Implementation depends on specific resolution strategy
        logger.info(f"Applied conflict resolution: {resolution['reasoning']}")
    
    def _flag_contradictory_knowledge(self, graph: nx.MultiDiGraph, conflict: Dict[str, Any]):
        """Flag contradictory knowledge for human review"""
        # Add contradiction flag to node/edge
        if conflict["type"] == "attribute_conflict":
            entity_id = conflict["entity_id"]
            if graph.has_node(entity_id):
                if "contradictions" not in graph.nodes[entity_id]:
                    graph.nodes[entity_id]["contradictions"] = []
                graph.nodes[entity_id]["contradictions"].append(conflict)
    
    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        import hashlib
        return f"{entity_type}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    
    def _find_entity_id(self, entity_text: str, entities: List[KnowledgeNode]) -> str:
        """Find entity ID from entity text"""
        for entity in entities:
            if entity.label == entity_text:
                return entity.id
        return self._generate_entity_id(entity_text, "UNKNOWN")
    
    def _classify_knowledge_type(self, relation: str) -> KnowledgeType:
        """Classify knowledge type based on relation"""
        relation_lower = relation.lower()
        
        if any(word in relation_lower for word in ["cause", "lead", "result", "trigger"]):
            return KnowledgeType.CAUSAL
        elif any(word in relation_lower for word in ["before", "after", "during", "when"]):
            return KnowledgeType.TEMPORAL
        elif any(word in relation_lower for word in ["contradict", "oppose", "conflict"]):
            return KnowledgeType.CONTRADICTORY
        else:
            return KnowledgeType.RELATIONAL
    
    def get_graph_size(self, session_id: str) -> Dict[str, int]:
        """Get size statistics for session graph"""
        if session_id not in self.graphs:
            return {"nodes": 0, "edges": 0}
        
        graph = self.graphs[session_id]
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges()
        }
    
    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive domain knowledge summary"""
        if domain not in self.domain_knowledge:
            return {"domain": domain, "status": "not_found"}
        
        domain_knowledge = self.domain_knowledge[domain]
        
        return {
            "domain": domain,
            "total_entities": len(domain_knowledge.entities),
            "total_relations": len(domain_knowledge.relations),
            "total_facts": len(domain_knowledge.facts),
            "confidence_distribution": domain_knowledge.confidence_distribution,
            "adaptation_history_length": len(domain_knowledge.adaptation_history),
            "performance_metrics": domain_knowledge.performance_metrics,
            "last_updated": max([e.last_updated for e in domain_knowledge.entities.values()]) if domain_knowledge.entities else None
        }
    
    def save_knowledge(self):
        """Save all knowledge graphs and domain knowledge"""
        try:
            # Save global graph
            global_path = os.path.join(self.storage_path, "global_graph.pkl")
            with open(global_path, 'wb') as f:
                pickle.dump(self.global_graph, f)
            
            # Save domain knowledge
            for domain, knowledge in self.domain_knowledge.items():
                domain_path = os.path.join(self.storage_path, "domains", f"{domain}.pkl")
                with open(domain_path, 'wb') as f:
                    pickle.dump(knowledge, f)
            
            # Save session graphs
            for session_id, graph in self.graphs.items():
                session_path = os.path.join(self.storage_path, "sessions", f"{session_id}.pkl")
                with open(session_path, 'wb') as f:
                    pickle.dump(graph, f)
            
            logger.info("Knowledge graphs saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
    
    def query_knowledge(self, session_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query knowledge graph for relevant information"""
        if session_id not in self.graphs:
            return []
        
        graph = self.graphs[session_id]
        results = []
        
        # Simple text-based search in node labels and attributes
        query_lower = query.lower()
        
        for node_id, node_data in graph.nodes(data=True):
            if query_lower in node_data.get("label", "").lower():
                results.append({
                    "type": "entity",
                    "id": node_id,
                    "label": node_data.get("label", ""),
                    "attributes": node_data.get("attributes", {}),
                    "confidence": node_data.get("confidence", 0.0)
                })
        
        # Search in edge relationships
        for source, target, edge_data in graph.edges(data=True):
            if query_lower in edge_data.get("relation_type", "").lower():
                results.append({
                    "type": "relationship",
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relation_type", ""),
                    "confidence": edge_data.get("confidence", 0.0)
                })
        
        # Sort by confidence and limit results
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results[:max_results]
