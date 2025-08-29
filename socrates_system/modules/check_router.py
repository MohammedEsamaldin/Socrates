"""
Check Router Module

This module is responsible for determining the most appropriate verification method
for each claim based on its categories, available verification services, and other factors.
"""
import logging
from typing import Dict, List, Optional, Tuple, Set
from socrates_system.modules.shared_structures import ExtractedClaim, ClaimCategory, ClaimCategoryType, VerificationMethod, VerificationRoute
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class CheckRouter:
    """
    Routes claims to the most appropriate verification method based on their categories
    for MLLM hallucination detection and other factors like cost and latency requirements.
    """
    
    # Verification methods mapped to MLLM hallucination detection categories
    VERIFICATION_METHODS = {
        VerificationMethod.CROSS_MODAL: {
            'cost': 0.8,
            'latency': 1.5,
            'categories': {ClaimCategoryType.VISUAL_GROUNDING_REQUIRED}
        },
        VerificationMethod.EXTERNAL_SOURCE: {
            'cost': 0.7,
            'latency': 2.0,
            'categories': {ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED}
        },
        VerificationMethod.KNOWLEDGE_GRAPH: {
            'cost': 0.2,
            'latency': 0.5,
            'categories': {ClaimCategoryType.SELF_CONSISTENCY_REQUIRED}
        },
        VerificationMethod.EXPERT_VERIFICATION: {
            'cost': 1.0,
            'latency': 86400.0,  # 24 hours
            'categories': {ClaimCategoryType.AMBIGUOUS_RESOLUTION_REQUIRED}
        },
        VerificationMethod.UNVERIFIABLE: {
            'cost': 0.0,
            'latency': 0.0,
            'categories': {ClaimCategoryType.SUBJECTIVE_OPINION, ClaimCategoryType.PROCEDURAL_DESCRIPTIVE}
        }
    }
    
    # Priority order for verification methods (lower number = higher priority)
    METHOD_PRIORITY = {
        VerificationMethod.CROSS_MODAL: 1,           # Visual grounding - highest priority
        VerificationMethod.EXTERNAL_SOURCE: 2,      # External knowledge - high priority
        VerificationMethod.KNOWLEDGE_GRAPH: 3,      # Self-consistency - medium priority
        VerificationMethod.EXPERT_VERIFICATION: 4,  # Ambiguous resolution - low priority
        VerificationMethod.UNVERIFIABLE: 5          # Subjective/procedural - lowest priority
    }

    def __init__(self, available_methods: Optional[Set[VerificationMethod]] = None):
        """Initialize the CheckRouter with available verification methods.
        
        Args:
            available_methods: Set of available verification methods.
                             If None, all methods are considered available.
        """
        self.available_methods = available_methods or set(self.VERIFICATION_METHODS.keys())
        logger.info(f"Check Router initialized with methods: {[m.name for m in self.available_methods]}")

    def route_claim(self, claim: ExtractedClaim) -> VerificationRoute:
        """
        Determine the best verification method for a claim based on its MLLM hallucination detection categories.
        
        Args:
            claim: The claim to route
            
        Returns:
            VerificationRoute: The recommended verification method and metadata
        """
        if not claim or not claim.text.strip() or not claim.categories:
            return self._create_route(VerificationMethod.UNVERIFIABLE, 0.0, "Invalid claim or no categories provided")
        
        # Extract all category types from the claim's categories
        claim_categories = {cat_obj.name for cat_obj in claim.categories}
        
        # Direct category-to-method mapping for MLLM hallucination detection
        if ClaimCategoryType.VISUAL_GROUNDING_REQUIRED in claim_categories:
            if VerificationMethod.CROSS_MODAL in self.available_methods:
                return self._create_route(
                    VerificationMethod.CROSS_MODAL,
                    confidence=0.95,
                    justification="Visual grounding required - routed to cross-modal verification"
                )
        
        # DISABLED: Route EXTERNAL_KNOWLEDGE_REQUIRED to cross-modal instead of external source
        # This ensures numerical calculation claims go to AGLA for MME evaluation
        if ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED in claim_categories:
            if VerificationMethod.CROSS_MODAL in self.available_methods:
                return self._create_route(
                    VerificationMethod.CROSS_MODAL,
                    confidence=0.90,
                    justification="External knowledge required - redirected to cross-modal verification (AGLA) for MME evaluation"
                )
            # Fallback to external source if cross-modal not available
            elif VerificationMethod.EXTERNAL_SOURCE in self.available_methods:
                return self._create_route(
                    VerificationMethod.EXTERNAL_SOURCE,
                    confidence=0.90,
                    justification="External knowledge required - using external source verification (fallback)"
                )
        
        if ClaimCategoryType.SELF_CONSISTENCY_REQUIRED in claim_categories:
            if VerificationMethod.KNOWLEDGE_GRAPH in self.available_methods:
                return self._create_route(
                    VerificationMethod.KNOWLEDGE_GRAPH,
                    confidence=0.85,
                    justification="Self-consistency check required - routed to knowledge graph verification"
                )
        
        if ClaimCategoryType.AMBIGUOUS_RESOLUTION_REQUIRED in claim_categories:
            return self._create_route(
                VerificationMethod.EXPERT_VERIFICATION,
                confidence=0.80,
                justification="Claim is ambiguous or unclear and requires expert review."
            )
        
        if ClaimCategoryType.SUBJECTIVE_OPINION in claim_categories:
            return self._create_route(
                VerificationMethod.UNVERIFIABLE,
                confidence=1.0,
                justification="Subjective opinion - no factuality check required"
            )
        
        if ClaimCategoryType.PROCEDURAL_DESCRIPTIVE in claim_categories:
            return self._create_route(
                VerificationMethod.UNVERIFIABLE,
                confidence=0.8,
                justification="Procedural description - context-dependent, not easily fact-checkable"
            )
        
        # Fallback: Find best matching method from available options
        applicable_methods = []
        
        for method in self.available_methods:
            if method not in self.VERIFICATION_METHODS:
                continue
                
            method_info = self.VERIFICATION_METHODS[method]
            
            # Calculate category overlap
            overlap = claim_categories.intersection(method_info['categories'])
            if not overlap:
                continue
                
            # Calculate confidence based on category matches
            confidence = len(overlap) / len(claim_categories) if claim_categories else 0.0
            
            # Adjust confidence based on method priority (higher priority = higher confidence)
            priority_boost = 1.0 - (self.METHOD_PRIORITY.get(method, 10) * 0.1)
            confidence = (confidence * 0.7) + (priority_boost * 0.3)
            
            applicable_methods.append((
                method,
                confidence,
                f"Matched categories: {', '.join([c.name for c in overlap])}"
            ))
        
        # Sort by confidence (highest first)
        applicable_methods.sort(key=lambda x: x[1], reverse=True)
        
        # Select the best method
        if applicable_methods:
            best_method, confidence, justification = applicable_methods[0]
            
            # If confidence is too low, mark as unverifiable
            if confidence < 0.3:
                return self._create_route(
                    VerificationMethod.UNVERIFIABLE,
                    1.0 - confidence,
                    f"Low confidence in verification methods. Best match was {best_method.name} with {confidence:.2f} confidence"
                )
                
            return self._create_route(
                best_method,
                confidence,
                justification,
                self.VERIFICATION_METHODS[best_method]['cost'],
                self.VERIFICATION_METHODS[best_method]['latency']
            )
        
        # No applicable methods found - default to external source as fallback
        if VerificationMethod.EXTERNAL_SOURCE in self.available_methods:
            return self._create_route(
                VerificationMethod.EXTERNAL_SOURCE,
                confidence=0.5,
                justification="No direct category match - defaulting to external source verification"
            )
        
        # Last resort
        return self._create_route(
            VerificationMethod.UNVERIFIABLE,
            1.0,
            "No suitable verification method found for the given categories"
        )

    def _create_route(
        self,
        method: VerificationMethod,
        confidence: float,
        justification: str,
        cost: Optional[float] = None,
        latency: Optional[float] = None
    ) -> VerificationRoute:
        """Helper to create a VerificationRoute with default values if not provided"""
        if cost is None:
            cost = self.VERIFICATION_METHODS.get(method, {}).get('cost', 1.0)
        if latency is None:
            latency = self.VERIFICATION_METHODS.get(method, {}).get('latency', 60.0)
            
        return VerificationRoute(
            method=method,
            confidence=confidence,
            justification=justification,
            estimated_cost=cost,
            estimated_latency=latency
        )
    
    def get_verification_methods(self) -> Dict[VerificationMethod, Dict[str, any]]:
        """Get information about all available verification methods"""
        return {
            method: info
            for method, info in self.VERIFICATION_METHODS.items()
            if method in self.available_methods
        }
