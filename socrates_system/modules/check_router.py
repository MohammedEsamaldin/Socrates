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
    and other factors like cost and latency requirements.
    """
    
    # Default verification methods and their properties
    VERIFICATION_METHODS = {
        VerificationMethod.EXTERNAL_SOURCE: {
            'cost': 0.7,
            'latency': 2.0,  # seconds
            'categories': {ClaimCategoryType.FACTUAL, ClaimCategoryType.QUANTITATIVE, ClaimCategoryType.TEMPORAL, ClaimCategoryType.RELATIONAL, ClaimCategoryType.CAUSAL}
        },
        VerificationMethod.KNOWLEDGE_GRAPH: {
            'cost': 0.2,
            'latency': 0.5,
            'categories': {ClaimCategoryType.FACTUAL, ClaimCategoryType.DEFINITIONAL, ClaimCategoryType.RELATIONAL}
        },
        VerificationMethod.CROSS_MODAL: {
            'cost': 0.8,
            'latency': 3.0,
            'categories': {ClaimCategoryType.CROSS_MODAL, ClaimCategoryType.FACTUAL}
        },
        VerificationMethod.EXPERT_VERIFICATION: {
            'cost': 1.0,
            'latency': 86400.0,  # 24 hours
            'categories': {ClaimCategoryType.AMBIGUOUS_UNCLEAR, ClaimCategoryType.HYPOTHETICAL_PREDICTIVE}
        },
        VerificationMethod.CALCULATION: {
            'cost': 0.3,
            'latency': 0.1,
            'categories': {ClaimCategoryType.QUANTITATIVE}
        },
        VerificationMethod.DEFINITIONAL: {
            'cost': 0.1,
            'latency': 0.2,
            'categories': {ClaimCategoryType.DEFINITIONAL}
        },
        VerificationMethod.UNVERIFIABLE: {
            'cost': 0.0,
            'latency': 0.0,
            'categories': {ClaimCategoryType.SUBJECTIVE_OPINION, ClaimCategoryType.SELF_REFERENTIAL}
        }
    }
    
    # Priority order for verification methods (lower number = higher priority)
    METHOD_PRIORITY = {
        VerificationMethod.KNOWLEDGE_GRAPH: 1,
        VerificationMethod.DEFINITIONAL: 2,
        VerificationMethod.CALCULATION: 3,
        VerificationMethod.EXTERNAL_SOURCE: 4,
        VerificationMethod.CROSS_MODAL: 5,
        VerificationMethod.EXPERT_VERIFICATION: 6,
        VerificationMethod.UNVERIFIABLE: 7
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
        Determine the best verification method for a claim based on its categories.
        
        Args:
            claim: The claim to route
            categories: List of categories assigned to the claim
            
        Returns:
            VerificationRoute: The recommended verification method and metadata
        """
        if not claim or not claim.text.strip() or not claim.categories:
            return self._create_route(VerificationMethod.UNVERIFIABLE, 0.0, "Invalid claim or no categories provided")
        
        # The new structure is a list of ClaimCategory objects, where each object's 'name' attribute is a list of enums.
        claim_categories = {enum_val for cat_obj in claim.categories for enum_val in cat_obj.name}

        # Handle AMBIGUOUS_UNCLEAR claims first
        if ClaimCategoryType.AMBIGUOUS_UNCLEAR in claim_categories:
            return self._create_route(
                VerificationMethod.EXPERT_VERIFICATION,
                confidence=0.9,
                justification="Claim is ambiguous or unclear and requires expert review."
            )
        
        # Find all applicable verification methods
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
        
        # No applicable methods found
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
