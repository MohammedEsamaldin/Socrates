"""
Claim Categorization Module

This module is responsible for categorizing extracted claims into one or more categories
using a combination of LLM-based classification and rule-based heuristics.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from socrates_system.config import CATEGORIZATION_CONFIDENCE_THRESHOLD
from socrates_system.modules.llm_manager import LLMManager
from socrates_system.modules.shared_structures import ExtractedClaim, ClaimCategory, ClaimCategoryType
from pathlib import Path

logger = logging.getLogger(__name__)

# Load prompt template
PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"
CLAIM_CATEGORIZATION_PROMPT = (PROMPT_TEMPLATES_DIR / "claim_categorisation.txt").read_text(encoding="utf-8")

class ClaimCategorizer:
    """
    Categorizes claims into one or more categories using LLM-based classification.
    
    This implementation uses a detailed prompt template to guide the LLM in accurately
    categorizing claims according to predefined categories, with special attention to
    identifying ambiguous claims based on the user's specific definition.
    """
    
    def __init__(self, llm_manager: Optional[LLMManager]):
        """Initialize the Claim Categorizer.
        
        Args:
            llm_manager: Required LLMManager instance for LLM-based categorization.
        """
        # Allow None: fall back to rule-based categorization when LLM is not available
        if not llm_manager:
            self.llm_manager = None
            logger.warning("LLMManager not provided; ClaimCategorizer will use rule-based fallback.")
        else:
            self.llm_manager = llm_manager
            logger.info("LLM-based Claim Categorizer initialized")
    
    def _format_categories_for_prompt(self) -> str:
        """Format the category information for the prompt template."""
        # This is just a fallback - the main prompt template already contains detailed descriptions
        return "\n".join([f"- {cat.name}: {cat.value}" for cat in ClaimCategoryType])
    
    def categorize_claim(self, claim: ExtractedClaim) -> ExtractedClaim:
        """
        Categorize a claim using LLM-based classification for MLLM hallucination detection.
        
        Args:
            claim: The extracted claim to categorize
            
        Returns:
            The input claim with updated categories
        """
        if not claim.text.strip():
            logger.warning("Received empty claim text for categorization")
            claim.categories = [ClaimCategory(
                name=ClaimCategoryType.AMBIGUOUS_RESOLUTION_REQUIRED,
                confidence=0.0,
                justification="Empty claim text requires clarification"
            )]
            return claim
            
        if self.llm_manager is None:
            # Use simple rules when no LLM
            claim.categories = self._categorize_with_rules(claim)
            return claim
        
        try:
            # Use LLM-based categorization - it will handle all validation and conversion
            llm_categories = self._categorize_with_llm(claim)
            
            if not llm_categories:
                raise ValueError("LLM returned no valid categories")
                
            # The categories are already validated and converted in _categorize_with_llm
            claim.categories = llm_categories
            return claim
            
        except Exception as e:
            logger.error(f"Error categorizing claim with LLM: {e}", exc_info=True)
            # Fallback to ambiguous resolution required if LLM fails
            claim.categories = [ClaimCategory(
                name=ClaimCategoryType.AMBIGUOUS_RESOLUTION_REQUIRED,
                confidence=0.0,
                justification=f"Error during LLM categorization, requires manual review: {str(e)[:200]}"
            )]
            return claim

    def _categorize_with_llm(self, claim: ExtractedClaim) -> List[ClaimCategory]:
        """
        Categorize a claim using the LLM with the detailed prompt template.
        
        The prompt template provides comprehensive guidance on claim categorization,
        with special attention to identifying ambiguous claims based on the user's definition.
        """
        try:
            # Prepare the prompt with the claim text
            prompt = CLAIM_CATEGORIZATION_PROMPT.format(claim_text=claim.text)
            
            # Get the LLM response with conservative settings for reliability
            response = self.llm_manager.generate_text(
                prompt=prompt,
                max_tokens=1000,  # Increased for detailed justifications
                temperature=0.2   # Lower temperature for more consistent results
            )
            
            # Clean and parse the response
            response_text = response.strip()
            
            # Handle potential markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            # Parse the JSON response
            try:
                categories_data = json.loads(response_text)
                if not isinstance(categories_data, list) or not categories_data:
                    raise ValueError("Expected a non-empty list of categories")
                
                # Process each category in the response
                result = []
                for cat_data in categories_data:
                    try:
                        # Get and validate category names
                        category_names = cat_data.get('categories', [])
                        if isinstance(category_names, str):
                            category_names = [category_names]
                        
                        if not category_names or not all(isinstance(c, str) for c in category_names):
                            logger.warning(f"Invalid category names in response: {category_names}")
                            continue
                            
                        # Convert string category names to ClaimCategoryType enums
                        category_enums = []
                        for cat_name in category_names:
                            try:
                                # Convert string to enum using the name lookup
                                cat_enum = ClaimCategoryType[cat_name.upper()]
                                category_enums.append(cat_enum)
                            except (KeyError, AttributeError) as e:
                                logger.warning(f"Invalid category name '{cat_name}': {e}")
                                continue
                        
                        if not category_enums:
                            logger.warning("No valid category enums after processing")
                            continue
                        
                        # Get and validate confidence
                        confidence = float(cat_data.get('confidence', 0.7))
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                        
                        # Get justification or provide a default
                        justification = str(cat_data.get('justification', 'No justification provided'))
                        
                        # Create separate category objects for each enum (ClaimCategory.name expects single enum)
                        for cat_enum in category_enums:
                            result.append(ClaimCategory(
                                name=cat_enum,
                                confidence=confidence,
                                justification=justification
                            ))
                        
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Error processing category data {cat_data}: {e}")
                        continue
                
                if not result:
                    raise ValueError("No valid categories after processing")
                    
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response as JSON: {e}\nResponse: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
                
        except Exception as e:
            logger.error(f"LLM categorization failed: {e}", exc_info=True)
            # Re-raise to be handled by the caller
            raise
    
    def _categorize_with_rules(self, claim: ExtractedClaim) -> List[ClaimCategory]:
        """Simple rule-based fallback for MLLM hallucination detection categories"""
        # Note: This is a basic fallback - LLM categorization is strongly preferred
        # for the sophisticated MLLM hallucination detection categories
        
        text = claim.text.lower()
        
        # Check for visual grounding indicators
        visual_indicators = ['color', 'shape', 'size', 'position', 'wearing', 'holding', 'sitting', 'standing', 
                           'visible', 'shown', 'depicted', 'image', 'picture', 'see', 'look', 'appears']
        if any(indicator in text for indicator in visual_indicators):
            return [ClaimCategory(
                name=ClaimCategoryType.VISUAL_GROUNDING_REQUIRED,
                confidence=0.6,
                justification="Contains visual reference indicators (rule-based fallback)"
            )]
        
        # Check for external knowledge indicators
        knowledge_indicators = ['was', 'were', 'is', 'are', 'has', 'have', 'in', 'on', 'at', 'during', 'between']
        if any(ind in text for ind in knowledge_indicators) or re.search(r"\b\d{3,4}\b", text):
            return [ClaimCategory(
                name=ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED,
                confidence=0.55,
                justification="Contains factual assertions likely requiring external knowledge (rule-based)"
            )]
        
        # Default to self-consistency
        opinion_indicators = ['beautiful', 'ugly', 'good', 'bad', 'better', 'worse', 'prefer', 'like', 'dislike',
                            'think', 'believe', 'opinion', 'seems', 'appears to be']
        if any(indicator in text for indicator in opinion_indicators):
            return [ClaimCategory(
                name=ClaimCategoryType.SUBJECTIVE_OPINION,
                confidence=0.6,
                justification="Contains subjective opinion indicators (rule-based fallback)"
            )]
        
        # Check for procedural indicators
        procedural_indicators = ['first', 'then', 'next', 'step', 'process', 'method', 'procedure', 'instructions']
        if any(indicator in text for indicator in procedural_indicators):
            return [ClaimCategory(
                name=ClaimCategoryType.PROCEDURAL_DESCRIPTIVE,
                confidence=0.6,
                justification="Contains procedural indicators (rule-based fallback)"
            )]
        
        # Default fallback to external knowledge required
        return [ClaimCategory(
            name=ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED,
            confidence=0.3,
            justification="Default external knowledge categorization (rule-based fallback)"
        )]

    def get_category_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available categories"""
        return {cat.name: cat.value for cat in ClaimCategoryType}
