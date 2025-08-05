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

@dataclass
class ClaimCategory:
    """Represents a category assigned to a claim"""
    name: str
    confidence: float
    justification: str

class ClaimCategorizer:
    """
    Categorizes claims into one or more categories using LLM-based classification
    with rule-based fallback mechanisms.
    """
    CATEGORIES = {cat.name: cat.value for cat in ClaimCategoryType}
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize the Claim Categorizer.
        
        Args:
            llm_manager: Optional LLMManager instance for LLM-based categorization.
                        If None, will use rule-based fallback only.
        """
        self.llm_manager = llm_manager
        self.compiled_patterns = {}
        self._initialize_patterns()
        logger.info("Claim Categorizer initialized")
    
    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for rule-based categorization"""
        raw_patterns = {
            "SELF_REFERENTIAL": [
                r'\b(I am|my name is|I am an AI assistant|you just asked me)\b',
                r'\b(you said|you mentioned|your question)\b'
            ],
            "SUBJECTIVE_OPINION": [
                r'\b(I think|I believe|in my opinion|my view is|I feel|it seems|appears to be)\b',
                r'\b(good|bad|great|terrible|beautiful|ugly|best|worst|important|trivial|happy|sad)\b',
            ],
            "QUANTITATIVE": [
                r'\b(\d+(\.\d+)?%?|\$?\d+(\.\d+)?|\d+[km]?\b)',
                r'\b(percent|percentage|ratio|fraction|average|mean|median)\b',
            ],
            "TEMPORAL": [
                r'\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
                r'\b(yesterday|today|tomorrow|now|recently|previously|initially|finally|after|before|during|since|until|when|while|as)\b',
            ],
            "COMPARATIVE": [
                r'\b(than|compared to|versus|vs\.?|as (?:much|many|long|far|little) as|more|less|fewer|better|worse|higher|lower|faster|slower|stronger|weaker)\b',
            ],
            "DEFINITIONAL": [
                r'\b(is defined as|means|refers to|is a type of|is a kind of|is an example of|is called|known as)\b',
            ],
            "CAUSAL": [
                r'\b(causes?|leads? to|results? in|because|due to|as a result|therefore|thus|hence|consequently|since|as|so)\b',
            ],
            "CROSS_MODAL": [
                r'\b(picture|image|photo|graph|chart|diagram|video|figure|illustration|this scene)\b',
                r'\b(shows|depicts|illustrates|contains|in this picture)\b'
            ],
            "HYPOTHETICAL_PREDICTIVE": [
                r'\b(if|when|will|would|could|should|might|may|predict|expect|hypothesize|assume|imagine)\b',
                r'\b(future|potential|possibility|hypothetical)\b'
            ],
            "AMBIGUOUS_UNCLEAR": [
                r'\b(something|somehow|somewhat|maybe|perhaps|unclear|vague|it seems)\b',
                r'\?$' # Ends with a question mark
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in raw_patterns.items()
        }
    
    def categorize_claim(self, claim: ExtractedClaim) -> ExtractedClaim:
        """
        Categorize a single claim into one or more categories.
        
        Args:
            claim: The claim to categorize
            
        Returns:
            The claim object, updated with categories.
        """
        if not claim or not claim.text.strip():
            return []
            
        try:
            # Try LLM-based categorization first if available
            if self.llm_manager:
                categories = self._categorize_with_llm(claim)
                if categories:  # Only use LLM results if we got valid categories
                    claim.categories = categories
                    return claim
            
            # Fall back to rule-based categorization
            categories = self._categorize_with_rules(claim)
            claim.categories = categories
            return claim
            
        except Exception as e:
            logger.error(f"Error categorizing claim '{claim.text}': {str(e)}", exc_info=True)
            # Return a default category if something goes wrong
            claim.categories = [
                ClaimCategory(
                    name="AMBIGUOUS/UNCLEAR",
                    confidence=0.5,
                    justification="Error occurred during categorization"
                )
            ]
    
    def _categorize_with_llm(self, claim: ExtractedClaim) -> List[ClaimCategory]:
        """Categorize a claim using the LLM"""
        try:
            # Format the prompt with claim text and available categories
            prompt = CLAIM_CATEGORIZATION_PROMPT.format(
                claim_text=claim.text,
                categories=json.dumps(self.CATEGORIES, indent=2)
            )
            
            # Get response from LLM
            response = self.llm_manager.generate_text(prompt)
            
            # Parse the response
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.warning(f"LLM-based categorization failed: {str(e)}")
            # Return a default category in the correct format on failure
            return [ClaimCategory(name=[ClaimCategoryType.FACTUAL], confidence=0.5, justification="Default fallback due to LLM error")]
    
    def _parse_llm_response(self, response: str) -> List[ClaimCategory]:
        """Parse the LLM's response into ClaimCategory objects"""
        try:
            # Try to extract JSON from the response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
                data = json.loads(json_str)
            else:
                # Try to find a JSON array in the response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    # If no JSON found, return empty list
                    return []
            
            # Convert to ClaimCategory objects
            categories = []
            for item in data:
                if not isinstance(item, dict) or 'categories' not in item:
                    continue

                # Create a single ClaimCategory object that holds all categories
                # This seems to be what the calling code expects.
                cats_from_llm = item.get('categories', [])
                valid_cats = []
                for cat_name in cats_from_llm:
                    # Normalize the category name to match the enum members (uppercase, no slashes)
                    normalized_cat = cat_name.upper().strip().replace('/', '_')
                    if normalized_cat in self.CATEGORIES:
                        valid_cats.append(ClaimCategoryType[normalized_cat])

                if valid_cats:
                    categories.append(ClaimCategory(
                        name=valid_cats, # list of enums
                        confidence=item.get('confidence', 0.85),
                        justification=item.get('justification', 'Categorized by LLM.')
                    ))
            
            return categories
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"LLM categorization failed or returned empty for response: '{response}'. Using default categorization.")
            return [ClaimCategory(name=[ClaimCategoryType.FACTUAL], confidence=0.5, justification="Default fallback")]
    
    def _categorize_with_rules(self, claim: ExtractedClaim) -> List[ClaimCategory]:
        """Categorize a claim using rule-based patterns"""
        categories = set()
        text = claim.text.lower()
        logger.debug(f"Categorizing with rules: '{text}'")
        
        # Check each category's patterns
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    logger.debug(f"  MATCH: '{pattern.pattern}' -> {category}")
                    categories.add(category)
                    break  # No need to check other patterns for this category
        
        # If no categories matched, use default
        if not categories:
            categories.add("FACTUAL")
        
        # Convert to ClaimCategory objects with confidence scores
        return [
            ClaimCategory(
                name=[ClaimCategoryType[cat]],  # Ensure name is a list of enums
                confidence=0.7,  # Lower confidence for rule-based categorization
                justification=f"Matched rule-based patterns for {cat}"
            )
            for cat in categories
        ]
    
    def get_category_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available categories"""
        return self.CATEGORIES.copy()
