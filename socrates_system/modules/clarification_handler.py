"""
Clarification Handler - Manages user clarifications and corrections
Handles the clarification process when contradictions or ambiguities are found
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ClarificationRequest:
    """Represents a clarification request"""
    id: str
    original_claim: str
    issue_type: str  # contradiction, ambiguity, alignment
    issue_details: str
    suggested_questions: List[str]
    timestamp: datetime

class ClarificationHandler:
    """
    Clarification handler for managing user clarifications
    Generates appropriate clarification requests and processes responses
    """
    
    def __init__(self):
        logger.info("Initializing Clarification Handler...")
        
        # Templates for different types of clarifications
        self.clarification_templates = {
            'contradiction': [
                "Your claim '{claim}' appears to contradict {source}. Could you clarify or provide additional context?",
                "There seems to be a discrepancy between your statement '{claim}' and {source}. How would you reconcile this?",
                "I found conflicting information regarding '{claim}'. Could you help clarify which interpretation is correct?"
            ],
            'ambiguity': [
                "Your statement '{claim}' contains some ambiguous terms. Could you provide more specific details?",
                "To better understand your claim '{claim}', could you clarify what you mean by {ambiguous_terms}?",
                "Your claim could be interpreted in multiple ways. Could you be more specific about '{claim}'?"
            ],
            'alignment': [
                "Your textual claim '{claim}' doesn't seem to align with what I observe in the image. Could you clarify?",
                "There appears to be a mismatch between your description '{claim}' and the visual content. Could you explain?",
                "I'm having difficulty reconciling your claim '{claim}' with the image content. Could you provide clarification?"
            ],
            'context': [
                "Your claim '{claim}' seems to depend on context that isn't clear. Could you provide more background?",
                "To properly verify '{claim}', I need more contextual information. Could you elaborate?",
                "Your statement '{claim}' contains references that need clarification. Could you be more specific?"
            ]
        }
        
        logger.info("Clarification Handler initialized")
    
    def generate_clarification(self, claim: str, conflicting_info: Any, 
                             issue_type: str = "contradiction") -> str:
        """
        Generate a clarification request for a problematic claim
        
        Args:
            claim: The original claim
            conflicting_info: Information that conflicts with the claim
            issue_type: Type of issue (contradiction, ambiguity, alignment, context)
            
        Returns:
            Clarification message string
        """
        logger.info(f"Generating clarification for {issue_type}: {claim[:50]}...")
        
        try:
            templates = self.clarification_templates.get(issue_type, self.clarification_templates['contradiction'])
            
            # Select appropriate template
            template = templates[0]  # Use first template for now
            
            # Format the template
            if issue_type == 'contradiction':
                if isinstance(conflicting_info, str):
                    source = f"external sources stating: '{conflicting_info}'"
                elif isinstance(conflicting_info, list):
                    source = f"external sources: {', '.join(conflicting_info[:2])}"
                else:
                    source = "available evidence"
                
                clarification = template.format(claim=claim, source=source)
                
            elif issue_type == 'ambiguity':
                if isinstance(conflicting_info, list):
                    ambiguous_terms = ', '.join(conflicting_info[:3])
                else:
                    ambiguous_terms = str(conflicting_info)
                
                clarification = template.format(claim=claim, ambiguous_terms=ambiguous_terms)
                
            elif issue_type == 'alignment':
                clarification = template.format(claim=claim)
                
            else:  # context
                clarification = template.format(claim=claim)
            
            # Add helpful follow-up
            follow_up = self._generate_follow_up_guidance(issue_type, claim, conflicting_info)
            if follow_up:
                clarification += f" {follow_up}"
            
            logger.info("Clarification generated successfully")
            return clarification
            
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            return f"Could you please clarify your claim '{claim}' as there seems to be some uncertainty about its accuracy?"
    
    def _generate_follow_up_guidance(self, issue_type: str, claim: str, conflicting_info: Any) -> str:
        """Generate helpful follow-up guidance"""
        guidance = {
            'contradiction': "Please provide additional evidence or context that supports your position.",
            'ambiguity': "Specific examples or more precise language would be helpful.",
            'alignment': "Please explain how your description relates to what's shown in the image.",
            'context': "Additional background information would help me better understand your claim."
        }
        
        return guidance.get(issue_type, "")
    
    def create_clarification_request(self, claim: str, issue_type: str, 
                                   issue_details: str, suggested_questions: List[str]) -> ClarificationRequest:
        """Create a structured clarification request"""
        request_id = f"clarify_{hash(claim)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ClarificationRequest(
            id=request_id,
            original_claim=claim,
            issue_type=issue_type,
            issue_details=issue_details,
            suggested_questions=suggested_questions,
            timestamp=datetime.now()
        )
    
    def process_clarification_response(self, original_claim: str, clarification_response: str) -> Dict[str, Any]:
        """
        Process user's clarification response
        
        Args:
            original_claim: The original problematic claim
            clarification_response: User's clarification
            
        Returns:
            Processed clarification result
        """
        logger.info(f"Processing clarification response for: {original_claim[:50]}...")
        
        try:
            # Analyze the clarification response
            response_analysis = self._analyze_clarification_response(original_claim, clarification_response)
            
            # Determine if clarification is sufficient
            is_sufficient = self._is_clarification_sufficient(response_analysis)
            
            # Generate updated claim if clarification is good
            updated_claim = None
            if is_sufficient:
                updated_claim = self._generate_updated_claim(original_claim, clarification_response)
            
            result = {
                "original_claim": original_claim,
                "clarification_response": clarification_response,
                "is_sufficient": is_sufficient,
                "updated_claim": updated_claim,
                "analysis": response_analysis,
                "next_steps": self._determine_next_steps(is_sufficient, response_analysis)
            }
            
            logger.info(f"Clarification processing completed: {'sufficient' if is_sufficient else 'needs more'}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing clarification: {str(e)}")
            return {
                "original_claim": original_claim,
                "clarification_response": clarification_response,
                "is_sufficient": False,
                "updated_claim": None,
                "analysis": {"error": str(e)},
                "next_steps": ["Please provide more detailed clarification"]
            }
    
    def _analyze_clarification_response(self, original_claim: str, response: str) -> Dict[str, Any]:
        """Analyze the quality and content of clarification response"""
        analysis = {
            "length": len(response.split()),
            "contains_specifics": False,
            "addresses_ambiguity": False,
            "provides_evidence": False,
            "changes_meaning": False
        }
        
        # Check for specific details
        specific_indicators = ['specifically', 'exactly', 'precisely', 'namely', 'for example']
        analysis["contains_specifics"] = any(indicator in response.lower() for indicator in specific_indicators)
        
        # Check if it addresses ambiguity
        clarifying_phrases = ['what i mean is', 'to clarify', 'more precisely', 'in other words']
        analysis["addresses_ambiguity"] = any(phrase in response.lower() for phrase in clarifying_phrases)
        
        # Check for evidence
        evidence_indicators = ['because', 'since', 'according to', 'evidence', 'source', 'study']
        analysis["provides_evidence"] = any(indicator in response.lower() for indicator in evidence_indicators)
        
        # Check if meaning significantly changes
        # Simple heuristic: if response is much longer and contains negations or corrections
        correction_indicators = ['actually', 'correction', 'mistake', 'wrong', 'not', 'instead']
        analysis["changes_meaning"] = (
            len(response.split()) > len(original_claim.split()) * 1.5 and
            any(indicator in response.lower() for indicator in correction_indicators)
        )
        
        return analysis
    
    def _is_clarification_sufficient(self, analysis: Dict[str, Any]) -> bool:
        """Determine if clarification response is sufficient"""
        # Sufficient if it contains specifics or addresses ambiguity, and is reasonably detailed
        return (
            analysis["length"] >= 5 and  # At least 5 words
            (analysis["contains_specifics"] or analysis["addresses_ambiguity"] or analysis["provides_evidence"])
        )
    
    def _generate_updated_claim(self, original_claim: str, clarification: str) -> str:
        """Generate an updated claim incorporating the clarification"""
        # Simple approach: combine original claim with clarification
        # In a more sophisticated system, this would use NLP to properly integrate
        
        if clarification.lower().startswith(('what i mean', 'to clarify', 'actually')):
            # Clarification is a replacement
            return clarification
        else:
            # Clarification is additional information
            return f"{original_claim} ({clarification})"
    
    def _determine_next_steps(self, is_sufficient: bool, analysis: Dict[str, Any]) -> List[str]:
        """Determine next steps based on clarification quality"""
        if is_sufficient:
            return ["Proceed with verification of updated claim"]
        else:
            next_steps = ["More clarification needed"]
            
            if analysis["length"] < 5:
                next_steps.append("Please provide more detailed explanation")
            
            if not analysis["contains_specifics"]:
                next_steps.append("Please be more specific in your clarification")
            
            if not analysis["addresses_ambiguity"]:
                next_steps.append("Please directly address the ambiguous aspects")
            
            return next_steps
    
    def generate_multiple_clarification_options(self, claim: str, issue_type: str, 
                                              conflicting_info: Any) -> List[str]:
        """Generate multiple clarification options for user to choose from"""
        templates = self.clarification_templates.get(issue_type, self.clarification_templates['contradiction'])
        
        clarifications = []
        for template in templates:
            try:
                if issue_type == 'contradiction':
                    source = f"external sources: {conflicting_info}" if isinstance(conflicting_info, str) else "available evidence"
                    clarification = template.format(claim=claim, source=source)
                elif issue_type == 'ambiguity':
                    ambiguous_terms = ', '.join(conflicting_info) if isinstance(conflicting_info, list) else str(conflicting_info)
                    clarification = template.format(claim=claim, ambiguous_terms=ambiguous_terms)
                else:
                    clarification = template.format(claim=claim)
                
                clarifications.append(clarification)
            except:
                continue
        
        return clarifications[:3]  # Return top 3 options
