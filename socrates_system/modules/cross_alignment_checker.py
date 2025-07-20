"""
Cross-Alignment Checker - Multimodal consistency verification
Checks alignment between textual claims and visual content using open-source models
"""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from typing import Dict, List, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

from ..utils.logger import setup_logger
from ..config import VISION_MODEL_NAME, NLP_MODEL_NAME, CONFIDENCE_THRESHOLD

logger = setup_logger(__name__)

@dataclass
class AlignmentResult:
    """Result of cross-alignment check"""
    status: str  # PASS, FAIL, UNCERTAIN
    confidence: float
    visual_description: str
    text_claim: str
    alignment_score: float
    contradictions: List[str]
    evidence: List[str]
    reasoning: str

class CrossAlignmentChecker:
    """
    Advanced cross-alignment checker for image-text consistency
    Uses state-of-the-art vision-language models for verification
    """
    
    def __init__(self):
        logger.info("Initializing Cross-Alignment Checker...")
        
        try:
            # Load vision-language model for image captioning
            self.vision_processor = BlipProcessor.from_pretrained(VISION_MODEL_NAME)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL_NAME)
            
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer(NLP_MODEL_NAME)
            
            # Device setup
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vision_model.to(self.device)
            
            # Alignment keywords for different types of claims
            self.alignment_keywords = {
                'location': ['in', 'at', 'on', 'near', 'inside', 'outside', 'behind', 'front'],
                'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray'],
                'size': ['big', 'small', 'large', 'tiny', 'huge', 'massive', 'little'],
                'quantity': ['one', 'two', 'three', 'many', 'few', 'several', 'multiple'],
                'action': ['running', 'walking', 'sitting', 'standing', 'flying', 'swimming'],
                'object': ['person', 'car', 'building', 'tree', 'animal', 'food', 'furniture']
            }
            
            logger.info("Cross-Alignment Checker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Cross-Alignment Checker: {str(e)}")
            raise
    
    def check_alignment(self, text_claim: str, image_path: str) -> Dict[str, Any]:
        """
        Check alignment between text claim and image content
        
        Args:
            text_claim: The textual claim to verify
            image_path: Path to the image file
            
        Returns:
            Dictionary containing alignment results
        """
        logger.info(f"Checking alignment for claim: {text_claim[:50]}...")
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Generate visual description
            visual_description = self._generate_visual_description(image)
            logger.info(f"Generated visual description: {visual_description[:100]}...")
            
            # Perform detailed alignment analysis
            alignment_result = self._analyze_alignment(text_claim, visual_description, image)
            
            # Format result
            result = {
                "status": alignment_result.status,
                "confidence": alignment_result.confidence,
                "visual_description": alignment_result.visual_description,
                "alignment_score": alignment_result.alignment_score,
                "contradictions": alignment_result.contradictions,
                "evidence": alignment_result.evidence,
                "reasoning": alignment_result.reasoning
            }
            
            logger.info(f"Alignment check completed: {alignment_result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Error in alignment check: {str(e)}")
            return {
                "status": "ERROR",
                "confidence": 0.0,
                "visual_description": "",
                "alignment_score": 0.0,
                "contradictions": [f"Error processing image: {str(e)}"],
                "evidence": [],
                "reasoning": "Technical error prevented alignment verification"
            }
    
    def _generate_visual_description(self, image: Image.Image) -> str:
        """Generate detailed description of image content"""
        try:
            # Generate unconditional caption
            inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            out = self.vision_model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.vision_processor.decode(out[0], skip_special_tokens=True)
            
            # Generate conditional captions for specific aspects
            conditional_prompts = [
                "What objects are visible in this image?",
                "What colors are prominent in this image?",
                "What is the setting or location of this image?",
                "What activities or actions are happening in this image?"
            ]
            
            detailed_descriptions = [caption]
            
            for prompt in conditional_prompts:
                inputs = self.vision_processor(image, prompt, return_tensors="pt").to(self.device)
                out = self.vision_model.generate(**inputs, max_length=50, num_beams=3)
                desc = self.vision_processor.decode(out[0], skip_special_tokens=True)
                if desc and desc not in detailed_descriptions:
                    detailed_descriptions.append(desc)
            
            # Combine descriptions
            full_description = ". ".join(detailed_descriptions)
            return full_description
            
        except Exception as e:
            logger.error(f"Error generating visual description: {str(e)}")
            return "Unable to generate visual description"
    
    def _analyze_alignment(self, text_claim: str, visual_description: str, image: Image.Image) -> AlignmentResult:
        """Perform detailed alignment analysis"""
        
        # Calculate semantic similarity
        claim_embedding = self.sentence_model.encode([text_claim])
        visual_embedding = self.sentence_model.encode([visual_description])
        similarity_score = np.dot(claim_embedding[0], visual_embedding[0]) / (
            np.linalg.norm(claim_embedding[0]) * np.linalg.norm(visual_embedding[0])
        )
        
        # Analyze specific aspects
        aspect_analysis = self._analyze_specific_aspects(text_claim, visual_description)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(text_claim, visual_description, aspect_analysis)
        
        # Gather supporting evidence
        evidence = self._gather_supporting_evidence(text_claim, visual_description, aspect_analysis)
        
        # Calculate overall confidence
        confidence = self._calculate_alignment_confidence(
            similarity_score, aspect_analysis, contradictions, evidence
        )
        
        # Determine status
        if contradictions and confidence < 0.4:
            status = "FAIL"
        elif confidence > CONFIDENCE_THRESHOLD:
            status = "PASS"
        else:
            status = "UNCERTAIN"
        
        # Generate reasoning
        reasoning = self._generate_alignment_reasoning(
            text_claim, visual_description, similarity_score, aspect_analysis, contradictions
        )
        
        return AlignmentResult(
            status=status,
            confidence=confidence,
            visual_description=visual_description,
            text_claim=text_claim,
            alignment_score=similarity_score,
            contradictions=contradictions,
            evidence=evidence,
            reasoning=reasoning
        )
    
    def _analyze_specific_aspects(self, text_claim: str, visual_description: str) -> Dict[str, Any]:
        """Analyze specific aspects like objects, colors, locations, etc."""
        analysis = {}
        
        text_lower = text_claim.lower()
        visual_lower = visual_description.lower()
        
        for aspect, keywords in self.alignment_keywords.items():
            text_matches = [kw for kw in keywords if kw in text_lower]
            visual_matches = [kw for kw in keywords if kw in visual_lower]
            
            analysis[aspect] = {
                'text_mentions': text_matches,
                'visual_mentions': visual_matches,
                'alignment': len(set(text_matches) & set(visual_matches)) > 0,
                'contradiction': bool(text_matches and visual_matches and 
                                   not set(text_matches) & set(visual_matches))
            }
        
        return analysis
    
    def _detect_contradictions(self, text_claim: str, visual_description: str, 
                             aspect_analysis: Dict[str, Any]) -> List[str]:
        """Detect specific contradictions between text and visual content"""
        contradictions = []
        
        # Check aspect-level contradictions
        for aspect, analysis in aspect_analysis.items():
            if analysis['contradiction']:
                contradictions.append(
                    f"{aspect.capitalize()} mismatch: text mentions {analysis['text_mentions']} "
                    f"but image shows {analysis['visual_mentions']}"
                )
        
        # Check for explicit negations
        text_lower = text_claim.lower()
        visual_lower = visual_description.lower()
        
        # Common contradiction patterns
        contradiction_patterns = [
            ('no', 'yes'), ('not', 'is'), ('empty', 'full'), ('absent', 'present'),
            ('indoor', 'outdoor'), ('day', 'night'), ('single', 'multiple')
        ]
        
        for neg_word, pos_word in contradiction_patterns:
            if neg_word in text_lower and pos_word in visual_lower:
                contradictions.append(f"Contradiction: text suggests '{neg_word}' but image shows '{pos_word}'")
        
        return contradictions
    
    def _gather_supporting_evidence(self, text_claim: str, visual_description: str, 
                                  aspect_analysis: Dict[str, Any]) -> List[str]:
        """Gather evidence supporting the alignment"""
        evidence = []
        
        # Check for direct matches
        text_words = set(text_claim.lower().split())
        visual_words = set(visual_description.lower().split())
        common_words = text_words & visual_words
        
        if common_words:
            evidence.append(f"Common elements found: {', '.join(list(common_words)[:5])}")
        
        # Check aspect alignments
        for aspect, analysis in aspect_analysis.items():
            if analysis['alignment']:
                evidence.append(f"{aspect.capitalize()} alignment: {analysis['text_mentions']} confirmed in image")
        
        return evidence
    
    def _calculate_alignment_confidence(self, similarity_score: float, aspect_analysis: Dict[str, Any], 
                                      contradictions: List[str], evidence: List[str]) -> float:
        """Calculate overall confidence in alignment"""
        confidence = similarity_score * 0.4  # Base semantic similarity
        
        # Boost for aspect alignments
        aligned_aspects = sum(1 for analysis in aspect_analysis.values() if analysis['alignment'])
        confidence += (aligned_aspects / len(aspect_analysis)) * 0.3
        
        # Boost for evidence
        confidence += min(len(evidence) * 0.1, 0.2)
        
        # Penalty for contradictions
        confidence -= min(len(contradictions) * 0.2, 0.4)
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_alignment_reasoning(self, text_claim: str, visual_description: str, 
                                    similarity_score: float, aspect_analysis: Dict[str, Any], 
                                    contradictions: List[str]) -> str:
        """Generate human-readable reasoning for the alignment decision"""
        reasoning_parts = []
        
        # Semantic similarity
        if similarity_score > 0.7:
            reasoning_parts.append(f"High semantic similarity ({similarity_score:.2f}) between text and visual content")
        elif similarity_score > 0.4:
            reasoning_parts.append(f"Moderate semantic similarity ({similarity_score:.2f}) detected")
        else:
            reasoning_parts.append(f"Low semantic similarity ({similarity_score:.2f}) between text and image")
        
        # Aspect analysis
        aligned_aspects = [aspect for aspect, analysis in aspect_analysis.items() if analysis['alignment']]
        if aligned_aspects:
            reasoning_parts.append(f"Aligned aspects: {', '.join(aligned_aspects)}")
        
        # Contradictions
        if contradictions:
            reasoning_parts.append(f"Contradictions detected: {len(contradictions)} issues found")
        
        return ". ".join(reasoning_parts)
    
    def batch_check_alignment(self, claims_and_images: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Check alignment for multiple claim-image pairs"""
        logger.info(f"Performing batch alignment check for {len(claims_and_images)} pairs")
        
        results = []
        for i, (claim, image_path) in enumerate(claims_and_images):
            logger.info(f"Processing pair {i+1}/{len(claims_and_images)}")
            result = self.check_alignment(claim, image_path)
            results.append(result)
        
        return results
