"""
Simplified Cross-Alignment Checker - Basic image-text consistency verification
Works without advanced vision models while maintaining core functionality
"""
from PIL import Image
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import re

from ..utils.logger import setup_logger
from ..config import CONFIDENCE_THRESHOLD

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
    Simplified cross-alignment checker for image-text consistency
    Uses basic image analysis and keyword matching
    """
    
    def __init__(self):
        logger.info("Initializing Simplified Cross-Alignment Checker...")
        
        # Keywords for different types of visual content
        self.visual_keywords = {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'orange', 'purple'],
            'size': ['big', 'small', 'large', 'tiny', 'huge', 'massive', 'little', 'enormous'],
            'quantity': ['one', 'two', 'three', 'four', 'five', 'many', 'few', 'several', 'multiple', 'single'],
            'location': ['in', 'at', 'on', 'near', 'inside', 'outside', 'behind', 'front', 'above', 'below'],
            'objects': ['person', 'people', 'car', 'building', 'tree', 'animal', 'food', 'furniture', 'house', 'road']
        }
        
        logger.info("Simplified Cross-Alignment Checker initialized successfully")
    
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
            # Load and analyze image
            image = Image.open(image_path)
            
            # Generate basic visual description
            visual_description = self._generate_basic_description(image, image_path)
            logger.info(f"Generated visual description: {visual_description}")
            
            # Perform alignment analysis
            alignment_result = self._analyze_alignment(text_claim, visual_description)
            
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
    
    def _generate_basic_description(self, image: Image.Image, image_path: str) -> str:
        """Generate basic description based on image properties"""
        try:
            # Get basic image properties
            width, height = image.size
            mode = image.mode
            
            # Analyze dominant colors (simplified)
            colors = self._analyze_colors(image)
            
            # Generate description
            description_parts = []
            
            # Size description
            if width > 1000 or height > 1000:
                description_parts.append("large image")
            elif width < 300 or height < 300:
                description_parts.append("small image")
            else:
                description_parts.append("medium-sized image")
            
            # Color description
            if colors:
                description_parts.append(f"with prominent colors: {', '.join(colors[:3])}")
            
            # Format description
            if 'landscape' in image_path.lower() or 'outdoor' in image_path.lower():
                description_parts.append("appears to be an outdoor scene")
            elif 'portrait' in image_path.lower() or 'person' in image_path.lower():
                description_parts.append("appears to contain people")
            elif 'building' in image_path.lower() or 'architecture' in image_path.lower():
                description_parts.append("appears to show buildings or architecture")
            
            return "Image " + " ".join(description_parts)
            
        except Exception as e:
            logger.warning(f"Error generating description: {str(e)}")
            return "Image analysis unavailable"
    
    def _analyze_colors(self, image: Image.Image) -> List[str]:
        """Simple color analysis"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Sample pixels from center region
            width, height = image.size
            center_x, center_y = width // 2, height // 2
            sample_size = min(width, height) // 4
            
            # Get a sample of pixels
            colors_found = []
            
            # Sample some pixels and categorize colors
            for x in range(max(0, center_x - sample_size), min(width, center_x + sample_size), 20):
                for y in range(max(0, center_y - sample_size), min(height, center_y + sample_size), 20):
                    try:
                        r, g, b = image.getpixel((x, y))
                        color_name = self._categorize_color(r, g, b)
                        if color_name and color_name not in colors_found:
                            colors_found.append(color_name)
                    except:
                        continue
            
            return colors_found[:5]  # Return top 5 colors
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return []
    
    def _categorize_color(self, r: int, g: int, b: int) -> Optional[str]:
        """Categorize RGB values into color names"""
        # Simple color categorization
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g + 50 and r > b + 50:
            return "red"
        elif g > r + 50 and g > b + 50:
            return "green"
        elif b > r + 50 and b > g + 50:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 100 and g < 100 and b > 100:
            return "purple"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif r > 100 and g > 100 and b > 100:
            return "gray"
        else:
            return None
    
    def _analyze_alignment(self, text_claim: str, visual_description: str) -> AlignmentResult:
        """Perform alignment analysis using keyword matching"""
        
        # Extract keywords from text and visual description
        text_keywords = self._extract_keywords(text_claim.lower())
        visual_keywords = self._extract_keywords(visual_description.lower())
        
        # Calculate alignment score
        common_keywords = set(text_keywords) & set(visual_keywords)
        total_keywords = set(text_keywords) | set(visual_keywords)
        
        if total_keywords:
            alignment_score = len(common_keywords) / len(total_keywords)
        else:
            alignment_score = 0.0
        
        # Detect contradictions
        contradictions = self._detect_simple_contradictions(text_claim, visual_description)
        
        # Gather evidence
        evidence = []
        if common_keywords:
            evidence.append(f"Common elements found: {', '.join(common_keywords)}")
        
        # Calculate confidence
        confidence = alignment_score
        if contradictions:
            confidence *= 0.5  # Reduce confidence if contradictions found
        
        # Determine status
        if contradictions and confidence < 0.4:
            status = "FAIL"
        elif confidence > CONFIDENCE_THRESHOLD:
            status = "PASS"
        else:
            status = "UNCERTAIN"
        
        # Generate reasoning
        reasoning = self._generate_alignment_reasoning(
            text_claim, visual_description, alignment_score, contradictions, common_keywords
        )
        
        return AlignmentResult(
            status=status,
            confidence=confidence,
            visual_description=visual_description,
            text_claim=text_claim,
            alignment_score=alignment_score,
            contradictions=contradictions,
            evidence=evidence,
            reasoning=reasoning
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []
        
        # Extract keywords from each category
        for category, category_keywords in self.visual_keywords.items():
            for keyword in category_keywords:
                if keyword in text:
                    keywords.append(keyword)
        
        # Also extract capitalized words (likely proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        keywords.extend([word.lower() for word in capitalized_words])
        
        return list(set(keywords))  # Remove duplicates
    
    def _detect_simple_contradictions(self, text_claim: str, visual_description: str) -> List[str]:
        """Detect simple contradictions between text and visual"""
        contradictions = []
        
        text_lower = text_claim.lower()
        visual_lower = visual_description.lower()
        
        # Check for color contradictions
        text_colors = [color for color in self.visual_keywords['color'] if color in text_lower]
        visual_colors = [color for color in self.visual_keywords['color'] if color in visual_lower]
        
        if text_colors and visual_colors:
            if not set(text_colors) & set(visual_colors):
                contradictions.append(f"Color mismatch: text mentions {text_colors} but image shows {visual_colors}")
        
        # Check for size contradictions
        text_sizes = [size for size in self.visual_keywords['size'] if size in text_lower]
        visual_sizes = [size for size in self.visual_keywords['size'] if size in visual_lower]
        
        if text_sizes and visual_sizes:
            # Simple contradiction check for opposing sizes
            opposing_sizes = [('big', 'small'), ('large', 'tiny'), ('huge', 'little')]
            for text_size in text_sizes:
                for visual_size in visual_sizes:
                    for pair in opposing_sizes:
                        if (text_size in pair and visual_size in pair and text_size != visual_size):
                            contradictions.append(f"Size contradiction: text says {text_size} but image appears {visual_size}")
        
        return contradictions
    
    def _generate_alignment_reasoning(self, text_claim: str, visual_description: str,
                                    alignment_score: float, contradictions: List[str],
                                    common_keywords: set) -> str:
        """Generate reasoning for alignment decision"""
        reasoning_parts = []
        
        # Alignment score
        if alignment_score > 0.7:
            reasoning_parts.append(f"High keyword alignment ({alignment_score:.2f}) between text and visual content")
        elif alignment_score > 0.4:
            reasoning_parts.append(f"Moderate keyword alignment ({alignment_score:.2f}) detected")
        else:
            reasoning_parts.append(f"Low keyword alignment ({alignment_score:.2f}) between text and image")
        
        # Common elements
        if common_keywords:
            reasoning_parts.append(f"Shared elements: {', '.join(list(common_keywords)[:3])}")
        
        # Contradictions
        if contradictions:
            reasoning_parts.append(f"Contradictions detected: {len(contradictions)} issues found")
        
        return ". ".join(reasoning_parts)
    
    def batch_check_alignment(self, claims_and_images: List[tuple]) -> List[Dict[str, Any]]:
        """Check alignment for multiple claim-image pairs"""
        logger.info(f"Performing batch alignment check for {len(claims_and_images)} pairs")
        
        results = []
        for i, (claim, image_path) in enumerate(claims_and_images):
            logger.info(f"Processing pair {i+1}/{len(claims_and_images)}")
            result = self.check_alignment(claim, image_path)
            results.append(result)
        
        return results
