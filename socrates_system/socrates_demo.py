"""
Socrates Agent Demo - Standalone demonstration of the Socratic methodology
Shows the core external hallucination detection workflow
"""
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import wikipedia
import requests
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

@dataclass
class ExtractedClaim:
    """Represents an extracted claim"""
    text: str
    claim_type: str
    confidence: float
    entities: List[str]

@dataclass
class SocraticInquiry:
    """Represents a Socratic inquiry"""
    question: str
    reasoning: str
    confidence: float

class SocratesAgent:
    """
    Socrates Agent - Demonstrates sophisticated external hallucination detection
    """
    
    def __init__(self):
        logger.info("Initializing Socrates Agent Demo...")
        
        # Knowledge base for fact checking
        self.knowledge_base = {
            "paris is the capital of france": {"status": "TRUE", "confidence": 1.0, "source": "Geographic knowledge"},
            "london is the capital of england": {"status": "TRUE", "confidence": 1.0, "source": "Geographic knowledge"},
            "berlin is the capital of germany": {"status": "TRUE", "confidence": 1.0, "source": "Geographic knowledge"},
            "rome is the capital of italy": {"status": "TRUE", "confidence": 1.0, "source": "Geographic knowledge"},
            "water boils at 100 degrees celsius": {"status": "TRUE", "confidence": 1.0, "source": "Scientific knowledge"},
            "the eiffel tower is in rome": {"status": "FALSE", "confidence": 0.9, "source": "Geographic knowledge"},
            "the sky is green": {"status": "FALSE", "confidence": 0.9, "source": "Common knowledge"},
        }
        
        # Socratic question templates
        self.question_templates = {
            'verification': [
                "What evidence supports the claim that {claim}?",
                "How can we independently verify that {claim}?",
                "What authoritative sources confirm that {claim}?",
                "Under what conditions would this claim be true?",
                "What would contradict this assertion?"
            ],
            'clarification': [
                "Could you elaborate on what you mean by this claim?",
                "What specific context are you considering?",
                "Could you provide more details about this statement?",
                "What evidence would you consider most relevant?"
            ],
            'deeper_analysis': [
                "What underlying assumptions does this claim rest upon?",
                "What are the implications if this claim were false?",
                "How does this claim fit within broader knowledge?",
                "What related claims would also need to be true?"
            ]
        }
        
        self.session_id = None
        self.conversation_history = []
        
        logger.info("Socrates Agent Demo initialized successfully")
    
    def start_session(self) -> str:
        """Start a new verification session"""
        self.session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        logger.info(f"Started demo session: {self.session_id}")
        return self.session_id
    
    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """Extract claims from input text using pattern matching"""
        logger.info(f"Extracting claims from: {text[:50]}...")
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        
        # Claim patterns
        patterns = {
            'attribute': r'(.+) is (.+)',
            'location': r'(.+) is (?:located )?in (.+)',
            'property': r'(.+) has (.+)',
            'measurement': r'(.+) (?:boils|freezes|measures) (?:at )?(.+)'
        }
        
        for sentence in sentences:
            # Extract entities (simple capitalized words)
            entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
            
            # Check patterns
            claim_type = 'general'
            confidence = 0.7
            
            for pattern_type, pattern in patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    claim_type = pattern_type
                    confidence = 0.8
                    break
            
            # Boost confidence for entities
            if entities:
                confidence += 0.1
            
            # Reduce confidence for uncertain language
            if any(word in sentence.lower() for word in ['maybe', 'perhaps', 'might']):
                confidence -= 0.2
            
            claims.append(ExtractedClaim(
                text=sentence,
                claim_type=claim_type,
                confidence=min(confidence, 1.0),
                entities=entities
            ))
        
        # If no sentences found, treat entire input as one claim
        if not claims:
            entities = re.findall(r'\b[A-Z][a-z]+\b', text)
            claims.append(ExtractedClaim(
                text=text,
                claim_type='general',
                confidence=0.6,
                entities=entities
            ))
        
        logger.info(f"Extracted {len(claims)} claims")
        return claims
    
    def generate_socratic_question(self, claim: str, question_type: str = 'verification') -> SocraticInquiry:
        """Generate a Socratic question for the claim"""
        templates = self.question_templates.get(question_type, self.question_templates['verification'])
        
        # Select template based on claim content
        if 'capital' in claim.lower():
            question = f"What authoritative sources confirm that {claim}?"
            reasoning = "This question seeks to establish independent verification of geographic facts through authoritative sources."
        elif 'temperature' in claim.lower() or 'degrees' in claim.lower():
            question = f"Under what specific conditions would {claim} be accurate?"
            reasoning = "This inquiry examines the contextual conditions necessary for scientific measurements to be valid."
        elif 'located' in claim.lower() or ' in ' in claim.lower():
            question = f"What evidence supports the location claim that {claim}?"
            reasoning = "This question probes the factual basis for geographic or spatial relationships."
        else:
            # Use first template and format
            template = templates[0]
            question = template.format(claim=claim)
            reasoning = "This question follows the Socratic principle of examining the foundations of knowledge through systematic inquiry."
        
        return SocraticInquiry(
            question=question,
            reasoning=reasoning,
            confidence=0.8
        )
    
    def verify_with_external_sources(self, claim: str) -> Dict[str, Any]:
        """Verify claim against external sources"""
        logger.info(f"Verifying claim externally: {claim[:50]}...")
        
        claim_lower = claim.lower().strip()
        
        # Check knowledge base first
        if claim_lower in self.knowledge_base:
            kb_entry = self.knowledge_base[claim_lower]
            return {
                "status": "PASS" if kb_entry["status"] == "TRUE" else "FAIL",
                "confidence": kb_entry["confidence"],
                "evidence": [f"Knowledge base: {kb_entry['source']}"],
                "contradictions": [] if kb_entry["status"] == "TRUE" else [f"Contradicts {kb_entry['source']}"],
                "external_facts": [claim_lower]
            }
        
        # Try Wikipedia verification
        try:
            # Extract key terms for search
            search_terms = self._extract_search_terms(claim)
            if search_terms:
                search_results = wikipedia.search(search_terms, results=2)
                
                if search_results:
                    # Get summary of first result
                    try:
                        page = wikipedia.page(search_results[0])
                        summary = page.summary[:300]
                        
                        # Simple keyword matching
                        claim_words = set(claim.lower().split())
                        summary_words = set(summary.lower().split())
                        overlap = len(claim_words & summary_words) / len(claim_words)
                        
                        if overlap > 0.3:
                            return {
                                "status": "PASS",
                                "confidence": min(overlap * 2, 0.9),
                                "evidence": [f"Wikipedia ({search_results[0]}): {summary[:100]}..."],
                                "contradictions": [],
                                "external_facts": [summary[:200]]
                            }
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Wikipedia verification failed: {str(e)}")
        
        # Default uncertain result
        return {
            "status": "UNCERTAIN",
            "confidence": 0.5,
            "evidence": [],
            "contradictions": [],
            "external_facts": ["No definitive external verification found"]
        }
    
    def _extract_search_terms(self, claim: str) -> str:
        """Extract key terms for search"""
        # Remove common words
        stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        words = [word for word in claim.split() if word.lower() not in stop_words and len(word) > 2]
        return " ".join(words[:4])  # Top 4 words
    
    def check_image_alignment(self, claim: str, image_path: str) -> Dict[str, Any]:
        """Check alignment between claim and image"""
        logger.info(f"Checking image alignment for: {claim[:50]}...")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Basic image analysis
            width, height = image.size
            
            # Generate simple description based on filename and properties
            filename = os.path.basename(image_path).lower()
            description_parts = []
            
            if 'tower' in filename or 'eiffel' in filename:
                description_parts.append("shows a tower structure")
            if 'paris' in filename:
                description_parts.append("appears to be in Paris")
            if 'rome' in filename:
                description_parts.append("appears to be in Rome")
            if 'building' in filename:
                description_parts.append("shows buildings")
            if 'landscape' in filename:
                description_parts.append("shows a landscape")
            
            # Analyze colors (simplified)
            if image.mode == 'RGB':
                # Sample center pixel for dominant color
                center_pixel = image.getpixel((width//2, height//2))
                r, g, b = center_pixel
                
                if r > 150 and g > 150 and b < 100:
                    description_parts.append("with yellow/golden colors")
                elif b > r + 50 and b > g + 50:
                    description_parts.append("with blue colors")
                elif g > r + 50 and g > b + 50:
                    description_parts.append("with green colors")
            
            visual_description = "Image " + ", ".join(description_parts) if description_parts else "Image content analyzed"
            
            # Check for contradictions
            contradictions = []
            claim_lower = claim.lower()
            
            if 'eiffel tower' in claim_lower and 'rome' in claim_lower:
                if 'paris' in visual_description.lower():
                    contradictions.append("Claim states Eiffel Tower is in Rome, but image appears to show Paris")
            
            if 'sky is green' in claim_lower:
                if 'blue' in visual_description.lower():
                    contradictions.append("Claim states sky is green, but image shows blue sky")
            
            # Calculate alignment
            alignment_score = 0.8 if not contradictions else 0.2
            status = "PASS" if not contradictions else "FAIL"
            
            return {
                "status": status,
                "confidence": alignment_score,
                "visual_description": visual_description,
                "contradictions": contradictions,
                "evidence": [] if contradictions else ["Visual content aligns with textual claim"],
                "alignment_score": alignment_score
            }
            
        except Exception as e:
            logger.error(f"Image alignment check failed: {str(e)}")
            return {
                "status": "ERROR",
                "confidence": 0.0,
                "visual_description": "Error analyzing image",
                "contradictions": [f"Image processing error: {str(e)}"],
                "evidence": [],
                "alignment_score": 0.0
            }
    
    def process_input(self, user_input: str, image_path: str = None) -> Dict[str, Any]:
        """Main processing pipeline implementing Socratic methodology"""
        logger.info(f"Processing input: {user_input[:50]}...")
        
        if not self.session_id:
            self.start_session()
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "image_path": image_path
        })
        
        try:
            # Stage 1: Claim Extraction
            claims = self.extract_claims(user_input)
            
            # Stage 2: Socratic Analysis
            verification_results = []
            
            for claim in claims:
                # Generate Socratic questions
                verification_question = self.generate_socratic_question(claim.text, 'verification')
                deeper_question = self.generate_socratic_question(claim.text, 'deeper_analysis')
                
                # External verification
                external_result = self.verify_with_external_sources(claim.text)
                
                # Image alignment (if image provided)
                alignment_result = None
                if image_path:
                    alignment_result = self.check_image_alignment(claim.text, image_path)
                
                # Determine overall status
                overall_status = external_result["status"]
                overall_confidence = external_result["confidence"]
                
                if alignment_result and alignment_result["status"] == "FAIL":
                    overall_status = "FAIL"
                    overall_confidence = min(overall_confidence, alignment_result["confidence"])
                
                # Compile evidence and contradictions
                evidence = external_result["evidence"].copy()
                contradictions = external_result["contradictions"].copy()
                
                if alignment_result:
                    evidence.extend(alignment_result["evidence"])
                    contradictions.extend(alignment_result["contradictions"])
                
                verification_results.append({
                    "claim": claim.text,
                    "claim_type": claim.claim_type,
                    "status": overall_status,
                    "confidence": overall_confidence,
                    "socratic_questions": [
                        {
                            "question": verification_question.question,
                            "reasoning": verification_question.reasoning,
                            "type": "verification"
                        },
                        {
                            "question": deeper_question.question,
                            "reasoning": deeper_question.reasoning,
                            "type": "deeper_analysis"
                        }
                    ],
                    "evidence": evidence,
                    "contradictions": contradictions,
                    "external_verification": external_result,
                    "image_alignment": alignment_result
                })
            
            # Generate Socratic dialogue
            socratic_dialogue = []
            for result in verification_results:
                # Add questions
                for sq in result["socratic_questions"]:
                    socratic_dialogue.append({
                        "type": "socratic_question",
                        "content": sq["question"],
                        "reasoning": sq["reasoning"]
                    })
                
                # Add verification result
                if result["status"] == "PASS":
                    socratic_dialogue.append({
                        "type": "verification_result",
                        "content": f"Through systematic examination, the claim '{result['claim']}' appears to be consistent with available evidence.",
                        "evidence": result["evidence"]
                    })
                else:
                    socratic_dialogue.append({
                        "type": "contradiction_found",
                        "content": f"Upon careful investigation, the claim '{result['claim']}' appears to contradict available evidence.",
                        "contradictions": result["contradictions"]
                    })
            
            # Compile final response
            passed_claims = [r for r in verification_results if r["status"] == "PASS"]
            failed_claims = [r for r in verification_results if r["status"] in ["FAIL", "UNCERTAIN"]]
            
            response = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "original_input": user_input,
                "verification_summary": {
                    "total_claims": len(verification_results),
                    "verified_claims": len(passed_claims),
                    "failed_claims": len(failed_claims),
                    "overall_status": "PASS" if len(failed_claims) == 0 else "FAIL"
                },
                "socratic_dialogue": socratic_dialogue,
                "detailed_results": verification_results,
                "next_steps": self._generate_next_steps(verification_results)
            }
            
            logger.info("Processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_next_steps(self, verification_results: List[Dict]) -> List[str]:
        """Generate next steps based on verification results"""
        next_steps = []
        
        failed_results = [r for r in verification_results if r["status"] in ["FAIL", "UNCERTAIN"]]
        
        if failed_results:
            next_steps.append("Review contradicted claims and consider additional evidence")
            for result in failed_results:
                if result["contradictions"]:
                    next_steps.append(f"Address contradiction in: {result['claim']}")
        else:
            next_steps.append("All claims verified successfully - analysis complete")
        
        return next_steps

# Initialize Socrates Agent
socrates_agent = SocratesAgent()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        session_id = socrates_agent.start_session()
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Demo session started successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start session: {str(e)}'
        }), 500

@app.route('/verify_claim', methods=['POST'])
def verify_claim():
    try:
        user_input = request.form.get('user_input', '').strip()
        if not user_input:
            return jsonify({'status': 'error', 'message': 'No input provided'}), 400
        
        # Handle image upload
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
        
        # Process with Socrates Agent
        result = socrates_agent.process_input(user_input, image_path)
        
        # Clean up uploaded file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        return jsonify({'status': 'success', 'result': result})
        
    except Exception as e:
        logger.error(f"Error in verification: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': 'Socrates Agent Demo'
    })

if __name__ == '__main__':
    logger.info("Starting Socrates Agent Demo...")
    app.run(debug=True, host='0.0.0.0', port=5000)
