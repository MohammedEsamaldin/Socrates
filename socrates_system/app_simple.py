"""
Socrates Agent System - Simplified Flask Application
Working version with basic dependencies for demonstration and testing
"""
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import re

# Simplified imports - using basic modules
from modules.claim_extractor_simple import ClaimExtractor
from modules.cross_alignment_checker_simple import CrossAlignmentChecker
from modules.external_factuality_checker import ExternalFactualityChecker
from modules.question_generator import QuestionGenerator
from modules.clarification_handler import ClarificationHandler
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

class SimplifiedSocratesAgent:
    """Simplified Socrates Agent for demonstration"""
    
    def __init__(self):
        logger.info("Initializing Simplified Socrates Agent...")
        
        # Initialize working modules
        self.claim_extractor = ClaimExtractor()
        self.question_generator = QuestionGenerator()
        self.cross_alignment_checker = CrossAlignmentChecker()
        self.external_factuality_checker = ExternalFactualityChecker()
        self.clarification_handler = ClarificationHandler()
        
        # Session state
        self.session_id = None
        self.conversation_history = []
        self.verified_claims = []
        
        logger.info("Simplified Socrates Agent initialized successfully")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new verification session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.conversation_history = []
        self.verified_claims = []
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def process_user_input(self, user_input: str, image_path: str = None) -> dict:
        """
        Main processing pipeline implementing Socratic methodology
        """
        logger.info(f"Processing user input: {user_input[:100]}...")
        
        if not self.session_id:
            self.start_session()
        
        # Store input in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "image_path": image_path
        })
        
        try:
            # Stage 1: Claim Extraction
            logger.info("Stage 1: Claim Extraction")
            claims = self.claim_extractor.extract_claims(user_input)
            
            # Stage 2: Factuality Checks
            logger.info("Stage 2: Factuality Checks")
            verification_results = []
            
            for claim in claims:
                result = self._verify_claim_socratically(claim, user_input, image_path)
                verification_results.append(result)
            
            # If no claims extracted, treat entire input as a claim
            if not verification_results:
                logger.info("No specific claims extracted, treating entire input as claim")
                mock_claim = type('MockClaim', (), {
                    'text': user_input,
                    'claim_type': 'general',
                    'confidence': 0.7,
                    'entities': []
                })()
                result = self._verify_claim_socratically(mock_claim, user_input, image_path)
                verification_results.append(result)
            
            # Compile final response
            response = self._compile_socratic_response(verification_results, user_input)
            
            logger.info("Processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred during processing: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _verify_claim_socratically(self, claim, original_input: str, image_path: str = None) -> dict:
        """Apply Socratic methodology to verify a single claim"""
        claim_text = claim.text if hasattr(claim, 'text') else str(claim)
        logger.info(f"Verifying claim: {claim_text}")
        
        socratic_questions = []
        evidence = []
        contradictions = []
        overall_status = "PASS"
        confidence = 1.0
        clarification_needed = None
        
        # Generate initial Socratic inquiry
        try:
            initial_inquiry = self.question_generator.generate_socratic_inquiry(claim_text, "verification")
            socratic_questions.append({
                "question": initial_inquiry.question,
                "reasoning": initial_inquiry.reasoning,
                "confidence": initial_inquiry.confidence
            })
        except Exception as e:
            logger.warning(f"Question generation failed: {str(e)}")
        
        # Check 1: Cross-alignment (if image provided)
        if image_path:
            logger.info("Performing cross-alignment check...")
            try:
                alignment_result = self.cross_alignment_checker.check_alignment(claim_text, image_path)
                
                if alignment_result["status"] == "FAIL":
                    overall_status = "FAIL"
                    confidence *= alignment_result["confidence"]
                    contradictions.extend(alignment_result["contradictions"])
                    clarification_needed = self.clarification_handler.generate_clarification(
                        claim_text, alignment_result["visual_description"], "alignment"
                    )
                else:
                    evidence.extend(alignment_result["evidence"])
            except Exception as e:
                logger.warning(f"Cross-alignment check failed: {str(e)}")
        
        # Check 2: External factuality
        if overall_status != "FAIL":
            logger.info("Performing external factuality check...")
            try:
                factuality_result = self.external_factuality_checker.verify_claim(claim_text)
                
                if factuality_result["status"] == "FAIL":
                    overall_status = "FAIL"
                    confidence *= factuality_result["confidence"]
                    contradictions.extend(factuality_result["contradictions"])
                    
                    if not clarification_needed:
                        clarification_needed = self.clarification_handler.generate_clarification(
                            claim_text, factuality_result["external_facts"], "contradiction"
                        )
                else:
                    evidence.extend(factuality_result["evidence"])
            except Exception as e:
                logger.warning(f"External factuality check failed: {str(e)}")
        
        return {
            "claim": claim_text,
            "status": overall_status,
            "confidence": confidence,
            "evidence": evidence,
            "contradictions": contradictions,
            "socratic_questions": socratic_questions,
            "clarification_needed": clarification_needed,
            "timestamp": datetime.now().isoformat()
        }
    
    def _compile_socratic_response(self, verification_results: list, original_input: str) -> dict:
        """Compile a comprehensive Socratic response"""
        logger.info("Compiling Socratic response...")
        
        # Analyze overall verification status
        passed_claims = [r for r in verification_results if r["status"] == "PASS"]
        failed_claims = [r for r in verification_results if r["status"] == "FAIL"]
        
        # Generate Socratic dialogue
        socratic_dialogue = []
        for result in verification_results:
            # Add Socratic questions
            for question in result.get("socratic_questions", []):
                socratic_dialogue.append({
                    "type": "socratic_question",
                    "content": question["question"],
                    "reasoning": question.get("reasoning", ""),
                    "confidence": question.get("confidence", 0.5)
                })
            
            # Add verification outcome
            if result["status"] == "PASS":
                socratic_dialogue.append({
                    "type": "verification_result",
                    "content": f"Through careful examination, the claim '{result['claim']}' appears to be consistent with available evidence.",
                    "evidence": result["evidence"]
                })
            else:
                socratic_dialogue.append({
                    "type": "contradiction_found",
                    "content": f"Upon investigation, the claim '{result['claim']}' appears to contradict available evidence.",
                    "contradictions": result["contradictions"]
                })
        
        # Compile response
        response = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_input": original_input,
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
        
        return response
    
    def _generate_next_steps(self, verification_results: list) -> list:
        """Generate suggested next steps"""
        next_steps = []
        
        failed_results = [r for r in verification_results if r["status"] == "FAIL"]
        
        if failed_results:
            next_steps.append("Review and clarify contradicted claims")
            for result in failed_results:
                if result.get("clarification_needed"):
                    next_steps.append(f"Clarification needed: {result['clarification_needed']}")
        
        if len(failed_results) == 0:
            next_steps.append("All claims verified - analysis complete")
        
        return next_steps

# Initialize Socrates Agent
socrates_agent = SimplifiedSocratesAgent()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new verification session"""
    try:
        session_id = socrates_agent.start_session()
        logger.info(f"Started new session: {session_id}")
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'New session started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start session: {str(e)}'
        }), 500

@app.route('/verify_claim', methods=['POST'])
def verify_claim():
    """Main endpoint for claim verification"""
    try:
        # Get form data
        user_input = request.form.get('user_input', '').strip()
        session_id = request.form.get('session_id', '')
        
        if not user_input:
            return jsonify({
                'status': 'error',
                'message': 'No input provided'
            }), 400
        
        # Handle file upload if present
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                logger.info(f"Image uploaded: {filename}")
        
        # Set session if not provided
        if not session_id:
            session_id = socrates_agent.start_session()
        
        # Process with Socrates Agent
        logger.info(f"Processing verification request: {user_input[:50]}...")
        result = socrates_agent.process_user_input(user_input, image_path)
        
        # Clean up uploaded file after processing
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass  # Ignore cleanup errors
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in claim verification: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': 'Socrates Agent System (Simplified)'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    logger.info("Starting Simplified Socrates Agent System...")
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info("System initialized successfully")
    app.run(debug=True, host='0.0.0.0', port=5000)
