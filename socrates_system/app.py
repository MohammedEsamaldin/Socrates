"""
Socrates Agent System - Main Flask Application
Advanced multimodal hallucination detection system with comprehensive fact-checking
"""
import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import json
from datetime import datetime

from core.socrates_agent import SocratesAgent
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, AGLA_API_URL, AGLA_API_VERIFY_PATH, AGLA_API_TIMEOUT
from modules.agla_client import AGLAClient
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure remote AGLA is ready before initializing heavy components
if AGLA_API_URL:
    try:
        logger.info("Waiting for remote AGLA API to become ready...")
        _agla_client_boot = AGLAClient(AGLA_API_URL, AGLA_API_VERIFY_PATH, AGLA_API_TIMEOUT)
        ready = _agla_client_boot.wait_until_ready(timeout=AGLA_API_TIMEOUT, interval=2.0)
        if not ready:
            logger.error("AGLA API is not ready. Aborting startup due to remote-only policy.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"AGLA readiness check encountered an error: {e}. Aborting startup due to remote-only policy.")
        sys.exit(1)

# Initialize Socrates Agent (will use remote AGLA if configured)
socrates_agent = SocratesAgent()

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

@app.route('/api/agla_verify', methods=['POST'])
def api_agla_verify():
    """API endpoint for AGLA cross-modal verification.
    Accepts multipart/form-data with fields:
      - image: uploaded file (required)
      - claim: textual claim (required)
      - socratic_question: optional Socratic question for context
      - use_agla, alpha, beta, return_debug: optional tuning params
    Returns JSON with boolean verdict and evidence list.
    """
    try:
        claim = request.form.get('claim', '').strip()
        soc_q = request.form.get('socratic_question', '').strip()

        if not claim:
            return jsonify({'status': 'error', 'message': 'Missing claim'}), 400

        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'Missing image file'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty image filename'}), 400
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'status': 'error', 'message': 'Empty image file'}), 400

        # Optional params
        use_agla_str = request.form.get('use_agla')
        alpha_str = request.form.get('alpha')
        beta_str = request.form.get('beta')
        return_debug = request.form.get('return_debug', 'false').lower() in ('1', 'true', 'yes')

        kwargs = {}
        if use_agla_str is not None:
            kwargs['use_agla'] = use_agla_str.lower() in ('1', 'true', 'yes')
        if alpha_str:
            try:
                kwargs['alpha'] = float(alpha_str)
            except Exception:
                pass
        if beta_str:
            try:
                kwargs['beta'] = float(beta_str)
            except Exception:
                pass

        # Remote-only AGLA verification
        if getattr(socrates_agent, 'agla_client', None) is not None:
            out = socrates_agent.agla_client.verify(
                image=image_bytes,
                claim=claim,
                socratic_question=soc_q,
                return_debug=return_debug,
                **kwargs,
            )
        else:
            return jsonify({'status': 'error', 'message': 'AGLA API not configured and local verifier disabled.'}), 503

        verdict_str = out.get('verdict', 'Uncertain')
        verdict_bool = True if verdict_str == 'True' else False if verdict_str == 'False' else None

        evidence = []
        if soc_q:
            evidence.append(f"Socratic question: {soc_q}")
        if verdict_str == 'False':
            truth = out.get('truth') or ''
            if truth:
                evidence.append(f"AGLA correction: {truth}")
        evidence.append(f"AGLA verdict: {verdict_str}")

        response = {
            'status': 'success',
            'verdict': verdict_bool,
            'evidence': evidence,
            'latency_ms': out.get('latency_ms'),
        }
        if return_debug and 'debug' in out:
            response['debug'] = out['debug']
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in AGLA verification: {str(e)}")
        return jsonify({'status': 'error', 'message': f'AGLA verification failed: {str(e)}'}), 500

@app.route('/session_summary/<session_id>')
def session_summary(session_id):
    """Get session summary"""
    try:
        # Set the session if not already set
        if socrates_agent.session_id != session_id:
            socrates_agent.session_id = session_id
        
        summary = socrates_agent.get_session_summary()
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting session summary: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get session summary: {str(e)}'
        }), 500

@app.route('/knowledge_graph/<session_id>')
def knowledge_graph(session_id):
    """Get knowledge graph for session"""
    try:
        kg_data = socrates_agent.kg_manager.export_session_graph(session_id)
        return jsonify({
            'status': 'success',
            'knowledge_graph': kg_data
        })
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get knowledge graph: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': 'Socrates Agent System'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting Socrates Agent System...")
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize knowledge graph manager connection
    try:
        socrates_agent.self_contradiction_checker.set_kg_manager(socrates_agent.kg_manager)
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
