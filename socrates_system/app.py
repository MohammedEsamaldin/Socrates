"""
Socrates Agent System - Main Flask Application
Advanced multimodal hallucination detection system with comprehensive fact-checking
"""
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import json
from datetime import datetime

from core.socrates_agent import SocratesAgent
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize Socrates Agent
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
