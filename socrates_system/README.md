# Socrates Agent System - External Hallucination Detection

## Overview

The Socrates Agent System is a sophisticated, modular implementation for detecting and mitigating external hallucinations in Multimodal Large Language Models (MLLMs). This system implements advanced Socratic methodology for comprehensive fact verification across text and image modalities.

## Architecture

### Core Components

1. **Socrates Agent** (`core/socrates_agent.py`)
   - Central coordinator implementing Socratic dialogue methodology
   - Manages the complete verification pipeline
   - Orchestrates all verification modules

2. **Claim Extractor** (`modules/claim_extractor.py`)
   - Advanced NLP-based claim extraction using spaCy and transformers
   - Identifies factual claims, relationships, and attributes
   - Supports multiple claim types: attribute, relationship, temporal, comparative

3. **Question Generator** (`modules/question_generator.py`)
   - Sophisticated Socratic question generation
   - Context-aware question templates for different verification scenarios
   - Generates follow-up questions for deeper inquiry

4. **Cross-Alignment Checker** (`modules/cross_alignment_checker.py`)
   - Multimodal consistency verification between text and images
   - Uses BLIP vision-language model for image understanding
   - Semantic similarity analysis for alignment scoring

5. **External Factuality Checker** (`modules/external_factuality_checker.py`)
   - Real-world fact verification using multiple sources
   - Wikipedia API integration
   - Expandable knowledge base with semantic matching
   - Web search simulation (extensible to real APIs)

6. **Knowledge Graph Manager** (`modules/knowledge_graph_manager.py`)
   - Advanced entity and relation extraction
   - NetworkX-based graph construction and querying
   - SQLite persistence for session data
   - Contradiction detection against established knowledge

7. **Self-Contradiction Checker** (`modules/self_contradiction_checker.py`)
   - Session consistency verification
   - Checks claims against established session knowledge

8. **Ambiguity Checker** (`modules/ambiguity_checker.py`)
   - Detects ambiguous terms and unclear statements
   - Generates clarification questions for vague claims

9. **Clarification Handler** (`modules/clarification_handler.py`)
   - Manages user clarifications and corrections
   - Processes clarification responses and updates claims

## Verification Pipeline

The Socrates Agent implements a comprehensive 4-stage verification process:

### Stage 1: Claim Extraction
- Extracts factual claims from user input using advanced NLP
- Identifies entities, relationships, and claim types
- Calculates confidence scores for extracted claims

### Stage 2: Factuality Checks (Socratic Methodology)

#### 2.1 Cross-Alignment Check (Multimodal)
- Verifies consistency between textual claims and visual content
- Uses BLIP model for image captioning and analysis
- Detects contradictions between text and image

#### 2.2 External Factuality Check
- Verifies claims against external knowledge sources
- Wikipedia API integration for authoritative information
- Knowledge base lookup with semantic similarity matching
- Aggregates results from multiple sources

#### 2.3 Self-Contradiction Check
- Checks claims against session knowledge graph
- Identifies contradictions with previously verified claims
- Maintains consistency within conversation context

#### 2.4 Ambiguity Check
- Identifies ambiguous terms and unclear statements
- Generates clarification questions for problematic claims
- Ensures precise understanding before verification

### Stage 3: Knowledge Base Update
- Updates session knowledge graph with verified claims
- Extracts entities and relationships for graph construction
- Maintains persistent storage using SQLite

## Features
- run python -m socrates_system.pipeline --help for full CLI command list
### Multimodal Support
- Image upload and processing capabilities
- Vision-language model integration (BLIP)
- Cross-modal consistency verification

### Advanced NLP
- spaCy integration for entity recognition
- Sentence transformers for semantic similarity
- Pattern-based and dependency-based relation extraction

### Real External Sources
- Wikipedia API integration
- Expandable knowledge base system
- Web search framework (extensible)

### Knowledge Graph
- Entity and relation extraction
- NetworkX-based graph operations
- SQLite persistence
- Query and contradiction detection capabilities

### Socratic Methodology
- Context-aware question generation
- Multiple question types (verification, clarification, deeper analysis)
- Follow-up question sequences
- Reasoning explanation for each inquiry

## Installation

1. **Clone the repository:**
```bash
cd /Users/mohammed/Desktop/Socrates/Socrates/socrates_system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

4. **Run the application:**
```bash
python app.py
```

5. **Access the web interface:**
Open `http://localhost:5000` in your browser

## Usage

### Web Interface
1. Start a new session
2. Enter a factual claim for verification
3. Optionally upload an image for multimodal verification
4. Review the comprehensive Socratic analysis results

### API Endpoints
- `POST /start_session` - Start a new verification session
- `POST /verify_claim` - Verify a claim with optional image
- `GET /session_summary/<session_id>` - Get session summary
- `GET /knowledge_graph/<session_id>` - Export knowledge graph
- `GET /api/health` - Health check

### Example Claims
- "Paris is the capital of France"
- "The Eiffel Tower is located in Rome" (contradiction test)
- "Water boils at 100 degrees Celsius"
- "The sky is green" (with image for cross-alignment)

## Configuration

Key settings in `config.py`:
- Model configurations (BLIP, sentence transformers, spaCy)
- Database paths and settings
- Confidence thresholds
- File upload settings

## Technical Details

### Dependencies
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **spaCy**: Advanced NLP library
- **sentence-transformers**: Semantic similarity
- **NetworkX**: Graph operations
- **SQLite**: Persistent storage
- **Wikipedia**: Wikipedia API client
- **Pillow**: Image processing

### Models Used
- **BLIP**: `Salesforce/blip-image-captioning-base` for vision-language understanding
- **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- **spaCy**: `en_core_web_sm` for entity recognition

### Data Storage
- **SQLite Database**: Persistent storage for entities, relations, and claims
- **Session Graphs**: In-memory NetworkX graphs for fast operations
- **File Uploads**: Temporary storage for image processing

## System Capabilities

### Claim Types Supported
- **Attribute Claims**: "X is Y", "X has Y"
- **Relationship Claims**: "X is located in Y", "X belongs to Y"
- **Temporal Claims**: "X happened in Y", "X was built in Y"
- **Comparative Claims**: "X is bigger than Y"

### Verification Methods
- **Pattern Matching**: Rule-based claim extraction
- **Semantic Similarity**: Transformer-based similarity scoring
- **Knowledge Base Lookup**: Direct and semantic matching
- **External API Integration**: Wikipedia and web search
- **Graph-based Reasoning**: Knowledge graph querying

### Multimodal Features
- **Image Captioning**: Automatic description generation
- **Visual-Textual Alignment**: Consistency checking
- **Contradiction Detection**: Cross-modal inconsistency identification

## Future Enhancements

1. **Internal Hallucination Detection**: Extend to model output verification
2. **Advanced RAG**: Implement full retrieval-augmented generation
3. **Real Web Search**: Integrate with Google/Bing APIs
4. **Enhanced Vision Models**: Support for more sophisticated vision-language models
5. **Batch Processing**: Support for multiple claim verification
6. **API Integration**: RESTful API for external system integration

## Research Context

This system implements the external hallucination detection stage of a comprehensive thesis on detecting and mitigating factual hallucinations in Multimodal LLMs. The Socratic methodology provides a principled approach to fact verification through systematic questioning and evidence gathering.

## License

This project is developed for academic research purposes as part of a thesis on multimodal hallucination detection.

## Contact

For questions or contributions related to this research project, please refer to the thesis documentation or contact the research team.
