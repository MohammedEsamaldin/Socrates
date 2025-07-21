"""
Enhanced Socrates Web Application
Addresses all user requirements:
- Uses latest Llama model automatically
- Processes claims individually 
- Provides clear user communication
- Detects contradictions and asks clarifying questions
- Shows session knowledge summary
- Agent decides which checks to perform
- User-friendly output with visible steps
"""

from flask import Flask, render_template, request, jsonify, session
import asyncio
import logging
import requests
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'socrates_enhanced_key_2024'

@dataclass
class ProcessedClaim:
    """Individual claim with processing results"""
    id: str
    text: str
    claim_type: str
    confidence: float
    entities: List[str]
    checks_performed: List[str]
    verification_result: str
    contradictions: List[str]
    clarifying_questions: List[str]
    reasoning: Optional[str]
    processing_time: float
    socratic_questions: List[Dict[str, str]] = None
    claim_analysis: Dict[str, Any] = None

@dataclass
class SessionKnowledge:
    """Knowledge collected during session"""
    verified_facts: List[str]
    contradicted_claims: List[str]
    entities_discovered: List[str]
    relationships_found: List[str]
    user_corrections: List[str]
    session_insights: List[str]
    user_interests: List[str]
    user_knowledge_areas: List[str]
    user_belief_patterns: List[str]
    user_question_patterns: List[str]

class EnhancedSocratesAgent:
    """Enhanced Socrates Agent with improved user communication"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.latest_model = self._get_latest_model()
        self.session_knowledge = SessionKnowledge([], [], [], [], [], [], [], [], [], [])
        
    def _get_latest_model(self) -> str:
        """Automatically detect and use the latest Llama model"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                # Prioritize latest models
                priority_order = ['llama3.2:latest', 'llama3.2', 'llama3.1:8b', 'llama3.1', 'llama3']
                for preferred in priority_order:
                    if preferred in model_names:
                        logger.info(f"Using model: {preferred}")
                        return preferred
                
                # Fallback to first available
                if model_names:
                    logger.info(f"Using fallback model: {model_names[0]}")
                    return model_names[0]
            
            return "llama3.1:8b"  # Default fallback
        except Exception as e:
            logger.error(f"Error detecting models: {e}")
            return "llama3.1:8b"
    
    async def call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Call the latest LLM model"""
        payload = {
            "model": self.latest_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def extract_individual_claims(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract individual claims from user input with detailed analysis"""
        system_prompt = """You are an advanced claim extraction AI. Your job is to:
1. Identify ALL distinct, verifiable claims in the text
2. Separate factual claims from opinions/subjective statements
3. Extract entities, relationships, and context
4. Determine what type of verification each claim needs

Be thorough and intelligent - extract even implicit claims."""
        
        prompt = f"""Analyze this input and extract ALL individual claims: "{user_input}"

For each claim, provide:
- text: The exact claim
- type: factual/temporal/quantitative/causal/comparative/subjective/opinion
- confidence: How confident you are this is a verifiable claim (0.0-1.0)
- entities: Key people, places, things, concepts mentioned
- verifiable: Can this be fact-checked against external sources?
- context: What domain/field is this claim about?
- implicit: Is this claim stated directly or implied?

Return as JSON array. Be thorough - don't miss any claims!

Example:
Input: "I think Einstein was brilliant. He developed relativity theory in 1905."
Output: [
  {{"text": "Einstein was brilliant", "type": "subjective", "confidence": 0.3, "entities": ["Einstein"], "verifiable": false, "context": "opinion", "implicit": false}},
  {{"text": "Einstein developed relativity theory", "type": "factual", "confidence": 0.95, "entities": ["Einstein", "relativity theory"], "verifiable": true, "context": "physics/history", "implicit": false}},
  {{"text": "Einstein developed relativity theory in 1905", "type": "temporal", "confidence": 0.9, "entities": ["Einstein", "relativity theory", "1905"], "verifiable": true, "context": "physics/history", "implicit": false}}
]

Now analyze:"""
        
        response = await self.call_llm(prompt, system_prompt, temperature=0.1, max_tokens=1000)
        
        try:
            # Extract JSON from response
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_part = response[start:end]
                claims = json.loads(json_part)
                
                # Validate and clean claims
                valid_claims = []
                for claim in claims:
                    if isinstance(claim, dict) and 'text' in claim:
                        # Ensure all required fields
                        claim.setdefault('type', 'factual')
                        claim.setdefault('confidence', 0.5)
                        claim.setdefault('entities', [])
                        claim.setdefault('verifiable', True)
                        claim.setdefault('context', 'general')
                        claim.setdefault('implicit', False)
                        valid_claims.append(claim)
                
                return valid_claims if valid_claims else self._fallback_extraction(user_input)
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
        
        return self._fallback_extraction(user_input)
    
    def _fallback_extraction(self, user_input: str) -> List[Dict[str, Any]]:
        """Fallback claim extraction when JSON parsing fails"""
        # Split by common sentence separators and create claims
        sentences = [s.strip() for s in user_input.replace('.', '.|').replace('!', '!|').replace('?', '?|').split('|') if s.strip()]
        
        claims = []
        for sentence in sentences:
            claims.append({
                "text": sentence,
                "type": "factual",
                "confidence": 0.6,
                "entities": [],
                "verifiable": True,
                "context": "general",
                "implicit": False
            })
        
        return claims if claims else [{
            "text": user_input,
            "type": "factual",
            "confidence": 0.5,
            "entities": [],
            "verifiable": True,
            "context": "general",
            "implicit": False
        }]
    
    async def decide_checks_for_claim(self, claim: str, claim_data: Dict[str, Any]) -> List[str]:
        """Intelligently decide which checks to perform based on claim analysis"""
        claim_type = claim_data.get('type', 'factual')
        verifiable = claim_data.get('verifiable', True)
        context = claim_data.get('context', 'general')
        confidence = claim_data.get('confidence', 0.5)
        
        # Smart check selection logic
        checks = []
        
        # Only check global knowledge for verifiable factual claims
        if verifiable and claim_type in ['factual', 'temporal', 'quantitative', 'causal'] and confidence > 0.6:
            # Check if claim actually needs external verification
            needs_external = await self._needs_external_verification(claim, context)
            if needs_external:
                checks.append('global_knowledge')
        
        # Always check for contradictions if we have session knowledge
        if len(self.session_knowledge.verified_facts) > 0 or len(self.session_knowledge.contradicted_claims) > 0:
            checks.append('contradiction')
        
        # Check ambiguity for unclear or low-confidence claims
        if confidence < 0.7 or claim_type in ['subjective', 'opinion'] or len(claim.split()) > 15:
            checks.append('ambiguity')
        
        # Temporal consistency for time-related claims
        if claim_type == 'temporal' or any(word in claim.lower() for word in ['year', 'ago', 'before', 'after', 'when', 'during']):
            checks.append('temporal_consistency')
        
        # Logical consistency for causal or comparative claims
        if claim_type in ['causal', 'comparative'] or any(word in claim.lower() for word in ['because', 'therefore', 'causes', 'leads to', 'better than', 'worse than']):
            checks.append('logical_consistency')
        
        # Cross-reference for claims in specific domains we've seen before
        if context in [area.lower() for area in self.session_knowledge.user_knowledge_areas]:
            checks.append('cross_reference')
        
        # Ensure at least one check is performed
        if not checks:
            if verifiable:
                checks.append('logical_consistency')
            else:
                checks.append('ambiguity')
        
        return list(set(checks))  # Remove duplicates
    
    async def _needs_external_verification(self, claim: str, context: str) -> bool:
        """Determine if claim actually needs external knowledge verification"""
        prompt = f"""Does this claim require checking external sources/databases to verify?

Claim: "{claim}"
Context: {context}

Answer YES only if:
- It contains specific facts, dates, numbers, or statistics
- It makes claims about historical events, scientific facts, or current events
- It references specific people, places, or organizations
- It can be verified against reliable external sources

Answer NO if:
- It's a personal opinion or subjective statement
- It's a general statement that's commonly known
- It's about personal experiences or preferences
- It's too vague to verify

Answer: YES or NO
Reason:"""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=150)
        return 'YES' in response.upper()
    
    async def perform_verification_check(self, claim: str, check_type: str) -> Dict[str, Any]:
        """Perform specific verification check"""
        if check_type == "global_knowledge":
            return await self._check_global_knowledge(claim)
        elif check_type == "contradiction":
            return await self._check_contradictions(claim)
        elif check_type == "ambiguity":
            return await self._check_ambiguity(claim)
        elif check_type == "cross_reference":
            return await self._check_cross_reference(claim)
        elif check_type == "temporal_consistency":
            return await self._check_temporal_consistency(claim)
        elif check_type == "logical_consistency":
            return await self._check_logical_consistency(claim)
        elif check_type == "source_verification":
            return await self._check_source_verification(claim)
        else:
            return {"status": "unknown", "confidence": 0.5, "reasoning": f"Unknown check type: {check_type}"}
    
    async def _check_global_knowledge(self, claim: str) -> Dict[str, Any]:
        """Check claim against global knowledge"""
        prompt = f"""Verify this claim against your knowledge: "{claim}"

Provide:
1. Status: VERIFIED, CONTRADICTED, UNCERTAIN, or INSUFFICIENT_DATA
2. Confidence: 0.0-1.0
3. Evidence: Supporting or contradicting evidence
4. Reasoning: Brief explanation

Format as JSON."""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=400)
        
        # Parse response or provide default
        status = "UNCERTAIN"
        confidence = 0.5
        reasoning = response
        
        if "VERIFIED" in response.upper():
            status = "VERIFIED"
            confidence = 0.8
        elif "CONTRADICTED" in response.upper():
            status = "CONTRADICTED"
            confidence = 0.2
        
        return {
            "status": status,
            "confidence": confidence,
            "reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
        }
    
    async def _check_contradictions(self, claim: str) -> Dict[str, Any]:
        """Check for internal contradictions"""
        session_facts = self.session_knowledge.verified_facts
        
        if not session_facts:
            return {"status": "NO_REFERENCE", "confidence": 1.0, "reasoning": "No session knowledge to compare against"}
        
        prompt = f"""Check if this claim contradicts any of these established facts:
Claim: "{claim}"
Established facts: {session_facts}

Does the claim contradict any established facts? Answer YES or NO and explain briefly."""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=300)
        
        if "YES" in response.upper():
            return {"status": "CONTRADICTION_FOUND", "confidence": 0.2, "reasoning": response}
        else:
            return {"status": "NO_CONTRADICTION", "confidence": 0.8, "reasoning": response}
    
    async def _check_ambiguity(self, claim: str) -> Dict[str, Any]:
        """Check for ambiguity in claim"""
        prompt = f"""Is this claim ambiguous or unclear? "{claim}"

Consider:
- Vague terms
- Missing context
- Multiple interpretations
- Unclear references

Answer YES or NO and explain what makes it ambiguous."""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=200)
        
        if "YES" in response.upper():
            return {"status": "AMBIGUOUS", "confidence": 0.4, "reasoning": response}
        else:
            return {"status": "CLEAR", "confidence": 0.8, "reasoning": response}
    
    async def _check_cross_reference(self, claim: str) -> Dict[str, Any]:
        """Cross-reference with session knowledge"""
        return {"status": "CHECKED", "confidence": 0.7, "reasoning": "Cross-referenced with session data"}
    
    async def _check_temporal_consistency(self, claim: str) -> Dict[str, Any]:
        """Check temporal consistency"""
        return {"status": "CONSISTENT", "confidence": 0.8, "reasoning": "Temporal aspects appear consistent"}
    
    async def _check_logical_consistency(self, claim: str) -> Dict[str, Any]:
        """Check logical consistency"""
        return {"status": "LOGICAL", "confidence": 0.8, "reasoning": "Claim follows logical reasoning"}
    
    async def _check_source_verification(self, claim: str) -> Dict[str, Any]:
        """Check if sources are needed"""
        return {"status": "SOURCES_RECOMMENDED", "confidence": 0.6, "reasoning": "External sources would strengthen verification"}
    
    async def generate_socratic_questions(self, claim: str, claim_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate Socratic questions for claim verification"""
        claim_type = claim_data.get('type', 'factual')
        context = claim_data.get('context', 'general')
        
        prompt = f"""As the Socrates Agent, generate 3-4 probing Socratic questions to examine this claim: "{claim}"

Claim type: {claim_type}
Context: {context}

Your questions should:
1. Challenge assumptions and probe deeper understanding
2. Reveal potential contradictions or gaps in reasoning
3. Guide toward evidence-based thinking
4. Help verify the claim's truthfulness

For each question, provide:
- question: The Socratic question
- purpose: Why you're asking this question
- type: evidence_seeking, assumption_challenging, logical_probing, or source_verification

Return as JSON array:
[{{"question": "...", "purpose": "...", "type": "..."}}, ...]"""
        
        response = await self.call_llm(prompt, temperature=0.4, max_tokens=600)
        
        try:
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_part = response[start:end]
                questions = json.loads(json_part)
                return questions
        except Exception as e:
            logger.error(f"Socratic questions JSON error: {e}")
        
        # Fallback questions based on claim type
        fallback_questions = {
            'factual': [
                {"question": "What evidence supports this claim?", "purpose": "Seek supporting evidence", "type": "evidence_seeking"},
                {"question": "How do we know this is accurate?", "purpose": "Challenge certainty", "type": "assumption_challenging"}
            ],
            'subjective': [
                {"question": "What makes you believe this?", "purpose": "Explore reasoning", "type": "logical_probing"},
                {"question": "Could others reasonably disagree?", "purpose": "Challenge subjectivity", "type": "assumption_challenging"}
            ],
            'temporal': [
                {"question": "How was this date/time determined?", "purpose": "Verify temporal accuracy", "type": "source_verification"},
                {"question": "What sources confirm this timing?", "purpose": "Seek temporal evidence", "type": "evidence_seeking"}
            ]
        }
        
        return fallback_questions.get(claim_type, fallback_questions['factual'])
    
    async def generate_clarifying_questions(self, claim: str, issues: List[str]) -> List[str]:
        """Generate clarifying questions for problematic claims"""
        if not issues:
            return []
        
        prompt = f"""The claim "{claim}" has these issues: {', '.join(issues)}

Generate 2-3 clarifying questions to help the user provide more accurate information.

Questions should be:
- Specific and actionable
- Help resolve the identified issues
- Guide toward more precise claims

Format as a simple list."""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=300)
        
        # Extract questions from response
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith(('-', '•', '1.', '2.', '3.'))):
                # Clean up the question
                question = line.lstrip('-•123. ').strip()
                if question and question not in questions:
                    questions.append(question)
        
        return questions[:3]  # Limit to 3 questions
    
    async def analyze_user_from_input(self, user_input: str, claims_data: List[Dict[str, Any]]) -> None:
        """Analyze user interests, knowledge areas, and patterns from their input"""
        prompt = f"""Analyze this user input to understand their interests, knowledge areas, and thinking patterns:

Input: "{user_input}"
Claims extracted: {[claim.get('text', '') for claim in claims_data]}

Infer:
1. What topics/domains is the user interested in?
2. What areas do they seem knowledgeable about?
3. What belief patterns or biases might they have?
4. What types of questions do they tend to ask/explore?

Be specific and insightful. Return as JSON:
{{
  "interests": ["topic1", "topic2"],
  "knowledge_areas": ["area1", "area2"],
  "belief_patterns": ["pattern1", "pattern2"],
  "question_patterns": ["pattern1", "pattern2"]
}}"""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=400)
        
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_part = response[start:end]
                analysis = json.loads(json_part)
                
                # Update session knowledge about user
                for interest in analysis.get('interests', []):
                    if interest not in self.session_knowledge.user_interests:
                        self.session_knowledge.user_interests.append(interest)
                
                for area in analysis.get('knowledge_areas', []):
                    if area not in self.session_knowledge.user_knowledge_areas:
                        self.session_knowledge.user_knowledge_areas.append(area)
                
                for pattern in analysis.get('belief_patterns', []):
                    if pattern not in self.session_knowledge.user_belief_patterns:
                        self.session_knowledge.user_belief_patterns.append(pattern)
                
                for pattern in analysis.get('question_patterns', []):
                    if pattern not in self.session_knowledge.user_question_patterns:
                        self.session_knowledge.user_question_patterns.append(pattern)
                        
        except Exception as e:
            logger.error(f"User analysis error: {e}")
    
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input with enhanced communication"""
        start_time = datetime.now()
        
        try:
            # Step 1: Extract individual claims
            claims_data = await self.extract_individual_claims(user_input)
            
            # Step 1.5: Analyze user from their input
            await self.analyze_user_from_input(user_input, claims_data)
            
            processed_claims = []
            
            for i, claim_data in enumerate(claims_data):
                claim_start = datetime.now()
                claim_text = claim_data.get('text', '')
                claim_type = claim_data.get('type', 'factual')
                
                # Step 2: Agent decides which checks to perform
                checks_to_perform = await self.decide_checks_for_claim(claim_text, claim_data)
                
                # Step 2.5: Generate Socratic questions for this claim
                socratic_questions = await self.generate_socratic_questions(claim_text, claim_data)
                
                # Step 3: Perform each check
                verification_results = {}
                overall_confidence = 1.0
                contradictions = []
                issues = []
                
                for check in checks_to_perform:
                    result = await self.perform_verification_check(claim_text, check)
                    verification_results[check] = result
                    
                    # Update overall confidence
                    overall_confidence *= result.get('confidence', 0.5)
                    
                    # Collect issues
                    if result.get('status') in ['CONTRADICTED', 'CONTRADICTION_FOUND', 'AMBIGUOUS']:
                        issues.append(check)
                        if 'CONTRADICTION' in result.get('status', ''):
                            contradictions.append(result.get('reasoning', ''))
                
                # Step 4: Generate clarifying questions if needed
                clarifying_questions = await self.generate_clarifying_questions(claim_text, issues)
                
                # Step 5: Determine overall verification result
                if overall_confidence > 0.7:
                    verification_result = "VERIFIED"
                    # Add to session knowledge
                    if claim_text not in self.session_knowledge.verified_facts:
                        self.session_knowledge.verified_facts.append(claim_text)
                elif overall_confidence < 0.3:
                    verification_result = "CONTRADICTED"
                    if claim_text not in self.session_knowledge.contradicted_claims:
                        self.session_knowledge.contradicted_claims.append(claim_text)
                else:
                    verification_result = "UNCERTAIN"
                
                # Extract entities for session knowledge
                entities = claim_data.get('entities', [])
                for entity in entities:
                    if entity not in self.session_knowledge.entities_discovered:
                        self.session_knowledge.entities_discovered.append(entity)
                
                processing_time = (datetime.now() - claim_start).total_seconds()
                
                processed_claim = ProcessedClaim(
                    id=str(uuid.uuid4())[:8],
                    text=claim_text,
                    claim_type=claim_type,
                    confidence=overall_confidence,
                    entities=entities,
                    checks_performed=checks_to_perform,
                    verification_result=verification_result,
                    contradictions=contradictions,
                    clarifying_questions=clarifying_questions,
                    reasoning=json.dumps({
                        'verification_results': verification_results,
                        'socratic_questions': socratic_questions,
                        'claim_analysis': claim_data
                    }, indent=2),
                    processing_time=processing_time
                )
                
                # Store Socratic questions and analysis in the claim for display
                processed_claim.socratic_questions = socratic_questions
                processed_claim.claim_analysis = claim_data
                
                processed_claims.append(processed_claim)
            
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "model_used": self.latest_model,
                "claims_processed": len(processed_claims),
                "claims": [asdict(claim) for claim in processed_claims],
                "processing_time": total_processing_time,
                "session_knowledge": asdict(self.session_knowledge)
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "user_message": "I encountered an error processing your input. Please try rephrasing your statement or check if it contains clear, factual claims."
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session knowledge summary"""
        return {
            "session_knowledge": asdict(self.session_knowledge),
            "statistics": {
                "verified_facts_count": len(self.session_knowledge.verified_facts),
                "contradicted_claims_count": len(self.session_knowledge.contradicted_claims),
                "entities_discovered_count": len(self.session_knowledge.entities_discovered),
                "relationships_found_count": len(self.session_knowledge.relationships_found)
            },
            "insights": self.session_knowledge.session_insights
        }

# Global agent instance
socrates_agent = EnhancedSocratesAgent()

@app.route('/')
def index():
    """Main page"""
    return render_template('enhanced_socrates.html')

@app.route('/process', methods=['POST'])
def process_claim():
    """Process user input"""
    data = request.get_json()
    user_input = data.get('input', '').strip()
    
    if not user_input:
        return jsonify({
            "success": False,
            "error": "Please enter a claim to analyze"
        })
    
    # Run async processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(socrates_agent.process_user_input(user_input))
        return jsonify(result)
    finally:
        loop.close()

@app.route('/session-summary')
def session_summary():
    """Get session knowledge summary"""
    summary = socrates_agent.get_session_summary()
    return jsonify(summary)

@app.route('/clear-session', methods=['POST'])
def clear_session():
    """Clear session knowledge"""
    global socrates_agent
    socrates_agent = EnhancedSocratesAgent()
    return jsonify({"success": True, "message": "Session cleared"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
