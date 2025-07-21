"""
Local LLM Manager - Foundation for all local Llama 3.1 interactions
Provides unified interface for claim extraction, Socratic questioning, and reasoning
"""
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMTaskType(Enum):
    """Types of LLM tasks for specialized prompting"""
    CLAIM_EXTRACTION = "claim_extraction"
    SOCRATIC_QUESTIONING = "socratic_questioning"
    REASONING_GENERATION = "reasoning_generation"
    FACTUAL_VERIFICATION = "factual_verification"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    FAITHFULNESS_ASSESSMENT = "faithfulness_assessment"

@dataclass
class LLMRequest:
    """Structured LLM request with context"""
    task_type: LLMTaskType
    prompt: str
    context: Dict[str, Any]
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: Optional[str] = None

@dataclass
class LLMResponse:
    """Structured LLM response with metadata"""
    content: str
    task_type: LLMTaskType
    confidence: float
    reasoning: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class LocalLLMManager:
    """
    Local LLM Manager for Llama 3.1 integration
    Handles all local LLM operations with specialized prompting strategies
    """
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 max_concurrent: int = 3):
        """
        Initialize Local LLM Manager
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b")
            base_url: Ollama server URL
            max_concurrent: Maximum concurrent requests
        """
        self.model_name = model_name
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = threading.Lock()
        
        # Initialize prompt templates
        self._init_prompt_templates()
        
        # Test connection
        self._test_connection()
        
        logger.info(f"LocalLLMManager initialized with model: {model_name}")
    
    def _init_prompt_templates(self):
        """Initialize specialized prompt templates for different tasks"""
        self.prompt_templates = {
            LLMTaskType.CLAIM_EXTRACTION: {
                "system": """You are an expert claim extraction system. Your task is to identify factual claims, their relationships, and attributes from text.

Extract claims that are:
1. Factual statements that can be verified
2. Specific and concrete (not opinions or subjective statements)
3. Complete and self-contained

For each claim, identify:
- The main assertion
- Entities involved and their relationships
- Temporal/spatial context if present
- Confidence level (0.0-1.0)

Output format: JSON with 'claims' array containing objects with 'text', 'entities', 'relationships', 'context', 'confidence'.""",
                
                "template": """Text to analyze: "{text}"

Extract all factual claims from this text. Focus on verifiable statements and their relationships.

Response:"""
            },
            
            LLMTaskType.SOCRATIC_QUESTIONING: {
                "system": """You are a Socratic questioning expert. Generate probing questions that help verify claims through systematic inquiry.

Your questions should:
1. Challenge assumptions and probe deeper understanding
2. Reveal potential contradictions or gaps
3. Guide toward evidence-based reasoning
4. Be specific and actionable

Generate questions that would help verify the truthfulness of claims by examining evidence, context, and logical consistency.""",
                
                "template": """Claim to examine: "{claim}"
Context: {context}

Generate 3-5 Socratic questions that would help verify this claim. Each question should probe different aspects:
1. Evidence and sources
2. Logical consistency
3. Alternative explanations
4. Assumptions and biases
5. Broader implications

Response format: JSON with 'questions' array containing objects with 'question', 'reasoning', 'focus_area', 'expected_answer_type'."""
            },
            
            LLMTaskType.REASONING_GENERATION: {
                "system": """You are an expert reasoning system. Generate clear, logical reasoning chains that explain how conclusions are reached.

Your reasoning should:
1. Be step-by-step and transparent
2. Identify key assumptions
3. Show logical connections
4. Acknowledge uncertainties
5. Provide confidence assessments""",
                
                "template": """Question: {question}
Available Evidence: {evidence}
Context: {context}

Provide detailed reasoning for answering this question based on the available evidence. Include:
1. Step-by-step logical analysis
2. Key assumptions made
3. Confidence level and reasoning
4. Alternative interpretations if any

Response:"""
            },
            
            LLMTaskType.FACTUAL_VERIFICATION: {
                "system": """You are a factual verification expert. Assess the truthfulness of claims based on provided evidence.

Your verification should:
1. Compare claims against evidence systematically
2. Identify supporting and contradicting information
3. Assess reliability of sources
4. Provide confidence scores with justification
5. Highlight areas needing additional verification""",
                
                "template": """Claim: "{claim}"
Evidence: {evidence}
Context: {context}

Verify this claim against the provided evidence. Provide:
1. Verification status (SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE)
2. Supporting evidence points
3. Contradicting evidence points
4. Confidence score (0.0-1.0) with justification
5. Areas needing additional verification

Response format: JSON with 'status', 'supporting_evidence', 'contradicting_evidence', 'confidence', 'reasoning', 'additional_verification_needed'."""
            },
            
            LLMTaskType.RELATIONSHIP_EXTRACTION: {
                "system": """You are an expert at extracting semantic relationships between entities in text.

Extract relationships that are:
1. Explicit and implicit connections between entities
2. Temporal, causal, and logical relationships
3. Hierarchical and categorical relationships
4. Functional and role-based relationships

Focus on relationships that are factually verifiable and contextually significant.""",
                
                "template": """Text: "{text}"
Entities: {entities}

Extract all meaningful relationships between the identified entities. For each relationship:
1. Source entity
2. Relationship type
3. Target entity
4. Confidence level
5. Supporting context

Response format: JSON with 'relationships' array containing objects with 'source', 'relation', 'target', 'confidence', 'context'."""
            },
            
            LLMTaskType.KNOWLEDGE_INTEGRATION: {
                "system": """You are a knowledge integration expert following KALA methodology. Integrate new information with existing knowledge while preserving consistency.

Your integration should:
1. Identify connections with existing knowledge
2. Resolve conflicts and contradictions
3. Maintain knowledge graph consistency
4. Preserve important relationships
5. Update confidence levels appropriately""",
                
                "template": """New Information: {new_info}
Existing Knowledge: {existing_knowledge}
Context: {context}

Integrate the new information with existing knowledge. Provide:
1. Updated knowledge structure
2. Resolved conflicts
3. New relationships discovered
4. Confidence updates
5. Integration reasoning

Response format: JSON with 'updated_knowledge', 'conflicts_resolved', 'new_relationships', 'confidence_updates', 'integration_reasoning'."""
            },
            
            LLMTaskType.FAITHFULNESS_ASSESSMENT: {
                "system": """You are a faithfulness assessment expert following Zero-shot Faithful Factual Error Correction methodology.

Assess faithfulness by:
1. Comparing claims against evidence systematically
2. Identifying factual errors and inconsistencies
3. Evaluating correction quality
4. Measuring consistency with evidence
5. Providing interpretable assessments""",
                
                "template": """Original Claim: "{original_claim}"
Corrected Claim: "{corrected_claim}"
Evidence: {evidence}

Assess the faithfulness of the correction. Provide:
1. Faithfulness score (0.0-1.0)
2. Factual accuracy assessment
3. Consistency with evidence
4. Error identification
5. Correction quality evaluation

Response format: JSON with 'faithfulness_score', 'accuracy_assessment', 'evidence_consistency', 'errors_identified', 'correction_quality', 'reasoning'."""
            }
        }
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                else:
                    logger.info(f"Successfully connected to Ollama. Model {self.model_name} is available.")
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single LLM request asynchronously"""
        start_time = time.time()
        
        try:
            # Get appropriate prompt template
            template_config = self.prompt_templates.get(request.task_type)
            if not template_config:
                raise ValueError(f"No template found for task type: {request.task_type}")
            
            # Build full prompt
            system_prompt = request.system_prompt or template_config["system"]
            full_prompt = template_config["template"].format(**request.context) if request.context else request.prompt
            
            # Make request to Ollama
            response = await self._call_ollama(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Parse response
            parsed_response = self._parse_response(response, request.task_type)
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=response,
                task_type=request.task_type,
                confidence=parsed_response.get('confidence', 0.8),
                reasoning=parsed_response.get('reasoning'),
                structured_output=parsed_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LLM request failed: {e}")
            
            return LLMResponse(
                content="",
                task_type=request.task_type,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _call_ollama(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Make async call to Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    def _parse_response(self, response: str, task_type: LLMTaskType) -> Dict[str, Any]:
        """Parse LLM response based on task type"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{') or response.strip().startswith('['):
                return json.loads(response)
            
            # For non-JSON responses, create structured output
            return {
                "content": response,
                "confidence": 0.8,  # Default confidence
                "reasoning": "Generated by LLM"
            }
            
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            return {
                "content": response,
                "confidence": 0.7,
                "reasoning": "Response parsing failed, using raw content"
            }
    
    async def batch_process(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple requests concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.process_request(request)
        
        tasks = [process_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    # Convenience methods for specific tasks
    
    async def extract_claims(self, text: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Extract claims from text"""
        request = LLMRequest(
            task_type=LLMTaskType.CLAIM_EXTRACTION,
            prompt="",
            context={"text": text, **(context or {})}
        )
        return await self.process_request(request)
    
    async def generate_socratic_questions(self, claim: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate Socratic questions for a claim"""
        request = LLMRequest(
            task_type=LLMTaskType.SOCRATIC_QUESTIONING,
            prompt="",
            context={"claim": claim, "context": json.dumps(context or {})}
        )
        return await self.process_request(request)
    
    async def generate_reasoning(self, question: str, evidence: List[str], context: Dict[str, Any] = None) -> LLMResponse:
        """Generate reasoning for a question"""
        request = LLMRequest(
            task_type=LLMTaskType.REASONING_GENERATION,
            prompt="",
            context={
                "question": question,
                "evidence": json.dumps(evidence),
                "context": json.dumps(context or {})
            }
        )
        return await self.process_request(request)
    
    async def verify_claim(self, claim: str, evidence: List[str], context: Dict[str, Any] = None) -> LLMResponse:
        """Verify a claim against evidence"""
        request = LLMRequest(
            task_type=LLMTaskType.FACTUAL_VERIFICATION,
            prompt="",
            context={
                "claim": claim,
                "evidence": json.dumps(evidence),
                "context": json.dumps(context or {})
            }
        )
        return await self.process_request(request)
    
    async def extract_relationships(self, text: str, entities: List[str], context: Dict[str, Any] = None) -> LLMResponse:
        """Extract relationships between entities"""
        request = LLMRequest(
            task_type=LLMTaskType.RELATIONSHIP_EXTRACTION,
            prompt="",
            context={
                "text": text,
                "entities": json.dumps(entities),
                **(context or {})
            }
        )
        return await self.process_request(request)
    
    async def integrate_knowledge(self, new_info: Dict[str, Any], existing_knowledge: Dict[str, Any], context: Dict[str, Any] = None) -> LLMResponse:
        """Integrate new knowledge with existing knowledge"""
        request = LLMRequest(
            task_type=LLMTaskType.KNOWLEDGE_INTEGRATION,
            prompt="",
            context={
                "new_info": json.dumps(new_info),
                "existing_knowledge": json.dumps(existing_knowledge),
                "context": json.dumps(context or {})
            }
        )
        return await self.process_request(request)
    
    async def assess_faithfulness(self, original_claim: str, corrected_claim: str, evidence: List[str]) -> LLMResponse:
        """Assess faithfulness of claim correction"""
        request = LLMRequest(
            task_type=LLMTaskType.FAITHFULNESS_ASSESSMENT,
            prompt="",
            context={
                "original_claim": original_claim,
                "corrected_claim": corrected_claim,
                "evidence": json.dumps(evidence)
            }
        )
        return await self.process_request(request)
    
    def shutdown(self):
        """Shutdown the LLM manager"""
        self.executor.shutdown(wait=True)
        logger.info("LocalLLMManager shutdown complete")

# Global instance for easy access
_llm_manager = None

def get_llm_manager() -> LocalLLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LocalLLMManager()
    return _llm_manager

def shutdown_llm_manager():
    """Shutdown global LLM manager"""
    global _llm_manager
    if _llm_manager:
        _llm_manager.shutdown()
        _llm_manager = None
