"""
Local LLM Manager - Foundation for all local Llama 3.1 interactions
Provides unified interface for claim extraction, Socratic questioning, and reasoning
"""
import logging
import json
import asyncio
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import base64
import mimetypes
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
# Note: avoid importing heavy HF providers at module load time

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
    CONTRADICTION_DETECTION = "contradiction_detection"
    CONTRADICTION_DETECTION_SIMPLE = "contradiction_detection_simple"

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    LLAVA_HF = "llava_hf"
    LLAVA_ORIGINAL = "llava_original"

@dataclass
class LLMRequest:
    """Structured LLM request with context"""
    task_type: LLMTaskType
    prompt: str
    context: Dict[str, Any]
    images: Optional[List[str]] = None
    temperature: float = 0.7
    max_tokens: int = 4096
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

class LLMManager:
    """
    Local LLM Manager for Llama 3.1 integration
    Handles all local LLM operations with specialized prompting strategies
    """
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 provider: Union[LLMProvider, str, None] = None,
                 base_url: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 openai_base_url: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 anthropic_base_url: Optional[str] = None,
                 max_concurrent: int = 3):
        """
        Initialize Local LLM Manager
        
        Args:
            model_name: Model name for the selected provider
            provider: One of {"ollama","openai","claude"} (or LLMProvider)
            base_url: Ollama server URL (if provider=ollama)
            openai_api_key: OpenAI API key (or env OPENAI_API_KEY)
            openai_base_url: OpenAI base URL (default https://api.openai.com/v1)
            anthropic_api_key: Anthropic API key (or env ANTHROPIC_API_KEY)
            anthropic_base_url: Anthropic base URL (default https://api.anthropic.com)
            max_concurrent: Maximum concurrent requests
        """
        # Resolve provider
        prov_str = (provider.value if isinstance(provider, LLMProvider) else provider) or os.getenv("SOC_LLM_PROVIDER", "ollama").lower()
        try:
            self.provider = LLMProvider(prov_str)
        except Exception:
            logger.warning(f"Unknown provider '{prov_str}', defaulting to 'ollama'")
            self.provider = LLMProvider.OLLAMA

        # Resolve model
        env_model = os.getenv("SOC_LLM_MODEL")
        if model_name:
            self.model_name = model_name
        elif env_model:
            self.model_name = env_model
        else:
            # Reasonable defaults per provider
            if self.provider == LLMProvider.OLLAMA:
                self.model_name = "llama3.1:8b"
            elif self.provider == LLMProvider.OPENAI:
                self.model_name = "gpt-4o-mini"
            elif self.provider == LLMProvider.LLAVA_HF:
                self.model_name = "llava-hf/llava-1.5-7b-hf"
            elif self.provider == LLMProvider.LLAVA_ORIGINAL:
                # Use the original LLaVA implementation weights by default
                self.model_name = "liuhaotian/llava-v1.5-7b"
            else:  # CLAUDE
                self.model_name = "claude-3-haiku-20240307"

        # Provider-specific endpoints/keys
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434"
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_base_url = anthropic_base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"

        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = threading.Lock()
        
        # Initialize prompt templates
        self._init_prompt_templates()
        
        # Test connection
        self._test_connection()
        
        logger.info(f"LLMManager initialized with provider={self.provider.value}, model={self.model_name}")

    def generate_text(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.2, system_prompt: str = None, images: Optional[List[str]] = None) -> str:
        """Synchronous wrapper for generating text. For easy integration with non-async code."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self._call_llm(prompt, system_prompt, temperature, max_tokens, images=images)
            )
            return response
        finally:
            loop.close()
    
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
            ,
            LLMTaskType.CONTRADICTION_DETECTION: {
    "system": """You are a high-precision self-contradiction detection engine that relies ONLY on the provided session knowledge.

Your job: analyze a current claim against previously established session knowledge to detect contradictions.

IMPORTANT CONSTRAINT: Use ONLY the information given in Existing Claims and Entity Knowledge. Do NOT use external world knowledge or assumptions. If a potential contradiction would require outside knowledge (e.g., geography, domain facts) that is not present in the provided knowledge, treat it as NOT a contradiction.

Consider contradictions when there is strong, explicit conflict in the provided session knowledge:
- Direct negations ("is red" vs "is not red")
- Mutually exclusive attributes for the SAME entity ("is blue" vs "is red"), supported by the provided knowledge
- Relationship contradictions ("A is part of B" vs "A is separate from B") present in the session knowledge
- Numeric/temporal mismatches for the SAME event/entity in the provided knowledge

Consistency Rules (avoid false positives):
- Specialization is NOT a contradiction: specific instances/details that refine a general description (e.g., "red Toyota Camry" is compatible with "red car").
- Elaboration is NOT a contradiction: details that fit within a broader scene/context (e.g., "a businessman on the sidewalk" is compatible with "a busy street scene").
- Location within a broader area is NOT a contradiction unless the provided knowledge explicitly states it is elsewhere. Do not use outside maps.
- Unspecified vs specified is NOT a contradiction (e.g., existing claim lacks color; new claim adds color).
- When in doubt, default to PASS (no contradiction).

Output STRICT JSON ONLY. No prose, no code fences, no extra text.

Required JSON schema:
{
  "status": "PASS" | "FAIL",
  "confidence": float (0.0-1.0),
  "contradictions": [
    {"against": string, "type": string, "explanation": string, "severity": float}
  ],
  "conflicting_claims": [string],
  "evidence": [string]
}
""",
    
    "template": """Current Claim: "{claim}"

Existing Claims from Session:
{existing_claims}

Entity Knowledge Available:
Entities: {entities}
Entity Knowledge: {entity_knowledge}

Context (may include compatibility_hints): {context}

Analyze the current claim against ONLY the provided existing claims AND the rich entity knowledge. Apply the Consistency Rules. If entity knowledge shows an explicit attribute for the SAME entity that conflicts with the claim, flag a contradiction; otherwise default to PASS.

Respond ONLY with the JSON object matching the required schema."""
            },
            LLMTaskType.CONTRADICTION_DETECTION_SIMPLE: {
                "system": """ROLE: You are a precision contradiction detector. Your task is simple: detect conflicts between a new claim and established session knowledge.\n\nCONTRADICTION CRITERIA:\n- Direct negation: \"car is red\" vs \"car is not red\"\n- Mutually exclusive attributes: \"car is blue\" vs \"car is red\" (same entity)\n- Logical impossibility: conflicting facts about the same entity\n\nNOT CONTRADICTIONS:\n- Specialization: \"car\" → \"red Toyota Camry\"\n- Elaboration: \"street scene\" → \"person walking on street\"\n- Missing details: no previous color → \"red car\"\n\nOutput STRICT JSON ONLY in this schema:\n{\n  \"contradiction\": true/false,\n  \"confidence\": 0.0-1.0,\n  \"explanation\": \"brief reason\",\n  \"conflicting_fact\": \"specific contradicting statement or null\"\n}\n\nUse ONLY the provided session facts. If facts are missing or insufficient, set contradiction=false and explain briefly.""",
                "template": """Current Claim: "{claim}"

Session Facts:
{session_facts}

Respond with only the required JSON object."""
            }
        }

    def _test_connection(self):
        """Test connection to configured provider."""
        try:
            if self.provider == LLMProvider.OLLAMA:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [model.get('name') for model in models]
                    if self.model_name not in model_names:
                        logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    else:
                        logger.info(f"Connected to Ollama @ {self.base_url}. Model {self.model_name} is available.")
                else:
                    logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
            elif self.provider == LLMProvider.OPENAI:
                if not self.openai_api_key:
                    logger.warning("OPENAI_API_KEY not set; OpenAI calls will fail")
                # Try a lightweight models list
                url = f"{self.openai_base_url.rstrip('/')}/models"
                resp = requests.get(url, headers={"Authorization": f"Bearer {self.openai_api_key}"}, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"Connected to OpenAI @ {self.openai_base_url} (model={self.model_name})")
                else:
                    logger.warning(f"OpenAI connectivity check failed: HTTP {resp.status_code}")
            elif self.provider == LLMProvider.CLAUDE:
                if not self.anthropic_api_key:
                    logger.warning("ANTHROPIC_API_KEY not set; Claude calls will fail")
                # Try models endpoint if available
                url = f"{self.anthropic_base_url.rstrip('/')}/v1/models"
                resp = requests.get(url, headers={"x-api-key": self.anthropic_api_key, "anthropic-version": "2023-06-01"}, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"Connected to Anthropic @ {self.anthropic_base_url} (model={self.model_name})")
                else:
                    logger.warning(f"Anthropic connectivity check failed: HTTP {resp.status_code}")
            elif self.provider == LLMProvider.LLAVA_HF:
                try:
                    # Attempt lightweight import check
                    from mllm_evaluation.providers.llava_hf import LlavaHFGenerator  # noqa: F401
                    logger.info(f"Using local LLaVA-HF provider (model={self.model_name})")
                except Exception as e:
                    logger.warning(f"LLaVA-HF availability check failed: {e}")
            elif self.provider == LLMProvider.LLAVA_ORIGINAL:
                try:
                    # Try importing the original LLaVA package
                    import importlib
                    importlib.import_module("llava")
                    logger.info(f"Original LLaVA package detected (model={self.model_name})")
                except Exception as e:
                    logger.warning(
                        "Original LLaVA not importable. Ensure it's installed via 'pip install git+https://github.com/haotian-liu/LLaVA.git'. "
                        f"Detail: {e}"
                    )
        except Exception as e:
            logger.warning(f"LLM connectivity check encountered an error: {e}")

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
            
            # Make request to configured provider
            response = await self._call_llm(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                images=request.images
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
    
    async def _call_llm(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Dispatch call to the configured LLM provider."""
        if self.provider == LLMProvider.OLLAMA:
            return await self._call_ollama(prompt, system_prompt, temperature, max_tokens, images=images)
        elif self.provider == LLMProvider.OPENAI:
            return await self._call_openai(prompt, system_prompt, temperature, max_tokens, images=images)
        elif self.provider == LLMProvider.CLAUDE:
            return await self._call_claude(prompt, system_prompt, temperature, max_tokens, images=images)
        elif self.provider == LLMProvider.LLAVA_HF:
            return await self._call_llava_hf(prompt, system_prompt, temperature, max_tokens, images=images)
        elif self.provider == LLMProvider.LLAVA_ORIGINAL:
            return await self._call_llava_original(prompt, system_prompt, temperature, max_tokens, images=images)
        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

    async def _call_llava_hf(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Call local HuggingFace LLaVA model for multimodal generation.
        Notes:
        - Only the first image (if any) is used.
        - System prompt is ignored; LLaVA chat template handles roles internally.
        """
        # Lazy import to avoid heavy dependencies on module load
        from mllm_evaluation.providers.llava_hf import LlavaHFGenerator
        # Environment toggles for local inference
        no_4bit = str(os.getenv("SOC_LLAVA_NO_4BIT", "")).lower() in ("1", "true", "yes")
        use_slow_tok = str(os.getenv("SOC_LLAVA_SLOW_TOKENIZER", "")).lower() in ("1", "true", "yes")

        image_path = images[0] if images else None
        generator = LlavaHFGenerator.get(self.model_name, no_4bit=no_4bit, use_slow_tokenizer=use_slow_tok)

        loop = asyncio.get_event_loop()
        def _run():
            return generator.generate(
                prompt=prompt,
                image_path=image_path,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        return await loop.run_in_executor(self.executor, _run)

    async def _call_llava_original(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Call the original LLaVA implementation via its CLI (subprocess) or Python API.
        Notes:
        - Requires the original LLaVA repo installed: pip install git+https://github.com/haotian-liu/LLaVA.git
        - Only the first image is used.
        - Uses conversation templates from the original implementation; system_prompt is ignored.
        """
        image_path = images[0] if images else None
        if not image_path:
            raise ValueError("llava_original requires at least one image for multimodal prompting")

        # Environment toggles
        use_cli = str(os.getenv("SOC_LLAVA_ORIG_USE_CLI", "")).lower() in ("1", "true", "yes")
        conv_template = os.getenv("SOC_LLAVA_CONV_TEMPLATE", "llava_v1")
        timeout_sec = int(os.getenv("SOC_LLAVA_TIMEOUT_SEC", "180"))

        # Prefer in-process Python API when possible; fallback to CLI if requested or import fails
        if not use_cli:
            try:
                # Lazy imports for original LLaVA
                import torch  # noqa: F401
                from llava.conversation import conv_templates
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import (
                    get_model_name_from_path,
                    process_images,
                    tokenizer_image_token,
                    KeywordsStoppingCriteria,
                )
                from PIL import Image

                # Cache model components
                if not hasattr(self, "_llava_orig_cache"):
                    model_name_clean = get_model_name_from_path(self.model_name)
                    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path=self.model_name,
                        model_base=None,
                        model_name=model_name_clean,
                    )
                    self._llava_orig_cache = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "image_processor": image_processor,
                        "context_len": context_len,
                    }

                tok = self._llava_orig_cache["tokenizer"]
                model = self._llava_orig_cache["model"]
                image_processor = self._llava_orig_cache["image_processor"]

                # Build conversation
                if conv_template not in conv_templates:
                    conv = conv_templates["llava_v1"].copy()
                else:
                    conv = conv_templates[conv_template].copy()
                conv.append_message(conv.roles[0], f"<image>\n{prompt}")
                conv.append_message(conv.roles[1], None)
                question = conv.get_prompt()

                # Prepare image tensor
                from pathlib import Path
                img = Image.open(Path(image_path)).convert("RGB")
                images_proc = process_images([img], image_processor, model.config)

                # Tokenize with image token
                input_ids = tokenizer_image_token(question, tok, return_tensors='pt').unsqueeze(0).to(model.device)

                # Stopping criteria based on template separators
                stop_str = conv.sep if hasattr(conv, 'sep') else None
                stopping_criteria = None
                if stop_str:
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], tok, input_ids)

                # Run generation in threadpool to avoid blocking
                loop = asyncio.get_event_loop()
                def _run_generate():
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_proc,
                            do_sample=(temperature > 0.0),
                            temperature=temperature,
                            max_new_tokens=max_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                        )
                    out = tok.decode(output_ids[0], skip_special_tokens=True)
                    # Heuristic: take text after the last assistant marker if present
                    if conv.sep2 in out if hasattr(conv, 'sep2') else False:
                        out = out.split(conv.sep2)[-1].strip()
                    return out.strip()

                return await loop.run_in_executor(self.executor, _run_generate)
            except Exception as e:
                logger.warning(f"Falling back to LLaVA CLI due to Python API error: {e}")

        # Subprocess CLI fallback
        import shlex
        import subprocess
        loop = asyncio.get_event_loop()

        def _run_cli() -> str:
            cmd = [
                sys.executable, "-m", "llava.eval.model_vqa_loader",
                "--model-path", str(self.model_name),
                "--image-file", str(image_path),
                "--question", str(prompt),
                "--temperature", str(temperature),
                "--max-new-tokens", str(max_tokens),
            ]
            if conv_template:
                cmd += ["--conv-template", conv_template]
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                )
            except Exception as se:
                raise RuntimeError(f"Failed running LLaVA CLI: {se}")
            if proc.returncode != 0:
                raise RuntimeError(f"LLaVA CLI error (code {proc.returncode}): {proc.stderr.strip()}")

            # Parse output: take the last non-empty line; strip common prefixes
            out_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if not out_lines:
                return ""
            ans = out_lines[-1]
            for prefix in ("ASSISTANT:", "Assistant:", "assistant:"):
                if ans.startswith(prefix):
                    ans = ans[len(prefix):].strip()
                    break
            return ans

        return await loop.run_in_executor(self.executor, _run_cli)

    # -------------------- Image helpers (multimodal support) --------------------
    @staticmethod
    def _is_url(src: str) -> bool:
        try:
            u = urlparse(src)
            return u.scheme in ("http", "https")
        except Exception:
            return False

    @staticmethod
    def _is_data_url(src: str) -> bool:
        return isinstance(src, str) and src.startswith("data:image")

    @staticmethod
    def _guess_mime(src: str) -> str:
        mime, _ = mimetypes.guess_type(src)
        return mime or "image/jpeg"

    def _load_image_b64(self, src: str) -> Dict[str, str]:
        """Load image from local path or URL and return dict with keys: mime, b64, data_url, src.
        If src is a data URL, parse and return components.
        """
        # Data URL: data:image/<type>;base64,<b64>
        if self._is_data_url(src):
            try:
                header, b64 = src.split(",", 1)
                mime = "image/jpeg"
                if ";base64" in header:
                    mime = header.split(":", 1)[1].split(";", 1)[0]
                return {"mime": mime, "b64": b64, "data_url": src, "src": src}
            except Exception:
                return {"mime": "image/jpeg", "b64": "", "data_url": src, "src": src}

        # Remote URL
        if self._is_url(src):
            try:
                resp = requests.get(src, timeout=30)
                resp.raise_for_status()
                content = resp.content
                b64 = base64.b64encode(content).decode("utf-8")
                mime = resp.headers.get("Content-Type") or self._guess_mime(src)
                data_url = f"data:{mime};base64,{b64}"
                return {"mime": mime, "b64": b64, "data_url": data_url, "src": src}
            except Exception:
                mime = self._guess_mime(src)
                return {"mime": mime, "b64": "", "data_url": src, "src": src}

        # Local file path
        try:
            with open(src, "rb") as f:
                content = f.read()
            b64 = base64.b64encode(content).decode("utf-8")
            mime = self._guess_mime(src)
            data_url = f"data:{mime};base64,{b64}"
            return {"mime": mime, "b64": b64, "data_url": data_url, "src": src}
        except Exception as e:
            logger.warning(f"Failed to load image '{src}': {e}")
            return {"mime": "image/jpeg", "b64": "", "data_url": src, "src": src}

    async def _call_ollama(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
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
        # Attach images if provided (vision models like llama3.2-vision / llava)
        if images:
            imgs: List[str] = []
            for src in images:
                try:
                    info = self._load_image_b64(src)
                    if info.get("b64"):
                        imgs.append(info["b64"])
                except Exception:
                    continue
            if imgs:
                payload["images"] = imgs
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

    async def _call_openai(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Make async call to OpenAI Chat Completions API."""
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not provided")
        url = f"{self.openai_base_url.rstrip('/')}/chat/completions"
        # Build multimodal content if images provided
        user_content: Any
        if images:
            parts: List[Dict[str, Any]] = []
            parts.append({"type": "text", "text": prompt})
            for src in images:
                try:
                    info = self._load_image_b64(src)
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": info.get("data_url", src)}
                    })
                except Exception:
                    continue
            user_content = parts
        else:
            user_content = prompt
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, json=payload, headers=headers, timeout=180)
        )
        if response.status_code == 200:
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(data)
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

    async def _call_claude(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Make async call to Anthropic Claude Messages API."""
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not provided")
        url = f"{self.anthropic_base_url.rstrip('/')}/v1/messages"
        # Build content blocks for multimodal input
        content_blocks: Any
        if images:
            blocks: List[Dict[str, Any]] = []
            # Prefer text first for Claude context
            blocks.append({"type": "text", "text": prompt})
            for src in images:
                try:
                    info = self._load_image_b64(src)
                    if info.get("b64"):
                        blocks.append({
                            "type": "input_image",
                            "source": {
                                "type": "base64",
                                "media_type": info.get("mime", "image/jpeg"),
                                "data": info.get("b64", ""),
                            },
                        })
                except Exception:
                    continue
            content_blocks = blocks
        else:
            content_blocks = prompt
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt or "",
            "messages": [{"role": "user", "content": content_blocks}],
        }
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, json=payload, headers=headers, timeout=180)
        )
        if response.status_code == 200:
            data = response.json()
            try:
                # Claude returns a list of content blocks
                blocks = data.get("content", [])
                text_parts = []
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                return "\n".join([t for t in text_parts if t]) or json.dumps(data)
            except Exception:
                return json.dumps(data)
        raise Exception(f"Claude API error: {response.status_code} - {response.text}")
    
    def _parse_response(self, response: str, task_type: LLMTaskType) -> Dict[str, Any]:
        """Parse LLM response based on task type. Attempts to robustly parse JSON, including code-fenced blocks."""
        raw = response or ""
        s = raw.strip()
        # Strip code fences like ```json ... ```
        if s.startswith("```"):
            # Remove opening ```json or ```
            s = re.sub(r'^```[a-zA-Z]*\n', '', s)
            # Remove closing ```
            s = re.sub(r'\n```\s*$', '', s)
            s = s.strip()
        
        try:
            # Try direct JSON
            if s.startswith('{') or s.startswith('['):
                return json.loads(s)
        except json.JSONDecodeError:
            # Try to extract the largest JSON object within the string
            try:
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = s[start:end+1]
                    return json.loads(candidate)
            except Exception:
                pass
        
        # Fallback for non-JSON responses
        return {
            "content": raw,
            "confidence": 0.8,  # Default confidence
            "reasoning": "Generated by LLM"
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
    
    async def extract_claims(self, text: str, context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Extract claims from text"""
        request = LLMRequest(
            task_type=LLMTaskType.CLAIM_EXTRACTION,
            prompt="",
            context={"text": text, **(context or {})},
            images=images,
        )
        return await self.process_request(request)
    
    async def generate_socratic_questions(self, claim: str, context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Generate Socratic questions for a claim"""
        request = LLMRequest(
            task_type=LLMTaskType.SOCRATIC_QUESTIONING,
            prompt="",
            context={"claim": claim, "context": json.dumps(context or {})},
            images=images,
        )
        return await self.process_request(request)
    
    async def generate_reasoning(self, question: str, evidence: List[str], context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Generate reasoning for a question"""
        request = LLMRequest(
            task_type=LLMTaskType.REASONING_GENERATION,
            prompt="",
            context={
                "question": question,
                "evidence": json.dumps(evidence),
                "context": json.dumps(context or {})
            },
            images=images,
        )
        return await self.process_request(request)
    
    async def verify_claim(self, claim: str, evidence: List[str], context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Verify a claim against evidence"""
        request = LLMRequest(
            task_type=LLMTaskType.FACTUAL_VERIFICATION,
            prompt="",
            context={
                "claim": claim,
                "evidence": json.dumps(evidence),
                "context": json.dumps(context or {})
            },
            images=images,
        )
        return await self.process_request(request)
    
    async def extract_relationships(self, text: str, entities: List[str], context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Extract relationships between entities"""
        request = LLMRequest(
            task_type=LLMTaskType.RELATIONSHIP_EXTRACTION,
            prompt="",
            context={
                "text": text,
                "entities": json.dumps(entities),
                **(context or {})
            },
            images=images,
        )
        return await self.process_request(request)
    
    async def integrate_knowledge(self, new_info: Dict[str, Any], existing_knowledge: Dict[str, Any], context: Dict[str, Any] = None, images: Optional[List[str]] = None) -> LLMResponse:
        """Integrate new knowledge with existing knowledge"""
        request = LLMRequest(
            task_type=LLMTaskType.KNOWLEDGE_INTEGRATION,
            prompt="",
            context={
                "new_info": json.dumps(new_info),
                "existing_knowledge": json.dumps(existing_knowledge),
                "context": json.dumps(context or {})
            },
            images=images,
        )
        return await self.process_request(request)
    
    async def assess_faithfulness(self, original_claim: str, corrected_claim: str, evidence: List[str], images: Optional[List[str]] = None) -> LLMResponse:
        """Assess faithfulness of claim correction"""
        request = LLMRequest(
            task_type=LLMTaskType.FAITHFULNESS_ASSESSMENT,
            prompt="",
            context={
                "original_claim": original_claim,
                "corrected_claim": corrected_claim,
                "evidence": json.dumps(evidence)
            },
            images=images,
        )
        return await self.process_request(request)

    async def detect_contradictions(
        self,
        claim: str,
        existing_claims: List[str],
        context: Dict[str, Any] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        entity_knowledge: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Detect contradictions between a claim and existing session claims (async). Accepts optional entities and entity_knowledge for richer analysis."""
        request = LLMRequest(
            task_type=LLMTaskType.CONTRADICTION_DETECTION,
            prompt="",
            context={
                "claim": claim,
                "existing_claims": json.dumps(existing_claims or []),
                "entities": json.dumps(entities or []),
                "entity_knowledge": json.dumps(entity_knowledge or {}),
                "context": json.dumps(context or {})
            },
            temperature=0.2,
            max_tokens=800,
            images=images,
        )
        return await self.process_request(request)

    def detect_contradictions_sync(
        self,
        claim: str,
        existing_claims: List[str],
        context: Dict[str, Any] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        entity_knowledge: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Synchronous wrapper for contradiction detection (for non-async callers)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.detect_contradictions(
                    claim,
                    existing_claims,
                    context,
                    entities=entities,
                    entity_knowledge=entity_knowledge,
                )
            )
        finally:
            loop.close()
    
    async def detect_contradiction_simple(
        self,
        claim: str,
        session_facts: str,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Simplified, GraphRAG-style contradiction detection over linearized session facts (async)."""
        request = LLMRequest(
            task_type=LLMTaskType.CONTRADICTION_DETECTION_SIMPLE,
            prompt="",
            context={
                "claim": claim,
                "session_facts": session_facts or "(none)",
            },
            temperature=0.1,
            max_tokens=500,
            images=images,
        )
        return await self.process_request(request)
    
    def detect_contradiction_simple_sync(
        self,
        claim: str,
        session_facts: str,
    ) -> LLMResponse:
        """Synchronous wrapper for simplified contradiction detection."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.detect_contradiction_simple(claim, session_facts)
            )
        finally:
            loop.close()
    
    def shutdown(self):
        """Shutdown the LLM manager"""
        self.executor.shutdown(wait=True)
        logger.info("LLMManager shutdown complete")

# Global instance for easy access
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def shutdown_llm_manager():
    """Shutdown global LLM manager"""
    global _llm_manager
    if _llm_manager:
        _llm_manager.shutdown()
        _llm_manager = None
