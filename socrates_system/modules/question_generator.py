"""
Question Generation Module - Sophisticated Socratic Question Generation
Implements advanced question generation strategies for different verification contexts
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from ..utils.logger import setup_logger
from .llm_manager import get_llm_manager
import re
from collections import OrderedDict

logger = setup_logger(__name__)

class QuestionType(Enum):
    VERIFICATION = "verification"
    CLARIFICATION = "clarification"
    DEEPER_ANALYSIS = "deeper_analysis"
    CONSISTENCY = "consistency"
    AMBIGUITY = "ambiguity"
    CROSS_ALIGNMENT = "cross_alignment"

@dataclass
class SocraticInquiry:
    """Represents a sophisticated Socratic inquiry"""
    question: str
    reasoning: str
    expected_answer_type: str
    confidence: float
    context: Dict[str, Any]
    follow_up_questions: List[str]

class QuestionGenerator:
    """
    Advanced Socratic Question Generator
    Creates contextually appropriate questions for different verification scenarios
    """
    
    def __init__(self):
        logger.info("Initializing Question Generator...")
        
        # Question templates for different contexts
        self.question_templates = {
            QuestionType.VERIFICATION: {
                'attribute': [
                    "What evidence supports the claim that {subject} {predicate}?",
                    "How can we independently verify that {subject} {predicate}?",
                    "What authoritative sources confirm that {subject} {predicate}?",
                    "Under what conditions would {subject} {predicate} be true?",
                    "What would contradict the assertion that {subject} {predicate}?"
                ],
                'relationship': [
                    "What establishes the relationship between {entity1} and {entity2}?",
                    "How is the connection between {entity1} and {entity2} documented?",
                    "What evidence links {entity1} to {entity2}?",
                    "Through what mechanism are {entity1} and {entity2} related?",
                    "What would sever the relationship between {entity1} and {entity2}?"
                ],
                'temporal': [
                    "What historical records support this temporal claim?",
                    "How is this date or time period verified?",
                    "What contemporary sources document this timing?",
                    "What sequence of events led to this temporal assertion?",
                    "How does this timing align with other known events?"
                ],
                'general': [
                    "What foundational evidence supports this claim?",
                    "How might we test the validity of this assertion?",
                    "What assumptions underlie this statement?",
                    "What would need to be true for this claim to hold?",
                    "How does this claim relate to established knowledge?"
                ]
            },
            QuestionType.CLARIFICATION: [
                "Could you elaborate on what you mean by '{ambiguous_term}'?",
                "When you say '{claim}', are you referring to {interpretation1} or {interpretation2}?",
                "What specific context are you considering for this claim?",
                "Could you provide more details about '{unclear_aspect}'?",
                "What evidence would you consider most relevant to this claim?"
            ],
            QuestionType.DEEPER_ANALYSIS: [
                "What underlying assumptions does this claim rest upon?",
                "How might cultural or contextual factors influence this assertion?",
                "What are the implications if this claim were false?",
                "What related claims would also need to be true?",
                "How does this claim fit within the broader domain of knowledge?"
            ],
            QuestionType.CONSISTENCY: [
                "How does this claim align with your previous statements about {topic}?",
                "What reconciles this assertion with the earlier claim that {previous_claim}?",
                "How do we resolve the apparent contradiction between this and {conflicting_claim}?",
                "What additional context might explain these seemingly inconsistent claims?",
                "Which of these conflicting assertions should take precedence and why?"
            ],
            QuestionType.AMBIGUITY: [
                "What specific interpretation of '{ambiguous_term}' are you using?",
                "Could this statement be understood in multiple ways?",
                "What additional information would make this claim more precise?",
                "Are there implicit assumptions in this statement that should be made explicit?",
                "What scope or limitations apply to this claim?"
            ],
            QuestionType.CROSS_ALIGNMENT: [
                "How does what you've described align with what we observe in the image?",
                "What visual elements support or contradict your textual claim?",
                "How might the visual context change the interpretation of your statement?",
                "What details in the image are most relevant to verifying this claim?",
                "How do we reconcile differences between the visual and textual information?"
            ]
        }
        
        # Reasoning templates for different question types
        self.reasoning_templates = {
            QuestionType.VERIFICATION: "This question seeks to establish independent evidence for the claim, following the Socratic principle of examining the foundations of knowledge.",
            QuestionType.CLARIFICATION: "This inquiry aims to resolve ambiguity and ensure precise understanding, as clarity is essential for meaningful verification.",
            QuestionType.DEEPER_ANALYSIS: "This question probes the underlying structure and implications of the claim, revealing hidden assumptions and connections.",
            QuestionType.CONSISTENCY: "This inquiry examines internal consistency within the knowledge framework, identifying potential contradictions that require resolution.",
            QuestionType.AMBIGUITY: "This question addresses linguistic or conceptual ambiguity that could lead to misinterpretation or false verification.",
            QuestionType.CROSS_ALIGNMENT: "This inquiry examines the consistency between different modalities of information, ensuring multimodal coherence."
        }
        
        logger.info("Question Generator initialized successfully")
    
    def generate_socratic_inquiry(self, claim: str, question_type: str, 
                                context: Optional[Dict[str, Any]] = None) -> SocraticInquiry:
        """
        Generate a sophisticated Socratic inquiry for a given claim and context
        
        Args:
            claim: The claim to generate questions for
            question_type: Type of question needed
            context: Additional context for question generation
            
        Returns:
            SocraticInquiry object with question and metadata
        """
        logger.info(f"Generating {question_type} question for claim: {claim[:50]}...")
        
        try:
            q_type = QuestionType(question_type)
            context = context or {}
            
            # Parse claim for key components
            claim_components = self._parse_claim_components(claim)
            
            # Generate primary question
            primary_question = self._generate_primary_question(claim, q_type, claim_components, context)
            
            # Generate follow-up questions
            follow_ups = self._generate_follow_up_questions(claim, q_type, claim_components, context)
            
            # Determine expected answer type
            expected_answer_type = self._determine_expected_answer_type(claim, q_type)
            
            # Calculate confidence
            confidence = self._calculate_question_confidence(claim, q_type, claim_components)
            
            # Get reasoning
            reasoning = self._generate_reasoning(q_type, claim, context)
            
            inquiry = SocraticInquiry(
                question=primary_question,
                reasoning=reasoning,
                expected_answer_type=expected_answer_type,
                confidence=confidence,
                context=context,
                follow_up_questions=follow_ups
            )
            
            logger.info(f"Generated Socratic inquiry: {primary_question[:50]}...")
            return inquiry
            
        except Exception as e:
            logger.error(f"Error generating Socratic inquiry: {str(e)}")
            # Return fallback question
            return SocraticInquiry(
                question=f"What evidence supports the claim: '{claim}'?",
                reasoning="Fallback question for basic verification",
                expected_answer_type="evidence",
                confidence=0.5,
                context=context or {},
                follow_up_questions=[]
            )
    
    def _parse_claim_components(self, claim: str) -> Dict[str, Any]:
        """Parse claim into components for question generation"""
        components = {
            'full_claim': claim,
            'subject': None,
            'predicate': None,
            'entities': [],
            'claim_type': 'general'
        }
        
        # Simple parsing for common patterns
        claim_lower = claim.lower().strip()
        
        # Attribute pattern: X is Y
        if ' is ' in claim_lower:
            parts = claim.split(' is ', 1)
            if len(parts) == 2:
                components['subject'] = parts[0].strip()
                components['predicate'] = parts[1].strip()
                components['claim_type'] = 'attribute'
        
        # Relationship pattern: X is in Y, X belongs to Y, etc.
        relationship_indicators = [' is in ', ' belongs to ', ' is part of ', ' is located in ']
        for indicator in relationship_indicators:
            if indicator in claim_lower:
                parts = claim.split(indicator, 1)
                if len(parts) == 2:
                    components['entity1'] = parts[0].strip()
                    components['entity2'] = parts[1].strip()
                    components['claim_type'] = 'relationship'
                break
        
        # Extract potential entities (simple heuristic)
        words = claim.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 1]
        components['entities'] = capitalized_words
        
        return components
    
    def _generate_primary_question(self, claim: str, q_type: QuestionType, 
                                 components: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate the primary Socratic question"""
        
        if q_type == QuestionType.VERIFICATION:
            claim_type = components.get('claim_type', 'general')
            templates = self.question_templates[q_type].get(claim_type, 
                                                          self.question_templates[q_type]['general'])
            template = random.choice(templates)
            
            # Fill in template with claim components
            if claim_type == 'attribute' and components.get('subject') and components.get('predicate'):
                return template.format(
                    subject=components['subject'],
                    predicate=components['predicate']
                )
            elif claim_type == 'relationship' and components.get('entity1') and components.get('entity2'):
                return template.format(
                    entity1=components['entity1'],
                    entity2=components['entity2']
                )
            else:
                return template.replace('{subject}', claim).replace('{predicate}', '').strip()
        
        else:
            templates = self.question_templates[q_type]
            template = random.choice(templates)
            
            # Fill in context-specific information
            if '{claim}' in template:
                template = template.replace('{claim}', claim)
            if '{ambiguous_term}' in template and context.get('ambiguous_terms'):
                template = template.replace('{ambiguous_term}', context['ambiguous_terms'][0])
            if '{topic}' in template and context.get('topic'):
                template = template.replace('{topic}', context['topic'])
            if '{previous_claim}' in template and context.get('previous_claim'):
                template = template.replace('{previous_claim}', context['previous_claim'])
            if '{conflicting_claim}' in template and context.get('conflicting_claim'):
                template = template.replace('{conflicting_claim}', context['conflicting_claim'])
            
            return template
    
    def _generate_follow_up_questions(self, claim: str, q_type: QuestionType, 
                                    components: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions for deeper inquiry"""
        follow_ups = []
        
        if q_type == QuestionType.VERIFICATION:
            follow_ups = [
                f"What would constitute sufficient evidence for '{claim}'?",
                f"What alternative explanations might account for the observations related to '{claim}'?",
                f"How might we distinguish between correlation and causation in '{claim}'?"
            ]
        
        elif q_type == QuestionType.CLARIFICATION:
            follow_ups = [
                "What specific examples would illustrate your point?",
                "How would you define the key terms in your statement?",
                "What context is most relevant to understanding this claim?"
            ]
        
        elif q_type == QuestionType.CONSISTENCY:
            follow_ups = [
                "What principles guide the resolution of such contradictions?",
                "How do we weigh conflicting pieces of evidence?",
                "What additional information might reconcile these differences?"
            ]
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _determine_expected_answer_type(self, claim: str, q_type: QuestionType) -> str:
        """Determine what type of answer is expected"""
        answer_types = {
            QuestionType.VERIFICATION: "evidence_and_reasoning",
            QuestionType.CLARIFICATION: "clarification_and_context",
            QuestionType.DEEPER_ANALYSIS: "analysis_and_implications",
            QuestionType.CONSISTENCY: "reconciliation_strategy",
            QuestionType.AMBIGUITY: "disambiguation",
            QuestionType.CROSS_ALIGNMENT: "multimodal_consistency_check"
        }
        return answer_types.get(q_type, "general_response")
    
    def _calculate_question_confidence(self, claim: str, q_type: QuestionType, 
                                     components: Dict[str, Any]) -> float:
        """Calculate confidence in the generated question"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence for well-structured claims
        if components.get('subject') and components.get('predicate'):
            confidence += 0.1
        
        # Boost confidence for claims with entities
        if components.get('entities'):
            confidence += 0.1
        
        # Boost confidence for specific question types
        if q_type in [QuestionType.VERIFICATION, QuestionType.CLARIFICATION]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, q_type: QuestionType, claim: str, context: Dict[str, Any]) -> str:
        """Generate reasoning for why this question is being asked"""
        base_reasoning = self.reasoning_templates[q_type]
        
        # Add context-specific reasoning
        if context.get('contradictions'):
            base_reasoning += f" Specifically, this addresses contradictions found in: {context['contradictions'][:100]}..."
        elif context.get('ambiguous_terms'):
            base_reasoning += f" This focuses on clarifying ambiguous terms: {', '.join(context['ambiguous_terms'][:3])}."
        
        return base_reasoning
    
    def generate_question_sequence(self, claim: str, verification_context: Dict[str, Any]) -> List[SocraticInquiry]:
        """Generate a sequence of related Socratic questions for comprehensive verification"""
        logger.info(f"Generating question sequence for claim: {claim[:50]}...")
        
        sequence = []
        
        # Start with verification
        verification_q = self.generate_socratic_inquiry(claim, "verification")
        sequence.append(verification_q)
        
        # Add clarification if needed
        if verification_context.get('needs_clarification'):
            clarification_q = self.generate_socratic_inquiry(claim, "clarification", verification_context)
            sequence.append(clarification_q)
        
        # Add deeper analysis
        deeper_q = self.generate_socratic_inquiry(claim, "deeper_analysis", verification_context)
        sequence.append(deeper_q)
        
        # Add consistency check if there are previous claims
        if verification_context.get('previous_claims'):
            consistency_q = self.generate_socratic_inquiry(claim, "consistency", verification_context)
            sequence.append(consistency_q)
        
        logger.info(f"Generated sequence of {len(sequence)} questions")
        return sequence

# ===================== New Category-Aware Socratic Question Generator =====================

@dataclass
class SocraticQuestion:
    """Structured Socratic question for downstream verification"""
    question: str
    category: str
    verification_hint: str
    confidence_score: float
    fallback: bool = False


@dataclass
class VerificationCapabilities:
    """Describes available downstream verification capabilities"""
    visual_grounding: List[str]
    external_knowledge: List[str]
    self_consistency: List[str]


class SocraticGeneratorError(Exception):
    """Base exception for Socratic question generation"""
    pass


class LLMGenerationError(SocraticGeneratorError):
    """LLM failed to generate questions"""
    pass


class ValidationError(SocraticGeneratorError):
    """Generated questions failed validation"""
    pass


class AmbiguityResolutionError(SocraticGeneratorError):
    """Failed to resolve ambiguous terms"""
    pass


class SocraticConfig:
    def __init__(self):
        self.questions_per_claim = 1
        self.questions_per_category = 1
        self.enable_fallback = True
        self.min_confidence_threshold = 0.50
        self.max_question_complexity_ratio = 1.6
        self.prioritize_visual_grounding = True
        # Retry settings for primary LLM generation before using fallback
        self.primary_retries = 2
        self.retry_temperature_increment = 0.2

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LLMInterfaceAdapter:
    """Adapter to bridge to local LLMManager with a simple generate() API"""
    def __init__(self, llm_manager=None):
        self.manager = llm_manager or get_llm_manager()

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        try:
            return self.manager.generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            raise LLMGenerationError(str(e))

    def generate_with_config(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 512)
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)


class QuestionValidator:
    """Ensures generated questions meet quality standards"""
    def __init__(self, max_complexity_ratio: float = 1.2, min_confidence_threshold: float = 0.7):
        self.max_complexity_ratio = max_complexity_ratio
        self.min_confidence_threshold = min_confidence_threshold

    def validate_questions(self, questions: List[str], original_claim: str, category: str) -> List[SocraticQuestion]:
        valid_questions: List[SocraticQuestion] = []
        for q in questions:
            score = self._calculate_quality_score(q, original_claim, category)
            if score >= self.min_confidence_threshold:
                valid_questions.append(
                    SocraticQuestion(
                        question=q.strip(),
                        category=category,
                        verification_hint=self._generate_verification_hint(category),
                        confidence_score=float(round(score, 3)),
                        fallback=False,
                    )
                )
        return valid_questions

    def _calculate_quality_score(self, question: str, claim: str, category: str) -> float:
        # Complexity check (lightly weighted)
        claim_len = max(1, len(claim.split()))
        question_len = max(1, len(question.split()))
        complexity_ratio = question_len / claim_len
        complexity_score = 1.0 if complexity_ratio <= self.max_complexity_ratio else 0.6

        # Relevance check (token overlap)
        claim_terms = set(re.findall(r"[a-zA-Z0-9_]+", claim.lower()))
        question_terms = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
        relevance_score = (len(claim_terms.intersection(question_terms)) / max(1, len(claim_terms)))

        # Category appropriateness
        category_score = self._check_category_appropriateness(question, category)

        # Question form score (does it look like a question?)
        q_strip = question.strip()
        has_qmark = q_strip.endswith('?') or ('?' in q_strip)
        q_lower = q_strip.lower().lstrip('\'" (')
        starters = (
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'is', 'are', 'does', 'do', 'did', 'can', 'could', 'would', 'will', 'was', 'were'
        )
        starts_ok = any(q_lower.startswith(s) for s in starters)
        if has_qmark and starts_ok:
            question_form_score = 1.0
        elif has_qmark:
            question_form_score = 0.7
        else:
            question_form_score = 0.2

        # Weighted combination: relevance and category matter most
        weights = {
            'complexity': 0.10,
            'relevance': 0.35,
            'category':  0.35,
            'form':      0.20,
        }
        score = (
            weights['complexity'] * complexity_score +
            weights['relevance']  * relevance_score +
            weights['category']   * category_score +
            weights['form']       * question_form_score
        )
        return float(score)

    def _check_category_appropriateness(self, question: str, category: str) -> float:
        category_keywords = {
            'VISUAL_GROUNDING_REQUIRED': [
                'see', 'image', 'visible', 'show', 'appear', 'display', 'in the image',
                'photo', 'picture', 'does the image', 'shown', 'visible in the image'
            ],
            'EXTERNAL_KNOWLEDGE_REQUIRED': [
                'fact', 'true', 'official', 'documented', 'known', 'according to',
                'evidence', 'source', 'reliable', 'verified', 'encyclopedia', 'database', 'record'
            ],
            'AMBIGUOUS_RESOLUTION_REQUIRED': [
                'mean', 'define', 'clarify', 'specify', 'in this context',
                'what do you mean', 'clarification', 'ambiguous', 'which do you mean'
            ],
            'SELF_CONSISTENCY_REQUIRED': [
                'consistent', 'contradict', 'previous', 'prior', 'knowledge graph',
                'earlier', 'previously', 'prior statements', 'inconsistent', 'conflict'
            ],
        }
        keywords = category_keywords.get(category, [])
        ql = question.lower()
        matches = sum(1 for kw in keywords if kw in ql)
        return min(matches / 2.0, 1.0)

    def _generate_verification_hint(self, category: str) -> str:
        mapping = {
            'VISUAL_GROUNDING_REQUIRED': 'Use visual analysis (object_detection, spatial_relationships)',
            'EXTERNAL_KNOWLEDGE_REQUIRED': 'Use external factual sources (Wikipedia, FactCheck, Wikidata)',
            'AMBIGUOUS_RESOLUTION_REQUIRED': 'Resolve ambiguity via user clarification before verification',
            'SELF_CONSISTENCY_REQUIRED': 'Check consistency against knowledge graph and prior claims',
        }
        return mapping.get(category, 'Use appropriate verification method')


class AmbiguityDetector:
    """Identifies and resolves ambiguous terms in claims using LLM assistance"""
    def __init__(self, llm_interface: LLMInterfaceAdapter):
        self.llm = llm_interface

    def detect_ambiguous_terms(self, claim: str) -> Dict[str, List[str]]:
        prompt = f"""
Analyze this claim for ambiguous terms: "{claim}"

STRICT CRITERIA: Only flag terms as ambiguous if they genuinely have multiple distinct meanings that would affect factual verification.

Common ambiguous patterns:
- Comparative adjectives without clear reference (faster, better, larger)
- Context-dependent nouns (apple, bank, ball)
- Subjective descriptors (beautiful, good, bad)

For each ambiguous term, provide 2-4 possible interpretations.

Format:
AMBIGUOUS_TERMS:
term1: interpretation1, interpretation2
term2: interpretation1, interpretation2

If no terms are meaningfully ambiguous for verification purposes, respond: "NO_AMBIGUITY"
"""
        response = self.llm.generate(prompt, temperature=0.1, max_tokens=300)
        return self._parse_ambiguity_response(response)

    def _parse_ambiguity_response(self, response: str) -> Dict[str, List[str]]:
        if not response or response.strip().upper().startswith("NO_AMBIGUITY"):
            return {}
        terms: Dict[str, List[str]] = {}
        started = False
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("AMBIGUOUS_TERMS"):
                started = True
                continue
            if not started:
                continue
            if ':' in line:
                term, interps = line.split(':', 1)
                term = term.strip()
                interpretations = [i.strip() for i in interps.split(',') if i.strip()]
                if term and interpretations:
                    terms[term] = interpretations[:4]
        return terms


class SocraticQuestionGenerator:
    """Primary category-aware Socratic Question Generator"""
    def __init__(self, verification_capabilities: VerificationCapabilities, llm_interface: Optional[Any] = None, config: Optional[SocraticConfig] = None):
        self.capabilities = verification_capabilities
        self.llm = llm_interface or LLMInterfaceAdapter()
        self.config = config or SocraticConfig()
        self.category_prompts = self._build_capability_aware_prompts()
        self.question_validator = QuestionValidator(
            max_complexity_ratio=self.config.max_question_complexity_ratio,
            min_confidence_threshold=self.config.min_confidence_threshold,
        )
        self.ambiguity_detector = AmbiguityDetector(self.llm)

    def _build_capability_aware_prompts(self) -> Dict[str, str]:
        return {
            'VISUAL_GROUNDING_REQUIRED': f"""
You are generating verification questions for visual claims.

CONSTRAINT: Our visual verification system can only:
{', '.join(self.capabilities.visual_grounding)}

Given claim: "{{claim}}"

Generate {{num_questions}} precise question(s) that:
1. Can be answered using ONLY the listed visual capabilities
2. Focus on observable evidence (not interpretations)
3. Use simple, direct language
4. Don't assume complex reasoning
 5. Each question MUST start with a question word (What/Which/Is/Are/Does/Do/Can/How/etc) and end with a '?' character

AVOID questions requiring: speed measurement, emotion detection, future prediction, complex inference

Format: Return only the question(s), one per line.
""",
            'EXTERNAL_KNOWLEDGE_REQUIRED': f"""
You are generating factual verification questions.

CONSTRAINT: Our external verification uses:
{', '.join(self.capabilities.external_knowledge)}

Given claim: "{{claim}}"

Generate {{num_questions}} question(s) that:
1. Can be fact-checked using reliable external sources
2. Focus on verifiable facts (not opinions)
3. Are specific enough for database/API queries
4. Don't require real-time data unless specified
 5. Each question MUST start with a question word (What/Which/Is/Are/Does/Do/Can/How/etc) and end with a '?' character

Examples: "What is the official top speed of [car model]?", "Is [entity] officially recognized as [attribute]?"

Format: Return only the question(s), one per line.
""",
            'AMBIGUOUS_RESOLUTION_REQUIRED': """
You are generating disambiguation questions for ambiguous claims.

Given potentially ambiguous claim: "{claim}"

Generate {num_questions} question(s) that:
1. Identify which terms could have multiple meanings
2. Ask for clarification WITHOUT suggesting interpretations
3. Focus on the most critical ambiguity for verification

Format: "What do you mean by [ambiguous_term] in this context?"

CRITICAL: Don't provide interpretation options - just ask for clarification of the ambiguous term.

Format: Return only the question(s), one per line.
""",
            'SELF_CONSISTENCY_REQUIRED': f"""
You are generating consistency verification questions.

CONSTRAINT: Our consistency checking compares against:
{', '.join(self.capabilities.self_consistency)}

Given claim: "{{claim}}"

Generate {{num_questions}} question(s) that:
1. Help identify what prior knowledge to check against
2. Focus on factual consistency (not subjective opinions)
3. Target specific entities or relationships
 4. Each question MUST start with a question word (What/Which/Is/Are/Does/Do/Can/How/etc) and end with a '?' character

Examples: "What previous claims about [entity] should this be consistent with?"

Format: Return only the question(s), one per line.
""",
        }

    def _normalize_category(self, category: Any) -> str:
        if isinstance(category, Enum):
            return category.name
        c = str(category)
        # Handle formats like "ClaimCategoryType.VISUAL_GROUNDING_REQUIRED"
        if '.' in c:
            c = c.split('.')[-1]
        return c.strip().upper()

    def _parse_questions_list(self, text: str) -> List[str]:
        lines = [re.sub(r"^\s*[-*\d\.)\]]\s*", "", ln).strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln]

    def _generate_primary_questions(self, claim: str, category: str, num_questions: int) -> List[str]:
        try:
            tmpl = self.category_prompts.get(category)
            if not tmpl:
                raise LLMGenerationError(f"No prompt template for category {category}")
            prompt = tmpl.format(claim=claim, num_questions=num_questions)
            raw = self.llm.generate(prompt, temperature=0.2, max_tokens=600)
            return self._parse_questions_list(raw)
        except Exception as e:
            logger.error(f"LLM generation failed for category {category}: {e}")
            raise LLMGenerationError(str(e))

    def _generate_fallback_questions(self, claim: str, category: str, needed: int) -> List[SocraticQuestion]:
        fallback_templates = {
            'VISUAL_GROUNDING_REQUIRED': [
                f"What visual elements in the image support or contradict: '{claim}'?",
                f"Is the claim '{claim}' visually verifiable in the provided image?",
            ],
            'EXTERNAL_KNOWLEDGE_REQUIRED': [
                f"Is the statement '{claim}' factually accurate according to reliable sources?",
                f"What evidence exists to support or refute: '{claim}'?",
            ],
            'AMBIGUOUS_RESOLUTION_REQUIRED': [
                f"Which terms in '{claim}' need clarification for accurate verification?",
                f"What additional context is needed to interpret '{claim}' precisely?",
            ],
            'SELF_CONSISTENCY_REQUIRED': [
                f"Does '{claim}' contradict any previously established facts?",
                f"Is '{claim}' consistent with prior knowledge about the entities mentioned?",
            ],
        }
        templates = fallback_templates.get(category, [f"How can '{claim}' be verified?"])
        questions: List[SocraticQuestion] = []
        for i in range(needed):
            t = templates[i % len(templates)]
            questions.append(
                SocraticQuestion(
                    question=t,
                    category=category,
                    verification_hint=QuestionValidator()._generate_verification_hint(category),
                    confidence_score=0.6,
                    fallback=True,
                )
            )
        return questions

    def _apply_prioritization(self, results: Dict[str, List[SocraticQuestion]], prioritize_category: str) -> Dict[str, List[SocraticQuestion]]:
        ordered = OrderedDict()
        if prioritize_category in results:
            ordered[prioritize_category] = results[prioritize_category]
        for k, v in results.items():
            if k == prioritize_category:
                continue
            ordered[k] = v
        return dict(ordered)

    def _validate_questions(self, questions: List[str], claim: str, category: str) -> List[SocraticQuestion]:
        return self.question_validator.validate_questions(questions, claim, category)

    def generate_questions(self, claim: str, categories: List[Any], num_questions: int = 1, prioritize_category: Optional[Any] = None) -> Dict[str, List[SocraticQuestion]]:
        results: Dict[str, List[SocraticQuestion]] = {}
        norm_categories = [self._normalize_category(c) for c in categories]
        priority = self._normalize_category(prioritize_category) if prioritize_category else None

        for category in norm_categories:
            try:
                primary_qs = self._generate_primary_questions(claim, category, num_questions)
                valid_qs = self._validate_questions(primary_qs, claim, category)

                # Attempt retries for remaining needed questions before fallback
                retries_left = getattr(self.config, 'primary_retries', 0)
                needed = max(0, num_questions - len(valid_qs))
                temp_base = 0.2
                while needed > 0 and retries_left > 0:
                    try:
                        # Nudge temperature up to diversify outputs
                        temp = temp_base + (self.config.retry_temperature_increment * (self.config.primary_retries - retries_left + 1))
                        tmpl = self.category_prompts.get(category)
                        if not tmpl:
                            break
                        prompt = tmpl.format(claim=claim, num_questions=needed)
                        raw = self.llm.generate(prompt, temperature=temp, max_tokens=600)
                        more_primary = self._parse_questions_list(raw)
                        more_valid = self._validate_questions(more_primary, claim, category)
                        # De-duplicate by question text
                        existing = {q.question for q in valid_qs}
                        for q in more_valid:
                            if q.question not in existing and len(valid_qs) < num_questions:
                                valid_qs.append(q)
                                existing.add(q.question)
                        needed = max(0, num_questions - len(valid_qs))
                    except Exception:
                        # If retry fails, proceed to next retry or fallback
                        pass
                    finally:
                        retries_left -= 1

                if len(valid_qs) < num_questions and self.config.enable_fallback:
                    needed = num_questions - len(valid_qs)
                    valid_qs.extend(self._generate_fallback_questions(claim, category, needed))
                results[category] = valid_qs[:num_questions]
            except LLMGenerationError:
                if self.config.enable_fallback:
                    results[category] = self._generate_fallback_questions(claim, category, num_questions)
                else:
                    raise

        if priority and priority in results:
            results = self._apply_prioritization(results, priority)
        return results

    def handle_multi_category_claims(self, claim: str, categories: List[Any], num_questions_per_category: int = 1) -> Dict[str, List[SocraticQuestion]]:
        norm_categories = [self._normalize_category(c) for c in categories]
        # Resolve ambiguity first
        if 'AMBIGUOUS_RESOLUTION_REQUIRED' in norm_categories:
            claim = self._resolve_ambiguity(claim)
            norm_categories = [c for c in norm_categories if c != 'AMBIGUOUS_RESOLUTION_REQUIRED']
        results: Dict[str, List[SocraticQuestion]] = {}
        for cat in norm_categories:
            res = self.generate_questions(claim, [cat], num_questions=num_questions_per_category)
            results.update(res)
        return results

    def _resolve_ambiguity(self, claim: str) -> str:
        ambiguous_terms = self.ambiguity_detector.detect_ambiguous_terms(claim)
        if not ambiguous_terms:
            return claim
        # In absence of interactive UI, choose first interpretation heuristically
        selections = self._get_user_interpretations(ambiguous_terms)
        disamb_claim = self._reconstruct_claim(claim, selections)
        try:
            ok = self._verify_user_input(f"Disambiguation applied: {selections}")
            if not ok:
                raise AmbiguityResolutionError("User disambiguation failed verification")
        except Exception:
            # fallback to original claim
            return claim
        return disamb_claim

    def _get_user_interpretations(self, ambiguous_terms: Dict[str, List[str]]) -> Dict[str, str]:
        # Placeholder: choose first interpretation per term
        return {term: choices[0] for term, choices in ambiguous_terms.items() if choices}

    def _reconstruct_claim(self, claim: str, interpretations: Dict[str, str]) -> str:
        new_claim = claim
        for term, interp in interpretations.items():
            # Simple reconstruction: append clarifying parenthetical the first time the term appears
            pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
            if pattern.search(new_claim):
                new_claim = pattern.sub(f"{term} ({interp})", new_claim, count=1)
        return new_claim

    def _verify_user_input(self, user_claim: str) -> bool:
        # Stub: return True; in future, route through verification pipeline
        return True

