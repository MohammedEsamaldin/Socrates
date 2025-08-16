"""
Claim Extraction Module

This module is responsible for extracting atomic, verifiable claims from input text
using a combination of prompt engineering and NLP techniques. It implements the first
stage of the Socrates Agent pipeline, focusing on identifying check-worthy factual assertions.
"""
import json
import re
import logging
import demjson3
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from sentence_transformers import SentenceTransformer, util
from socrates_system.modules.shared_structures import ExtractedClaim, ExtractedEntity, ExtractedRelationship, ClaimCategory, VerificationRoute
import spacy
from spacy.tokens import Span
from pathlib import Path

from ..utils.logger import setup_logger
from socrates_system.config import ENTITY_MODEL_NAME, NLP_MODEL_NAME, SIMILARITY_THRESHOLD
from socrates_system.modules.llm_manager import LLMManager  # Assuming LLMManager is available for LLM interactions

logger = setup_logger(__name__)

# Load prompt templates
PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"
CLAIM_EXTRACTION_PROMPT = (PROMPT_TEMPLATES_DIR / "claim_extraction.txt").read_text(encoding="utf-8")


class ClaimExtractor:
    """
    Advanced claim extraction using prompt engineering and NLP techniques.

    This class implements the first stage of the Socrates Agent pipeline, focusing on
    identifying atomic, check-worthy factual assertions from input text while filtering out
    opinions, rhetorical questions, and non-verifiable content.
    """

    def __init__(self, llm_manager: Optional[Any] = None):
        """Initialize the Claim Extractor with optional LLM manager.

        Args:
            llm_manager: An instance of LLMManager for LLM interactions. If None,
                        a rule-based fallback will be used.
        """
        logger.info("Initializing Claim Extractor...")

        try:
            # Load spaCy model for entity recognition and sentence splitting
            self.nlp = spacy.load(ENTITY_MODEL_NAME)

            # Initialize LLM manager if provided
            self.llm_manager = llm_manager

            # Fallback patterns for when LLM is not available
            self.fallback_patterns = [
                # Attribute patterns
                r'([A-Z][^.!?]*\b(is|are|was|were|has|have|contains|includes)\b[^.!?]+[.!?])',
                # Relationship patterns
                r'([A-Z][^.!?]*(\b(?:is located in|belongs to|is part of|is connected to|is related to|causes)\b)[^.!?]+[.!?])',
                # Temporal patterns
                r'([A-Z][^.!?]*(\b(?:happened in|was built in|occurred on|started in|ended in|in the year|on the)\b[^.!?]+[.!?]))',
                # Comparative patterns
                r'([A-Z][^.!?]*(\b(?:than|more than|less than|bigger than|smaller than|faster than|slower than)\b)[^.!?]+[.!?])',
                # Definition patterns
                r'([A-Z][^.!?]*(\b(?:is defined as|means|refers to|is a type of)\b)[^.!?]+[.!?])'
            ]

            # Compile regex patterns for fallback extraction
            self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.fallback_patterns]

            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.similarity_threshold = SIMILARITY_THRESHOLD

            logger.info("Claim Extractor initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Claim Extractor: {str(e)}")
            raise

    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract atomic, verifiable claims from input text.

        This is the main entry point for claim extraction. It processes the input text,
        extracts claims using either LLM-based or rule-based methods, and returns
        a list of atomic, verifiable claims.

        Args:
            text: Input text to analyze

        Returns:
            List of ExtractedClaim objects representing atomic, verifiable claims
        """
        logger.info(f"Extracting claims from text: {text[:100]}...")

        if not text or not text.strip():
            logger.warning("Empty input text provided")
            return []

        try:
            # Process text with spaCy for sentence splitting and entity recognition
            doc = self.nlp(text)

            # Use LLM-based extraction if available, otherwise fall back to rule-based
            if self.llm_manager:
                claims = self._extract_claims_with_llm(text, doc)
            else:
                logger.info("LLM not available, using rule-based extraction")
                claims = self._extract_claims_with_rules(text, doc)

            # Post-process claims to ensure atomicity and remove duplicates
            claims = self._post_process_claims(claims, text)

            logger.info(f"Successfully extracted {len(claims)} claims")
            return claims

        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}", exc_info=True)
            return []

    def _extract_claims_with_llm(self, text: str, doc) -> List[ExtractedClaim]:
        """
        Extract claims using LLM-based approach with prompt engineering.

        This method uses the LLM to extract atomic, verifiable claims from the input text
        based on the provided prompt template.

        Args:
            text: Input text to analyze
            doc: spaCy Doc object for the input text

        Returns:
            List of extracted claims
        """
        try:
            # Create a JSON example for the prompt
            json_example = {
                "claims": [
                    {
                        "claim_text": "The Apollo 11 mission was the first to land humans on the Moon.",
                        "confidence": 0.98,
                        "entities": [
                            {"text": "Apollo 11", "label": "EVENT"},
                            {"text": "the Moon", "label": "LOC"}
                        ]
                    },
                    {
                        "claim_text": "The crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins.",
                        "confidence": 0.95,
                        "entities": [
                            {"text": "Neil Armstrong", "label": "PERSON"},
                            {"text": "Buzz Aldrin", "label": "PERSON"},
                            {"text": "Michael Collins", "label": "PERSON"}
                        ]
                    }
                ]
            }
            # Format JSON example safely to avoid KeyError with curly braces
            json_example_str = json.dumps(json_example, indent=2)
            # Format the prompt, injecting both the input text and the JSON example
            prompt = CLAIM_EXTRACTION_PROMPT.format(
                input_text=text, 
                json_example=json_example_str
            )
            llm_response_str = self.llm_manager.generate_text(prompt, max_tokens=8192)
            logger.info(f"RAW LLM RESPONSE: '{llm_response_str}'")
            claims = self._parse_llm_response(llm_response_str, doc)
            logger.info(f"Successfully extracted {len(claims)} claims using LLM.")
            return claims

        except Exception as e:
            logger.error(f"Error in LLM-based claim extraction: {e}, falling back to rule-based extraction.", exc_info=True)
            return self._extract_claims_with_rules(text, doc)

    def _extract_claims_with_rules(self, text: str, doc: spacy.tokens.Doc) -> List[ExtractedClaim]:
        """
        Fallback method to extract claims using rule-based patterns.

        This is used when LLM is not available or fails. It's less accurate but
        more reliable than LLM-based extraction.

        Args:
            text: Input text to analyze
            doc: spaCy Doc object for the input text

        Returns:
            List of extracted claims
        """
        claims = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 4:
                continue

            # Check against fallback patterns
            for pattern in self.compiled_patterns:
                if pattern.search(sent_text):
                    # If a pattern matches, treat the whole sentence as a potential claim
                    confidence = self._score_confidence(sent)
                    if confidence > 0.5:
                        entities = [
                            ExtractedEntity(
                                text=ent.text,
                                label=ent.label_,
                                start_char=ent.start_char,
                                end_char=ent.end_char
                            )
                            for ent in sent.ents
                        ]
                        context_window = self._get_context_window(doc, sent.start_char, sent.end_char)
                        claim = ExtractedClaim(
                            text=self._normalize_claim_text(sent.text.strip()),
                            start_char=sent.start_char,
                            end_char=sent.end_char,
                            confidence=confidence,
                            source_text=text,
                            entities=entities,
                            context_window=context_window
                        )
                        claims.append(claim)
                        # Once a claim is created from a sentence, move to the next sentence
                        break
        return claims

    def _parse_llm_response(self, llm_response_str: str, doc: spacy.tokens.Doc) -> List[ExtractedClaim]:
        """Parse the JSON response from the LLM into a list of ExtractedClaim objects."""
        try:
            response = llm_response_str

            # If no markdown, assume the whole response is the JSON content.
            # Find the first '{' and the last '}' to extract the JSON object.
            start = response.find('{')
            if start != -1:
                # Find the matching closing brace for the object.
                open_braces = 0
                end = -1
                for i, char in enumerate(response[start:]):
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                    if open_braces == 0:
                        end = start + i
                        break
                if end != -1:
                    json_str = response[start:end+1]
                else:
                    # Fallback for unterminated JSON, add the closing brace
                    json_str = response[start:] + '}' 
            else:
                json_str = response

            # Sanitize common LLM JSON mistakes (unquoted identifiers)
            # Quote common confidence tokens
            json_sanitized = re.sub(r'("confidence"\s*:\s*)(Low|Medium|High)(\s*[,}])', r'\1"\2"\3', json_str)
            # Quote common enum-like fields (e.g., type_hint)
            json_sanitized = re.sub(r'("type_hint"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[,}])', r'\1"\2"\3', json_sanitized)
            # First parse attempt
            try:
                data = demjson3.decode(json_sanitized)
            except (demjson3.JSONDecodeError, Exception):
                # General fallback: quote bare word identifiers after ':' unless true/false/null/number
                def _quote_bare_ident(m):
                    leading, ident, trailing = m.group(1), m.group(2), m.group(3)
                    low = ident.lower()
                    if low in {"true", "false", "null"}:
                        return f"{leading}{ident}{trailing}"
                    # Leave numbers alone
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", ident):
                        return f"{leading}{ident}{trailing}"
                    return f"{leading}\"{ident}\"{trailing}"
                generic_sanitized = re.sub(r'(:\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[,}])', _quote_bare_ident, json_sanitized)
                data = demjson3.decode(generic_sanitized)
            claim_data_list = data.get('claims', [])
            if not isinstance(claim_data_list, list):
                logger.error(f"LLM 'claims' field is not a list: {claim_data_list}")
                return []

        except (demjson3.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(f"Failed to decode or parse LLM JSON response: {e}\nResponse: '{llm_response_str}'")
            return []

        if not claim_data_list:
            logger.warning("LLM response contained an empty 'claims' list.")
            return []

        # Optimized Clause-Based Matching
        source_sents = list(doc.sents)
        source_units, sent_map = [], []
        for i, sent in enumerate(source_sents):
            clauses = [sent.text] + [conj.text for conj in sent.conjuncts]
            clauses.extend([chunk.text for chunk in sent.noun_chunks if len(chunk.text.split()) > 3])
            unique_clauses = list(dict.fromkeys(clauses))
            source_units.extend(unique_clauses)
            sent_map.extend([i] * len(unique_clauses))

        if not source_units:
            return []

        llm_claims_text = [self._normalize_claim_text(c.get('claim_text', '')) for c in claim_data_list]
        
        llm_claim_embeddings = self.similarity_model.encode(llm_claims_text, convert_to_tensor=True)
        source_unit_embeddings = self.similarity_model.encode(source_units, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(llm_claim_embeddings, source_unit_embeddings)

        extracted_claims = []
        for i, llm_claim in enumerate(claim_data_list):
            if not llm_claims_text[i]: continue # Skip if claim text is empty

            best_match_unit_idx = cosine_scores[i].argmax().item()
            best_score = cosine_scores[i][best_match_unit_idx].item()

            if best_score >= self.similarity_threshold:
                original_sent_idx = sent_map[best_match_unit_idx]
                best_match_sent = source_sents[original_sent_idx]
                claim_text = llm_claims_text[i]
                logger.info(f"Matched LLM claim (score: {best_score:.2f}): '{claim_text}' -> '{best_match_sent.text}'")

                entities = []
                for entity in llm_claim.get('entities', []):
                    entity_text = entity.get('text')
                    if not entity_text:
                        continue
                    try:
                        start_char_in_sent = best_match_sent.text.index(entity_text)
                        start_char = start_char_in_sent + best_match_sent.start_char
                        end_char = start_char + len(entity_text)
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            label=entity.get('label', 'UNKNOWN'),
                            start_char=start_char,
                            end_char=end_char
                        ))
                    except ValueError:
                        logger.warning(f"LLM-extracted entity '{{{entity_text}}}' not found verbatim in source sentence: '{{{best_match_sent.text}}}'. Skipping entity.")

                claim = ExtractedClaim(
                    text=claim_text,
                    start_char=best_match_sent.start_char,
                    end_char=best_match_sent.end_char,
                    confidence=(
                        (lambda raw: (
                            0.2 if isinstance(raw, str) and raw.lower() == 'low' else
                            0.5 if isinstance(raw, str) and raw.lower() == 'medium' else
                            0.8 if isinstance(raw, str) and raw.lower() == 'high' else
                            float(raw) if isinstance(raw, (int, float)) else
                            best_score
                        ))(llm_claim.get('confidence', best_score))
                    ),
                    source_text=doc.text,
                    entities=entities,
                    context_window=self._get_context_window(doc, best_match_sent.start_char, best_match_sent.end_char),
                    ambiguity_reason=llm_claim.get('ambiguity_reason')
                )
                extracted_claims.append(claim)
            else:
                logger.warning(f"Skipping LLM claim with low similarity score ({best_score:.2f}): '{llm_claims_text[i]}'")
        
        return extracted_claims

    def _post_process_claims(self, claims: List[ExtractedClaim], source_text: str) -> List[ExtractedClaim]:
        """Post-process extracted claims to ensure quality and remove duplicates."""
        # Placeholder for future logic like deduplication or merging.
        return claims

    def _normalize_claim_text(self, text: str) -> str:
        """Normalize claim text for consistent processing."""
        text = ' '.join(text.split())
        text = text.replace('"', "'")
        if text.endswith(('.', '!', '?')) and not any(c.isupper() for c in text[-3:]):
            text = text[:-1].strip()
        return text

    def _get_context_window(self, doc: spacy.tokens.Doc, start_char: int, end_char: int, window_size: int = 2) -> str:
        """Get a context window of sentences around a given character span."""
        all_sents = list(doc.sents)
        if not all_sents:
            return ""

        claim_sent_indices = []
        for i, sent in enumerate(all_sents):
            if sent.start_char < end_char and sent.end_char > start_char:
                claim_sent_indices.append(i)

        if not claim_sent_indices:
            return ""

        start_index = max(0, claim_sent_indices[0] - window_size)
        end_index = min(len(all_sents), claim_sent_indices[-1] + window_size + 1)

        context_sents = [s.text for s in all_sents[start_index:end_index]]
        return " ".join(context_sents).strip()

    def _score_confidence(self, sent: Span) -> float:
        """More sophisticated confidence scoring based on linguistic features."""
        base_score = 0.6
        if any(tok.dep_ == 'nsubj' for tok in sent) and any(tok.dep_ in ('dobj', 'pobj', 'attr') for tok in sent):
            base_score += 0.15
        if any(ent.label_ == 'CARDINAL' for ent in sent.ents):
            base_score += 0.1
        if len(sent) < 5 or len(sent) > 40:
            base_score -= 0.1
        return min(1.0, max(0.0, base_score))

    def extract_claims_batch(self, texts: List[str]) -> List[List[ExtractedClaim]]:
        """Process a batch of texts for claim extraction."""
        all_claims = []
        for text in texts:
            all_claims.append(self.extract_claims(text))
        return all_claims

    def to_json(self, claims: List[ExtractedClaim]) -> str:
        """Convert a list of claims to JSON format."""
        return json.dumps([claim.to_dict() for claim in claims], indent=2, ensure_ascii=False)

    def from_json(self, json_str: str) -> List[ExtractedClaim]:
        """Create ExtractedClaim objects from JSON string."""
        try:
            data = json.loads(json_str)
            return [ExtractedClaim.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse claims from JSON: {str(e)}")
            return []

    def get_claim_summary(self, claims: List[ExtractedClaim]) -> Dict[str, Any]:
        """
        Generate summary statistics for the extracted claims.
        
        Args:
            claims: List of extracted claims
            
        Returns:
            Dictionary with claim statistics
        """
        if not claims:
            return {"total_claims": 0}
        
        claim_types = {}
        total_entities = 0
        avg_confidence = 0.0
        
        for claim in claims:
            # Count claim types
            if hasattr(claim, 'claim_type'):
                claim_types[claim.claim_type] = claim_types.get(claim.claim_type, 0) + 1
            
            # Sum entities and confidence
            total_entities += len(claim.entities)
            avg_confidence += claim.confidence
        
        # Calculate averages
        avg_confidence = avg_confidence / len(claims) if claims else 0
        
        return {
            "total_claims": len(claims),
            "total_entities": total_entities,
            "avg_entities_per_claim": total_entities / len(claims) if claims else 0,
            "avg_confidence": avg_confidence,
            "claim_types": claim_types,
            "highest_confidence": max(claim.confidence for claim in claims) if claims else 0,
            "lowest_confidence": min(claim.confidence for claim in claims) if claims else 0
        }
