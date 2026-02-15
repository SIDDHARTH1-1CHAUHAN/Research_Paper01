"""
Translator - LLM-based translation with quality verification
Uses back-translation for quality assessment and supports batch operations
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import json


@dataclass
class TranslationResult:
    """Result of a translation operation"""
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    quality_score: float
    back_translation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Translator:
    """
    LLM-based translator with quality verification.

    Features:
    - Translate-then-verify approach using back-translation
    - Batch translation for efficiency
    - Quality scoring based on semantic preservation
    - Supports all languages handled by the LLM
    """

    # Language name mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ja': 'Japanese',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'it': 'Italian',
        'ko': 'Korean',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'id': 'Indonesian',
        'uk': 'Ukrainian',
        'cs': 'Czech',
    }

    def __init__(self, llm_interface=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize translator.

        Args:
            llm_interface: LLM interface for translation
            config: Configuration options
        """
        self.llm = llm_interface
        self.config = config or {}
        self.enable_quality_check = self.config.get('enable_quality_check', True)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)

    def set_llm(self, llm_interface):
        """Set the LLM interface (for deferred initialization)"""
        self.llm = llm_interface

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = 'en',
        verify: bool = False
    ) -> TranslationResult:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (default: 'en')
            verify: If True, perform back-translation quality check

        Returns:
            TranslationResult with translated text and quality metrics
        """
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                quality_score=1.0
            )

        # Skip if source and target are the same
        if source_lang == target_lang:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                quality_score=1.0
            )

        if self.llm:
            return self._translate_with_llm(text, source_lang, target_lang, verify)
        else:
            logger.warning("No LLM available for translation. Returning original text.")
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                quality_score=0.0,
                metadata={'error': 'No LLM available'}
            )

    def _translate_with_llm(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        verify: bool
    ) -> TranslationResult:
        """Translate using LLM"""
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        system_prompt = f"""You are a professional translator specializing in accurate translations.
Translate the given text from {source_name} to {target_name}.
Preserve the meaning, tone, and context as accurately as possible.
For factual claims, ensure all facts, numbers, dates, and proper nouns are preserved exactly.
Respond with ONLY the translated text, nothing else."""

        user_prompt = f"""Translate the following text from {source_name} to {target_name}:

{text}

Translation:"""

        try:
            translated = self.llm.generate(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.1  # Low temperature for consistent translation
            )
            translated = translated.strip()

            # Quality verification via back-translation
            quality_score = 0.8  # Default score
            back_translation = None

            if verify and self.enable_quality_check:
                back_translation, quality_score = self._verify_translation(
                    text, translated, source_lang, target_lang
                )

            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                quality_score=quality_score,
                back_translation=back_translation,
                metadata={'method': 'llm'}
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                quality_score=0.0,
                metadata={'error': str(e)}
            )

    def _verify_translation(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> Tuple[str, float]:
        """
        Verify translation quality using back-translation.

        Returns:
            Tuple of (back_translated_text, quality_score)
        """
        try:
            # Back-translate to original language
            back_result = self.translate(
                translated,
                source_lang=target_lang,
                target_lang=source_lang,
                verify=False  # Avoid infinite recursion
            )
            back_translation = back_result.translated_text

            # Calculate similarity score
            quality_score = self._calculate_similarity(original, back_translation)

            return back_translation, quality_score

        except Exception as e:
            logger.warning(f"Back-translation verification failed: {e}")
            return None, 0.5

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Uses simple word overlap for basic similarity.
        In production, could use sentence embeddings.
        """
        # Normalize texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 1.0 if not words1 and not words2 else 0.0

        return intersection / union

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str = 'en'
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of TranslationResult objects
        """
        results = []

        for text in texts:
            result = self.translate(text, source_lang, target_lang)
            results.append(result)

        return results

    def back_translate_verify(
        self,
        text: str,
        source_lang: str,
        target_lang: str = 'en'
    ) -> Tuple[str, float]:
        """
        Translate text and verify quality with back-translation.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Tuple of (translated_text, quality_score)
        """
        result = self.translate(text, source_lang, target_lang, verify=True)
        return result.translated_text, result.quality_score

    def translate_explanation(
        self,
        explanation: str,
        target_lang: str,
        template_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Translate an explanation/verdict to target language.

        Args:
            explanation: Explanation text in English
            target_lang: Target language code
            template_vars: Optional template variables to preserve

        Returns:
            Translated explanation
        """
        if target_lang == 'en':
            return explanation

        # Translate the explanation
        result = self.translate(explanation, 'en', target_lang)

        return result.translated_text

    def get_language_name(self, code: str) -> str:
        """Get human-readable language name"""
        return self.LANGUAGE_NAMES.get(code, code.upper())


# Singleton instance for convenience
_translator_instance = None


def get_translator(llm_interface=None) -> Translator:
    """Get or create translator instance"""
    global _translator_instance

    if _translator_instance is None:
        _translator_instance = Translator(llm_interface)
    elif llm_interface and _translator_instance.llm is None:
        _translator_instance.set_llm(llm_interface)

    return _translator_instance
