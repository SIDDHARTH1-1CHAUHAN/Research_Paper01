"""
Language Detector - FastText-based language detection with code-switching support
Supports 176 languages at 120k sentences/sec for production-grade performance
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import re


@dataclass
class LanguageSegment:
    """Represents a text segment with its detected language"""
    text: str
    language: str
    confidence: float
    start_index: int
    end_index: int


@dataclass
class DetectionResult:
    """Result of language detection"""
    primary_language: str
    primary_confidence: float
    all_predictions: List[Tuple[str, float]]
    has_code_switching: bool
    segments: Optional[List[LanguageSegment]] = None


class LanguageDetector:
    """
    FastText-based language detection with code-switching support.

    Features:
    - Detects 176 languages with high accuracy
    - Identifies code-switching (mixed language text)
    - Segments text by language for mixed content
    - Falls back to heuristic detection if FastText unavailable
    """

    # ISO 639-1 to language name mapping for supported languages
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

    # Search regions for DuckDuckGo by language
    SEARCH_REGIONS = {
        'en': 'wt-wt',      # Worldwide
        'es': 'es-es',      # Spain
        'fr': 'fr-fr',      # France
        'de': 'de-de',      # Germany
        'zh': 'cn-zh',      # China
        'ar': 'xa-ar',      # Arabia
        'hi': 'in-en',      # India
        'ja': 'jp-jp',      # Japan
        'pt': 'br-pt',      # Brazil
        'ru': 'ru-ru',      # Russia
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize language detector.

        Args:
            model_path: Path to FastText language detection model.
                       If None, will attempt to download/use default.
        """
        self.model = None
        self.model_path = model_path
        self._setup_fasttext()

    def _setup_fasttext(self):
        """Initialize FastText model for language detection"""
        try:
            import fasttext

            # Try to load model from specified path or default locations
            model_paths = [
                self.model_path,
                Path("models/lid.176.bin"),
                Path("models/lid.176.ftz"),
                Path.home() / ".fasttext" / "lid.176.bin",
            ]

            for path in model_paths:
                if path and Path(path).exists():
                    # Suppress FastText warnings
                    fasttext.FastText.eprint = lambda x: None
                    self.model = fasttext.load_model(str(path))
                    logger.info(f"FastText model loaded from {path}")
                    return

            # If no model found, try to use compressed version
            logger.warning("FastText model not found. Using heuristic fallback.")
            logger.info("To enable FastText: download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html")

        except ImportError:
            logger.warning("FastText not installed. Using heuristic language detection.")
            logger.info("Install with: pip install fasttext")
        except Exception as e:
            logger.error(f"FastText initialization failed: {e}")

    def detect(self, text: str, k: int = 1) -> DetectionResult:
        """
        Detect the primary language of text.

        Args:
            text: Text to analyze
            k: Number of top predictions to return

        Returns:
            DetectionResult with primary language and confidence
        """
        if not text or not text.strip():
            return DetectionResult(
                primary_language='en',
                primary_confidence=0.0,
                all_predictions=[('en', 0.0)],
                has_code_switching=False
            )

        # Clean text for detection
        clean_text = self._preprocess_text(text)

        if self.model:
            return self._detect_fasttext(clean_text, k)
        else:
            return self._detect_heuristic(clean_text, k)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for language detection"""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove numbers (keep text)
        text = re.sub(r'\d+', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    def _detect_fasttext(self, text: str, k: int) -> DetectionResult:
        """Detect language using FastText"""
        # Get predictions
        predictions = self.model.predict(text, k=k)
        labels, scores = predictions

        # Parse results (FastText returns '__label__en' format)
        results = []
        for label, score in zip(labels, scores):
            lang_code = label.replace('__label__', '')
            results.append((lang_code, float(score)))

        primary_lang, primary_conf = results[0]

        # Check for code-switching
        has_switching = self._check_code_switching(text)

        return DetectionResult(
            primary_language=primary_lang,
            primary_confidence=primary_conf,
            all_predictions=results,
            has_code_switching=has_switching
        )

    def _detect_heuristic(self, text: str, k: int) -> DetectionResult:
        """Heuristic-based language detection (fallback)"""
        scores = {}

        # Character-based detection
        for lang, patterns in self._get_language_patterns().items():
            score = 0
            for pattern, weight in patterns:
                if re.search(pattern, text):
                    score += weight
            scores[lang] = score

        # Default to English if no patterns match
        if not scores or all(s == 0 for s in scores.values()):
            scores['en'] = 0.5

        # Normalize and sort
        total = sum(scores.values()) or 1
        sorted_scores = sorted(
            [(lang, score/total) for lang, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]

        primary_lang, primary_conf = sorted_scores[0]

        return DetectionResult(
            primary_language=primary_lang,
            primary_confidence=primary_conf,
            all_predictions=sorted_scores,
            has_code_switching=False
        )

    def _get_language_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get regex patterns for heuristic language detection"""
        return {
            'zh': [
                (r'[\u4e00-\u9fff]', 1.0),  # Chinese characters
            ],
            'ja': [
                (r'[\u3040-\u309f]', 1.0),  # Hiragana
                (r'[\u30a0-\u30ff]', 1.0),  # Katakana
            ],
            'ko': [
                (r'[\uac00-\ud7af]', 1.0),  # Hangul
            ],
            'ar': [
                (r'[\u0600-\u06ff]', 1.0),  # Arabic
            ],
            'hi': [
                (r'[\u0900-\u097f]', 1.0),  # Devanagari
            ],
            'ru': [
                (r'[\u0400-\u04ff]', 1.0),  # Cyrillic
            ],
            'th': [
                (r'[\u0e00-\u0e7f]', 1.0),  # Thai
            ],
            'es': [
                (r'[ñáéíóúü¿¡]', 0.3),  # Spanish special chars
                (r'\b(el|la|los|las|de|en|que|es|por|un|una)\b', 0.2),
            ],
            'fr': [
                (r'[àâçéèêëîïôùûü]', 0.3),  # French accents
                (r'\b(le|la|les|de|des|en|est|que|un|une|et)\b', 0.2),
            ],
            'de': [
                (r'[äöüß]', 0.3),  # German special chars
                (r'\b(der|die|das|und|ist|von|den|dem|ein|eine)\b', 0.2),
            ],
            'pt': [
                (r'[ãõáéíóúâêôç]', 0.3),  # Portuguese special chars
                (r'\b(o|a|os|as|de|da|do|em|que|um|uma|e)\b', 0.2),
            ],
            'en': [
                (r'\b(the|is|are|was|were|be|been|have|has|had|do|does|did)\b', 0.1),
                (r'\b(and|or|but|in|on|at|to|for|of|with|by)\b', 0.1),
            ],
        }

    def _check_code_switching(self, text: str) -> bool:
        """Check if text contains multiple languages (code-switching)"""
        # Split into sentences/segments
        segments = re.split(r'[.!?。！？]', text)
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) < 2:
            return False

        # Detect language for each segment
        languages = set()
        for segment in segments[:5]:  # Check first 5 segments
            if len(segment) > 10:
                result = self.detect(segment)
                if result.primary_confidence > 0.5:
                    languages.add(result.primary_language)

        return len(languages) > 1

    def has_code_switching(self, text: str) -> bool:
        """
        Check if text contains code-switching (multiple languages).

        Args:
            text: Text to analyze

        Returns:
            True if text contains multiple languages
        """
        return self._check_code_switching(text)

    def segment_by_language(self, text: str) -> List[LanguageSegment]:
        """
        Segment text by language for code-switched content.

        Args:
            text: Text to segment

        Returns:
            List of LanguageSegment objects
        """
        segments = []

        # Split by sentence boundaries
        sentence_pattern = r'([.!?。！？]+)'
        parts = re.split(sentence_pattern, text)

        current_pos = 0
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            delimiter = parts[i + 1] if i + 1 < len(parts) else ''
            full_segment = sentence + delimiter

            if full_segment.strip():
                result = self.detect(sentence)
                segments.append(LanguageSegment(
                    text=full_segment,
                    language=result.primary_language,
                    confidence=result.primary_confidence,
                    start_index=current_pos,
                    end_index=current_pos + len(full_segment)
                ))

            current_pos += len(full_segment)

        return segments

    def detect_multiple(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k language predictions with confidence scores.

        Args:
            text: Text to analyze
            k: Number of predictions to return

        Returns:
            List of (language_code, confidence) tuples
        """
        result = self.detect(text, k=k)
        return result.all_predictions

    def get_language_name(self, code: str) -> str:
        """Get human-readable language name from ISO 639-1 code"""
        return self.LANGUAGE_NAMES.get(code, code.upper())

    def get_search_region(self, code: str) -> str:
        """Get DuckDuckGo search region for language code"""
        return self.SEARCH_REGIONS.get(code, 'wt-wt')

    def is_supported(self, code: str) -> bool:
        """Check if a language code is in the primary supported set"""
        return code in self.LANGUAGE_NAMES


# Convenience function
def detect_language(text: str) -> Tuple[str, float]:
    """
    Quick language detection for a text string.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (language_code, confidence)
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result.primary_language, result.primary_confidence
