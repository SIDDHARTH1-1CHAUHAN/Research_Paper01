"""
Multilingual Configuration Loader
Loads language-specific configuration files for keywords, prompts, stopwords, and credibility
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from loguru import logger
import yaml


class MultilingualConfig:
    """
    Loader for multilingual configuration files.

    Loads and caches:
    - Language settings (supported_languages.yaml)
    - Keywords for each language (keywords/{lang}.yaml)
    - Prompts for each language (prompts/{lang}.yaml)
    - Stopwords for each language (stopwords/{lang}.txt)
    - Credibility sources for each language (credibility/{lang}.yaml)
    """

    def __init__(self, config_base_path: Optional[str] = None):
        """
        Initialize multilingual configuration loader.

        Args:
            config_base_path: Base path to config/languages directory
        """
        self.config_base = self._find_config_path(config_base_path)
        self._cache: Dict[str, Any] = {}
        self._languages_config: Optional[Dict] = None

    def _find_config_path(self, config_path: Optional[str]) -> Path:
        """Find the languages configuration directory"""
        if config_path:
            return Path(config_path)

        possible_paths = [
            Path("config/languages"),
            Path("../config/languages"),
            Path("../../config/languages"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        logger.warning("Languages config directory not found, using default path")
        return Path("config/languages")

    def get_supported_languages(self) -> Dict[str, Any]:
        """
        Get supported languages configuration.

        Returns:
            Dictionary with language settings
        """
        if self._languages_config is None:
            config_file = self.config_base / "supported_languages.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._languages_config = yaml.safe_load(f)
            else:
                logger.warning("supported_languages.yaml not found, using defaults")
                self._languages_config = self._default_languages_config()

        return self._languages_config

    def _default_languages_config(self) -> Dict[str, Any]:
        """Return default language configuration"""
        return {
            'languages': [
                {'code': 'en', 'name': 'English', 'native_name': 'English',
                 'search_region': 'wt-wt', 'primary': True},
                {'code': 'es', 'name': 'Spanish', 'native_name': 'Español',
                 'search_region': 'es-es', 'primary': False},
            ],
            'default_language': 'en',
            'detection': {
                'min_confidence': 0.5,
                'fallback_to_english': True
            }
        }

    def get_language_info(self, lang_code: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific language"""
        config = self.get_supported_languages()
        for lang in config.get('languages', []):
            if lang['code'] == lang_code:
                return lang
        return None

    def is_supported(self, lang_code: str) -> bool:
        """Check if a language is supported"""
        return self.get_language_info(lang_code) is not None

    def get_search_region(self, lang_code: str) -> str:
        """Get DuckDuckGo search region for a language"""
        lang_info = self.get_language_info(lang_code)
        if lang_info:
            return lang_info.get('search_region', 'wt-wt')
        return 'wt-wt'

    def get_accept_language(self, lang_code: str) -> str:
        """Get Accept-Language header value for a language"""
        lang_info = self.get_language_info(lang_code)
        if lang_info:
            return lang_info.get('accept_language', 'en-US,en;q=0.9')
        return 'en-US,en;q=0.9'

    def get_keywords(self, lang_code: str) -> Dict[str, Any]:
        """
        Get keywords configuration for a language.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Dictionary with opinion_words, future_words, vague_words, etc.
        """
        cache_key = f"keywords_{lang_code}"

        if cache_key not in self._cache:
            keywords_file = self.config_base / "keywords" / f"{lang_code}.yaml"

            if keywords_file.exists():
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    self._cache[cache_key] = yaml.safe_load(f)
            else:
                # Fallback to English
                logger.warning(f"Keywords file not found for {lang_code}, using English")
                return self.get_keywords('en')

        return self._cache.get(cache_key, {})

    def get_opinion_words(self, lang_code: str) -> List[str]:
        """Get opinion indicator words for a language"""
        keywords = self.get_keywords(lang_code)
        return keywords.get('opinion_words', [])

    def get_future_words(self, lang_code: str) -> List[str]:
        """Get future prediction indicator words for a language"""
        keywords = self.get_keywords(lang_code)
        return keywords.get('future_words', [])

    def get_vague_words(self, lang_code: str) -> List[str]:
        """Get vague claim indicator words for a language"""
        keywords = self.get_keywords(lang_code)
        return keywords.get('vague_words', [])

    def get_prompts(self, lang_code: str) -> Dict[str, Any]:
        """
        Get prompts configuration for a language.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Dictionary with prompt templates
        """
        cache_key = f"prompts_{lang_code}"

        if cache_key not in self._cache:
            prompts_file = self.config_base / "prompts" / f"{lang_code}.yaml"

            if prompts_file.exists():
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    self._cache[cache_key] = yaml.safe_load(f)
            else:
                # Fallback to English
                if lang_code != 'en':
                    logger.warning(f"Prompts file not found for {lang_code}, using English")
                    return self.get_prompts('en')
                self._cache[cache_key] = {}

        return self._cache.get(cache_key, {})

    def get_verdict_template(self, lang_code: str, verdict_type: str) -> str:
        """Get verdict explanation template for a language"""
        prompts = self.get_prompts(lang_code)
        templates = prompts.get('verdict_explanation', {})
        return templates.get(verdict_type, '')

    def get_stopwords(self, lang_code: str) -> Set[str]:
        """
        Get stopwords for a language.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Set of stopwords
        """
        cache_key = f"stopwords_{lang_code}"

        if cache_key not in self._cache:
            stopwords_file = self.config_base / "stopwords" / f"{lang_code}.txt"

            if stopwords_file.exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                    self._cache[cache_key] = set(words)
            else:
                # Fallback to English
                if lang_code != 'en':
                    logger.warning(f"Stopwords file not found for {lang_code}, using English")
                    return self.get_stopwords('en')
                self._cache[cache_key] = set()

        return self._cache.get(cache_key, set())

    def get_credibility_config(self, lang_code: str) -> Dict[str, Any]:
        """
        Get credibility sources configuration for a language.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Dictionary with high_credibility, medium_credibility, fact_checkers
        """
        cache_key = f"credibility_{lang_code}"

        if cache_key not in self._cache:
            cred_file = self.config_base / "credibility" / f"{lang_code}.yaml"

            if cred_file.exists():
                with open(cred_file, 'r', encoding='utf-8') as f:
                    self._cache[cache_key] = yaml.safe_load(f)
            else:
                # Fallback to English
                if lang_code != 'en':
                    logger.debug(f"Credibility config not found for {lang_code}, using English")
                    return self.get_credibility_config('en')
                self._cache[cache_key] = {}

        return self._cache.get(cache_key, {})

    def get_high_credibility_domains(self, lang_code: str) -> List[Dict[str, Any]]:
        """Get high credibility domains for a language"""
        config = self.get_credibility_config(lang_code)
        return config.get('high_credibility', [])

    def get_fact_checkers(self, lang_code: str) -> List[Dict[str, Any]]:
        """Get fact-checking organizations for a language"""
        config = self.get_credibility_config(lang_code)
        return config.get('fact_checkers', [])

    def clear_cache(self):
        """Clear all cached configurations"""
        self._cache.clear()
        self._languages_config = None


# Singleton instance
_config_instance: Optional[MultilingualConfig] = None


def get_multilingual_config() -> MultilingualConfig:
    """Get or create multilingual configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = MultilingualConfig()
    return _config_instance
