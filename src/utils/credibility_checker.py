"""
Credibility Checker - Assesses source reliability using heuristics
Fallback to MBFC API if configured
Supports multilingual operation with language-specific trusted sources
"""

from typing import Optional, Dict, Any, Set, List
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse
from loguru import logger
import requests

from src.utils.multilingual_config import get_multilingual_config


class CredibilityLevel(Enum):
    """Source credibility levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CredibilityScore:
    """Credibility assessment result"""
    level: CredibilityLevel
    score: float  # 0.0 - 1.0
    method: str  # "heuristic" or "mbfc_api"
    details: Dict[str, Any]


class CredibilityChecker:
    """
    Assesses source credibility using multiple methods:
    1. Heuristic-based (free, fast)
    2. MBFC API (paid, more accurate)

    Supports multilingual operation with language-specific trusted sources.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, language: str = 'en'):
        """
        Initialize credibility checker.

        Args:
            config: Credibility configuration from api_config.yaml
            language: Language code for regional credibility sources (default: 'en')
        """
        self.config = config or {}
        self.mbfc_api_key = self.config.get('mbfc', {}).get('api_key')
        self.language = language
        self.ml_config = get_multilingual_config()

        # Load language-specific credibility sources
        self._load_credibility_sources()

    def _load_credibility_sources(self):
        """Load language-specific credibility sources from configuration"""
        cred_config = self.ml_config.get_credibility_config(self.language)

        # Extract high credibility domains
        high_cred = cred_config.get('high_credibility', [])
        self.high_credibility_domains = set()
        self.high_credibility_scores = {}
        for item in high_cred:
            domain = item.get('domain', '')
            self.high_credibility_domains.add(domain)
            self.high_credibility_scores[domain] = item.get('score', 0.9)

        # Extract medium credibility domains
        med_cred = cred_config.get('medium_credibility', [])
        self.medium_credibility_domains = set()
        self.medium_credibility_scores = {}
        for item in med_cred:
            domain = item.get('domain', '')
            self.medium_credibility_domains.add(domain)
            self.medium_credibility_scores[domain] = item.get('score', 0.6)

        # Extract fact-checkers
        fact_checkers = cred_config.get('fact_checkers', [])
        self.fact_checker_domains = set()
        for item in fact_checkers:
            self.fact_checker_domains.add(item.get('domain', ''))

        # Add default domains if empty
        if not self.high_credibility_domains:
            self.high_credibility_domains = {
                'edu', 'gov', 'ac.uk', 'nature.com', 'science.org',
                'nih.gov', 'cdc.gov', 'who.int', 'nasa.gov',
                'reuters.com', 'apnews.com', 'bbc.com', 'pbs.org'
            }

        if not self.medium_credibility_domains:
            self.medium_credibility_domains = {
                'wikipedia.org', 'britannica.com', 'nationalgeographic.com',
                'scientificamerican.com', 'theconversation.com'
            }

    def set_language(self, language: str):
        """
        Set the language for credibility checking.

        Args:
            language: ISO 639-1 language code
        """
        if language != self.language:
            self.language = language
            self._load_credibility_sources()
            logger.debug(f"CredibilityChecker language set to: {language}")

    def check(self, url: str) -> CredibilityScore:
        """
        Assess credibility of a source URL.

        Args:
            url: Source URL to assess

        Returns:
            Credibility score with details
        """
        # Try MBFC API first if available
        if self.mbfc_api_key and self.config.get('fallback_method') == 'mbfc':
            try:
                return self._check_mbfc(url)
            except Exception as e:
                logger.warning(f"MBFC API check failed: {e}, falling back to heuristic")

        # Fallback to heuristic method
        return self._check_heuristic(url)

    def _check_heuristic(self, url: str) -> CredibilityScore:
        """
        Heuristic-based credibility assessment.

        Checks:
        - Domain suffix (.edu, .gov, .org)
        - Known high-quality sources
        - Domain characteristics
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www.
            domain = domain.replace('www.', '')

            details = {'domain': domain, 'checks': []}

            # Check 1: Educational or Government domains
            if domain.endswith('.edu') or domain.endswith('.gov') or domain.endswith('.ac.uk'):
                details['checks'].append('edu_gov_domain')
                return CredibilityScore(
                    level=CredibilityLevel.HIGH,
                    score=0.9,
                    method='heuristic',
                    details=details
                )

            # Check 2: Known high credibility sources
            for high_domain in self.high_credibility_domains:
                if domain.endswith(high_domain) or domain == high_domain:
                    details['checks'].append('known_high_credibility')
                    return CredibilityScore(
                        level=CredibilityLevel.HIGH,
                        score=0.85,
                        method='heuristic',
                        details=details
                    )

            # Check 3: Known medium credibility sources
            for med_domain in self.medium_credibility_domains:
                if domain.endswith(med_domain) or domain == med_domain:
                    details['checks'].append('known_medium_credibility')
                    return CredibilityScore(
                        level=CredibilityLevel.MEDIUM,
                        score=0.6,
                        method='heuristic',
                        details=details
                    )

            # Check 4: .org domains (generally trustworthy but varies)
            if domain.endswith('.org'):
                details['checks'].append('org_domain')
                return CredibilityScore(
                    level=CredibilityLevel.MEDIUM,
                    score=0.55,
                    method='heuristic',
                    details=details
                )

            # Check 5: HTTPS (basic security check)
            if parsed.scheme == 'https':
                details['checks'].append('https')
                score = 0.4
            else:
                score = 0.3

            # Default: Unknown .com or other domains
            details['checks'].append('unknown_domain')
            return CredibilityScore(
                level=CredibilityLevel.LOW,
                score=score,
                method='heuristic',
                details=details
            )

        except Exception as e:
            logger.error(f"Heuristic check failed for {url}: {e}")
            return CredibilityScore(
                level=CredibilityLevel.LOW,
                score=0.2,
                method='heuristic',
                details={'error': str(e)}
            )

    def _check_mbfc(self, url: str) -> CredibilityScore:
        """
        Check credibility using Media Bias/Fact Check API.

        Note: Requires MBFC API key (paid service)
        """
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')

        api_url = f"{self.config['mbfc']['endpoint']}/check"

        response = requests.get(
            api_url,
            params={'url': domain},
            headers={'Authorization': f'Bearer {self.mbfc_api_key}'},
            timeout=10
        )

        if response.status_code != 200:
            raise Exception(f"MBFC API error: {response.status_code}")

        data = response.json()

        # Map MBFC ratings to our levels
        factuality = data.get('factuality', '').lower()
        bias = data.get('bias', '').lower()

        allowed_factuality = self.config['mbfc'].get('allowed_factuality', [])
        allowed_bias = self.config['mbfc'].get('allowed_bias', [])

        if factuality in allowed_factuality and bias in allowed_bias:
            if factuality in ['very high', 'high']:
                level = CredibilityLevel.HIGH
                score = 0.9
            elif factuality in ['mostly factual']:
                level = CredibilityLevel.MEDIUM
                score = 0.7
            else:
                level = CredibilityLevel.MEDIUM
                score = 0.6
        else:
            level = CredibilityLevel.LOW
            score = 0.3

        return CredibilityScore(
            level=level,
            score=score,
            method='mbfc_api',
            details={
                'factuality': factuality,
                'bias': bias,
                'mbfc_data': data
            }
        )

    def meets_threshold(self, score: CredibilityScore, threshold: str = 'medium') -> bool:
        """
        Check if credibility score meets minimum threshold.

        Args:
            score: Credibility score
            threshold: Minimum level ("low", "medium", "high")

        Returns:
            True if meets threshold
        """
        threshold_map = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.8
        }

        return score.score >= threshold_map.get(threshold, 0.5)
