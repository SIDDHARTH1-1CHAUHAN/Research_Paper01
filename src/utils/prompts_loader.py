"""
Prompts Loader - Load agent prompts from YAML configuration
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import re


class PromptsLoader:
    """
    Loads and manages agent prompts from prompts.yaml configuration.
    Supports template variable substitution using Jinja2-style {{ variable }} syntax.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize prompts loader.

        Args:
            config_path: Path to prompts.yaml file
        """
        self.config_path = config_path or self._find_config()
        self.prompts = self._load_prompts()
        logger.info(f"Prompts loaded from: {self.config_path}")

    def _find_config(self) -> str:
        """Find prompts configuration file in project structure"""
        possible_paths = [
            Path("config/prompts.yaml"),
            Path("../config/prompts.yaml"),
            Path("../../config/prompts.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        logger.warning("prompts.yaml not found")
        return None

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        if not self.config_path or not Path(self.config_path).exists():
            logger.warning("Prompts config not found, returning empty config")
            return {}

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'claim_decomposition_agent')

        Returns:
            Agent configuration dictionary
        """
        agents = self.prompts.get('agents', {})
        return agents.get(agent_name, {})

    def get_prompt(self, agent_name: str, **variables) -> str:
        """
        Get formatted prompt for an agent with variable substitution.

        Args:
            agent_name: Name of the agent
            **variables: Variables to substitute in the prompt template

        Returns:
            Formatted prompt string
        """
        agent_config = self.get_agent_config(agent_name)
        prompt_template = agent_config.get('prompt', '')

        if not prompt_template:
            logger.warning(f"No prompt found for agent: {agent_name}")
            return ''

        # Substitute variables using {{ variable }} syntax
        formatted_prompt = self._substitute_variables(prompt_template, variables)
        return formatted_prompt

    def _substitute_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Substitute {{ variable }} placeholders with actual values.

        Args:
            template: Prompt template with {{ variable }} placeholders
            variables: Dictionary of variable values

        Returns:
            Formatted string with substituted values
        """
        result = template

        for key, value in variables.items():
            # Handle both {{ var }} and {{var}} syntax
            pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
            result = re.sub(pattern, str(value), result)

        return result

    def get_model(self, agent_name: str) -> str:
        """
        Get the recommended model for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Model name string
        """
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('model', 'qwen-2.5-3b')

    def get_output_schema(self, agent_name: str) -> Dict[str, Any]:
        """
        Get the output schema for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Output schema dictionary
        """
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('output_schema', {})

    def get_pipeline_flow(self) -> list:
        """
        Get the pipeline flow order.

        Returns:
            List of agent names in execution order
        """
        pipeline = self.prompts.get('pipeline', {})
        return pipeline.get('flow', [])

    def list_agents(self) -> list:
        """
        List all available agents.

        Returns:
            List of agent names
        """
        return list(self.prompts.get('agents', {}).keys())


# Singleton instance
_prompts_loader = None


def get_prompts_loader() -> PromptsLoader:
    """Get singleton prompts loader instance"""
    global _prompts_loader
    if _prompts_loader is None:
        _prompts_loader = PromptsLoader()
    return _prompts_loader
