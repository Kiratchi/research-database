# prompts/__init__.py
"""
Prompt loader for TXT files - Smart Methodology Research Agent
Loads prompt templates from separate text files for better maintainability
Structure: prompts/ is at the same level as core/
"""

import os
from pathlib import Path
from typing import Dict

class PromptLoader:
    """Utility class to load prompt templates from TXT files."""
    
    def __init__(self):
        # prompts/ directory is at the same level as core/
        self.prompts_dir = Path(__file__).parent
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from a TXT file.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        # Build file path
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        # Load and cache the prompt
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
        except Exception as e:
            raise IOError(f"Error reading prompt file {prompt_file}: {e}")
        
        self._cache[prompt_name] = prompt_content
        return prompt_content
    
    def get_planning_prompt(self) -> str:
        """Get the planning prompt template."""
        return self.load_prompt('planning_prompt')
    
    def get_execution_prompt(self) -> str:
        """Get the execution prompt template."""
        return self.load_prompt('execution_prompt')
    
    def get_replanning_prompt(self) -> str:
        """Get the replanning prompt template."""
        return self.load_prompt('replanning_prompt')
    
    def reload_prompts(self) -> None:
        """Clear cache and reload all prompts from files."""
        self._cache.clear()
        print("🔄 Prompt cache cleared - prompts will be reloaded from files")

# Create a global instance for easy importing
_loader = PromptLoader()

# Convenience functions for backward compatibility
def get_prompt_template(prompt_type: str) -> str:
    """
    Get prompt template by type.
    
    Args:
        prompt_type: One of 'planning', 'execution', 'replanning'
        
    Returns:
        Prompt template string
        
    Raises:
        ValueError: If prompt_type is not recognized
    """
    prompt_methods = {
        'planning': _loader.get_planning_prompt,
        'execution': _loader.get_execution_prompt,
        'replanning': _loader.get_replanning_prompt
    }
    
    if prompt_type not in prompt_methods:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(prompt_methods.keys())}")
    
    return prompt_methods[prompt_type]()

def reload_all_prompts() -> None:
    """Utility function to reload all prompts from files."""
    _loader.reload_prompts()

# Direct access to templates (loaded once at import time)
try:
    PLANNING_PROMPT_TEMPLATE = _loader.get_planning_prompt()
    EXECUTION_PROMPT_TEMPLATE = _loader.get_execution_prompt()
    REPLANNING_PROMPT_TEMPLATE = _loader.get_replanning_prompt()
    print("✅ Prompt templates loaded successfully from TXT files")
except Exception as e:
    print(f"⚠️ Warning: Could not load prompt templates: {e}")
    # Fallback templates to prevent import errors
    PLANNING_PROMPT_TEMPLATE = "Create a research plan for: {query}"
    EXECUTION_PROMPT_TEMPLATE = "Execute task: {task}"
    REPLANNING_PROMPT_TEMPLATE = "Decide next action based on: {research_summary}"

__all__ = [
    'PromptLoader',
    'get_prompt_template',
    'reload_all_prompts',
    'PLANNING_PROMPT_TEMPLATE',
    'EXECUTION_PROMPT_TEMPLATE', 
    'REPLANNING_PROMPT_TEMPLATE'
]