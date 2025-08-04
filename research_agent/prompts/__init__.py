"""
Streamlined prompt loader for TXT files
"""

from pathlib import Path
from functools import lru_cache

# Get prompts directory
PROMPTS_DIR = Path(__file__).parent

@lru_cache(maxsize=None)
def _load_prompt(filename: str) -> str:
    """Load and cache a prompt file."""
    try:
        return (PROMPTS_DIR / filename).read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_DIR / filename}")
    except Exception as e:
        raise IOError(f"Error reading {filename}: {e}")

# Load prompts with fallbacks
try:
    PLANNING_PROMPT_TEMPLATE = _load_prompt('planning_prompt.txt')
    EXECUTION_PROMPT_TEMPLATE = _load_prompt('execution_prompt.txt')
    REPLANNING_PROMPT_TEMPLATE = _load_prompt('replanning_prompt.txt')
    print("✅ Prompt templates loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load prompt templates: {e}")
    # Fallback templates
    PLANNING_PROMPT_TEMPLATE = "Create a research plan for: {query}"
    EXECUTION_PROMPT_TEMPLATE = "Execute task: {task}"
    REPLANNING_PROMPT_TEMPLATE = "Decide next action based on: {research_summary}"

def reload_prompts():
    """Clear cache and reload all prompts."""
    _load_prompt.cache_clear()
    globals().update({
        'PLANNING_PROMPT_TEMPLATE': _load_prompt('planning_prompt.txt'),
        'EXECUTION_PROMPT_TEMPLATE': _load_prompt('execution_prompt.txt'),
        'REPLANNING_PROMPT_TEMPLATE': _load_prompt('replanning_prompt.txt')
    })
    print("🔄 Prompts reloaded")

__all__ = [
    'PLANNING_PROMPT_TEMPLATE',
    'EXECUTION_PROMPT_TEMPLATE', 
    'REPLANNING_PROMPT_TEMPLATE',
    'reload_prompts'
]