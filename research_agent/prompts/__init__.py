"""
Simple prompt loader for research agent.
"""
from pathlib import Path

# Get prompts directory
PROMPTS_DIR = Path(__file__).parent

def _load_prompt(filename: str) -> str:
    """Load a prompt file with error handling."""
    try:
        file_path = PROMPTS_DIR / filename
        return file_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        print(f"⚠️ Prompt file not found: {filename}")
        return f"# Fallback prompt for {filename}\nPlease provide your request."
    except Exception as e:
        print(f"⚠️ Error loading {filename}: {e}")
        return f"# Error loading prompt\nPlease provide your request."

# Load all prompts
PLANNING_PROMPT_TEMPLATE = _load_prompt('planning_prompt.txt')
EXECUTION_PROMPT_TEMPLATE = _load_prompt('execution_prompt.txt')
REPLANNING_PROMPT_TEMPLATE = _load_prompt('replanning_prompt.txt')

__all__ = [
    'PLANNING_PROMPT_TEMPLATE',
    'EXECUTION_PROMPT_TEMPLATE',
    'REPLANNING_PROMPT_TEMPLATE'
]