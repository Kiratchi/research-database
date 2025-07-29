"""
Prompt loader utility for research agent workflow.
"""

import os
from pathlib import Path
from typing import Dict

def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Prompt content as string
    """
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    prompts_dir = current_dir.parent / "prompts"
    
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_all_prompts() -> Dict[str, str]:
    """
    Load all prompt templates from the prompts directory.
    
    Returns:
        Dictionary mapping prompt names to their content
    """
    current_dir = Path(__file__).parent
    prompts_dir = current_dir.parent / "prompts"
    
    prompts = {}
    
    if prompts_dir.exists():
        for prompt_file in prompts_dir.glob("*.txt"):
            prompt_name = prompt_file.stem
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read().strip()
    
    return prompts


def format_prompt(prompt_template: str, **kwargs) -> str:
    """
    Format a prompt template with the provided variables.
    
    Args:
        prompt_template: The prompt template string
        **kwargs: Variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")


# Convenience functions for specific prompts
def get_executor_prompt(execution_tool_descriptions: str) -> str:
    """Get the formatted executor prompt."""
    template = load_prompt("executor_prompt")
    return format_prompt(template, execution_tool_descriptions=execution_tool_descriptions)


def get_planner_system_prompt(planning_tool_descriptions: str) -> str:
    """Get the formatted planner system prompt."""
    template = load_prompt("planner_system_prompt")
    return format_prompt(template, planning_tool_descriptions=planning_tool_descriptions)


def get_replanner_prompt(planning_tool_descriptions: str) -> str:
    """Get the formatted replanner prompt."""
    template = load_prompt("replanner_prompt")
    return format_prompt(template, planning_tool_descriptions=planning_tool_descriptions)


def get_task_format_template(original_query: str, plan_str: str, task: str) -> str:
    """Get the formatted task template."""
    template = load_prompt("task_format_template")
    return format_prompt(template, 
                        original_query=original_query,
                        plan_str=plan_str,
                        task=task)


def get_context_aware_prompt(context_summary: str, query: str) -> str:
    """Get the formatted context-aware prompt."""
    template = load_prompt("context_aware_prompt")
    return format_prompt(template,
                        context_summary=context_summary,
                        query=query)


def get_standard_planning_prompt(query: str) -> str:
    """Get the formatted standard planning prompt."""
    template = load_prompt("standard_planning_prompt")
    return format_prompt(template, query=query)