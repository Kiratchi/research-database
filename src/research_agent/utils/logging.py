"""
Logging utilities for the research agent.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the research agent.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    # Set up logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Default log format
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger('research_agent')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file() -> str:
    """
    Get the default log file path.
    
    Returns:
        Default log file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/research_agent_{timestamp}.log"


class AgentLogger:
    """
    Logger wrapper for the research agent with structured logging.
    """
    
    def __init__(self, name: str = "research_agent", level: str = "INFO"):
        """
        Initialize the agent logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = setup_logging(level=level)
        self.name = name
    
    def log_query_start(self, query: str):
        """Log the start of a query."""
        self.logger.info(f"Starting query: {query}")
    
    def log_query_end(self, query: str, result: dict):
        """Log the end of a query."""
        response = result.get("response", "No response")
        self.logger.info(f"Query completed: {query} -> {response[:100]}...")
    
    def log_plan_generated(self, plan: list):
        """Log when a plan is generated."""
        self.logger.info(f"Plan generated with {len(plan)} steps:")
        for i, step in enumerate(plan, 1):
            self.logger.info(f"  {i}. {step}")
    
    def log_step_start(self, step: str, step_number: int):
        """Log the start of a step execution."""
        self.logger.info(f"Executing step {step_number}: {step}")
    
    def log_step_end(self, step: str, result: str, step_number: int):
        """Log the end of a step execution."""
        self.logger.info(f"Step {step_number} completed: {step} -> {result[:100]}...")
    
    def log_error(self, error: str, context: dict = None):
        """Log an error with context."""
        self.logger.error(f"Error: {error}")
        if context:
            self.logger.error(f"Context: {context}")
    
    def log_warning(self, warning: str):
        """Log a warning."""
        self.logger.warning(warning)
    
    def log_debug(self, message: str, data: dict = None):
        """Log debug information."""
        self.logger.debug(message)
        if data:
            self.logger.debug(f"Data: {data}")
    
    def log_tool_call(self, tool_name: str, parameters: dict, result: str):
        """Log a tool call."""
        self.logger.debug(f"Tool call: {tool_name}")
        self.logger.debug(f"Parameters: {parameters}")
        self.logger.debug(f"Result: {result[:200]}...")
    
    def log_state_update(self, state: dict):
        """Log state updates."""
        self.logger.debug(f"State update: {state}")


# Global logger instance
agent_logger = None


def get_logger() -> AgentLogger:
    """
    Get the global agent logger instance.
    
    Returns:
        Global agent logger
    """
    global agent_logger
    if agent_logger is None:
        agent_logger = AgentLogger()
    return agent_logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger()
    
    # Test logging
    logger.log_query_start("How many papers has Christian Fager published?")
    logger.log_plan_generated([
        "Search for publications by Christian Fager",
        "Count the total number of publications"
    ])
    logger.log_step_start("Search for publications by Christian Fager", 1)
    logger.log_step_end("Search for publications by Christian Fager", "Found 25 publications", 1)
    logger.log_query_end("How many papers has Christian Fager published?", {"response": "25 publications"})
    
    # Test error logging
    logger.log_error("Test error", {"context": "test"})
    logger.log_warning("Test warning")
    logger.log_debug("Test debug", {"data": "test"})