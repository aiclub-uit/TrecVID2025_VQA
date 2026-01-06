"""Custom logger implementation using loguru for better logging capabilities."""

import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger as loguru_logger


class CustomLogger:
    """
    A custom logger class that wraps loguru for enhanced logging capabilities.
    
    Features:
    - Colored console output with timestamps
    - Automatic file logging with rotation
    - Context-aware logging with trace IDs
    - Performance timing utilities
    - Different log levels with custom formatting
    """
    
    def __init__(
        self,
        name: str = "VQA",
        log_dir: Optional[Union[str, Path]] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        rotation: str = "10 MB",
        retention: str = "7 days",
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Initialize the custom logger.
        
        Args:
            name: Logger name/identifier
            log_dir: Directory to save log files (if None, uses ./logs)
            console_level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            file_level: File logging level
            rotation: Log file rotation policy
            retention: Log file retention policy
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.console_level = console_level
        self.file_level = file_level
        self.rotation = rotation
        self.retention = retention
        
        # Remove default handler
        loguru_logger.remove()
        
        # Setup console handler
        if enable_console:
            self._setup_console_handler()
        
        # Setup file handler
        if enable_file:
            self._setup_file_handler()
        
        # Store the configured logger
        self.logger = loguru_logger
    
    def _setup_console_handler(self):
        """Setup console logging handler with colored output."""
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        loguru_logger.add(
            sys.stdout,
            format=console_format,
            level=self.console_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    def _setup_file_handler(self):
        """Setup file logging handler with rotation."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = self.log_dir / f"{self.name.lower()}.log"
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        loguru_logger.add(
            str(log_file),
            format=file_format,
            level=self.file_level,
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
        # Error-only log file
        error_log_file = self.log_dir / f"{self.name.lower()}_errors.log"
        loguru_logger.add(
            str(error_log_file),
            format=file_format,
            level="ERROR",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message (loguru special level)."""
        self.logger.success(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    def with_context(self, **context):
        """Add context to logger (useful for trace IDs, batch info, etc.)."""
        return self.logger.bind(**context)
    
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.time_operation("batch_inference"):
                # operation code here
                pass
        """
        return TimedOperation(self.logger, operation_name)


class TimedOperation:
    """Context manager for timing operations with automatic logging."""
    
    def __init__(self, logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.success(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions


def create_logger(
    name: str = "VQA",
    log_dir: Optional[Union[str, Path]] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    **kwargs
) -> CustomLogger:
    """
    Factory function to create a configured logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
        **kwargs: Additional arguments for CustomLogger
    
    Returns:
        Configured CustomLogger instance
    """
    return CustomLogger(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        **kwargs
    )
