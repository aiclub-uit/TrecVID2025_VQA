"""
Base inference class with common logic for VQA models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .utils.utils import video_to_frames, ensure_cache_dir, detect_model_format
from .utils.prompt_handler import PromptHandler


class BaseVQAModel(ABC):
    """
    Base class for VQA models with common functionality.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.model_name = self.config.get("model_name", "")

        # Initialize shared components
        self.template_type = self.config.get("prompt_template", "audio_enhanced")
        self.model_format = detect_model_format(self.model_name)

        # Initialize prompt handler with the template name from config
        self.prompt_handler = PromptHandler(
            template_name=self.template_type,
            config_path=self.config.get("prompt_config_path", None),
        )

        # Common configuration
        self.cache_dir = ensure_cache_dir(self.config.get("cache_dir", "caches"))
        self.num_frames = self.config.get("num_frames", 16)

        print(f"Using model: {self.model_name}")
        print(f"Using cache directory: {self.cache_dir}")
        print(f"Using prompt template: {self.template_type}")

    def _extract_frames(self, video_path: str, logger, traceid: str) -> List[Any]:
        """
        Extract frames from video with error handling.

        Args:
            video_path: Path to video file
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            List of extracted frames

        Raises:
            RuntimeError: If frame extraction fails
        """
        frames = video_to_frames(video_path, num_frames=self.num_frames)

        if not frames:
            error_msg = f"Could not extract frames from video {video_path}"
            logger.error(f"trace_id: {traceid} - {error_msg}")
            raise RuntimeError(f"Error: {error_msg}")

        return frames

    def _prepare_inference_data(
        self, video_path: str, question_or_data: Any, logger, traceid: str
    ) -> Dict[str, Any]:
        """
        Prepare all data needed for inference.

        Args:
            video_path: Path to video file
            question_or_data: Question string or data dictionary
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Dictionary containing prepared data
        """
        logger.info(
            f"trace_id: {traceid} - Starting inference with model {self.model_name} on video {video_path}"
        )

        # Process question and create prompt
        question, prompt, data = self.prompt_handler.process_question_data(
            question_or_data=question_or_data,
            video_path=video_path,
            model_format=self.model_format,
            logger=logger,
            traceid=traceid,
        )

        # Extract frames
        frames = self._extract_frames(video_path, logger, traceid)

        return {"question": question, "prompt": prompt, "data": data, "frames": frames, "video_path": video_path}

    @abstractmethod
    def _run_model_inference(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ) -> str:
        """
        Run the actual model inference. Must be implemented by subclasses.

        Args:
            inference_data: Prepared inference data
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Model output string
        """
        pass

    def infer(
        self, video_path: str, question_or_data: Any, logger, traceid: str
    ) -> str:
        """
        Run inference on a video to answer a question.
        Supports both legacy string questions and new data format with audio.

        Args:
            video_path: The path to the video file.
            question_or_data: Either a string question (legacy) or dict with full data including audio
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            The model's output as a string.
        """
        try:
            # Prepare all inference data
            inference_data = self._prepare_inference_data(
                video_path, question_or_data, logger, traceid
            )

            # Run model-specific inference
            result = self._run_model_inference(inference_data, logger, traceid)

            logger.info(f"trace_id: {traceid} - Inference completed successfully")
            return result

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(f"trace_id: {traceid} - {error_msg}")
            logger.exception(e)
            return f"Error: {error_msg}"
