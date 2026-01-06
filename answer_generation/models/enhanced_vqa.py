"""
Enhanced VQA Model with video context generation support.
"""

from .base_vqa import BaseVQAModel
from .utils.prompt_handler import PromptHandler
from typing import Any, Dict
from prompt_builder import (
    create_video_context_prompt,
    create_enhanced_audio_visual_prompt,
    create_visual_only_prompt,
    create_audio_visual_prompt
)


class EnhancedVQAModel(BaseVQAModel):
    """
    Enhanced VQA model that supports two-step inference:
    1. Generate video context description
    2. Use video context + audio + question for final answer
    """

    def __init__(self, config=None):
        super().__init__(config)
        
        # Debug configuration loading
        print(f"🔧 DEBUG: Enhanced VQA Model initialization")
        print(f"🔧 DEBUG: Config type: {type(self.config)}")
        print(f"🔧 DEBUG: Enhanced section: {self.config.get('enhanced_prompting', 'NOT FOUND')}")
        
        self.enhanced_prompting_enabled = self.config.get("enhanced_prompting", {}).get("enabled", False)
        self.video_context_generation = self.config.get("enhanced_prompting", {}).get("video_context_generation", True)
        self.use_enhanced_template = self.config.get("enhanced_prompting", {}).get("use_audio_visual_enhanced", True)
        
        print(f"🔧 DEBUG: enhanced_prompting_enabled = {self.enhanced_prompting_enabled} ({type(self.enhanced_prompting_enabled)})")
        
        # Override template type if enhanced prompting is enabled
        if self.enhanced_prompting_enabled:
            prompt_type = self.config.get("prompt", {}).get("type", "enhanced_audio_visual")
            self.template_type = prompt_type
            # Recreate prompt handler with enhanced template
            self.prompt_handler = PromptHandler(
                template_name=self.template_type,
                config_path=self.config.get("prompt_config_path", None),
            )
            print(f"Enhanced prompting enabled - Updated prompt template to: {self.template_type}")
        
        print(f"Enhanced prompting: {'enabled' if self.enhanced_prompting_enabled else 'disabled'}")
        if self.enhanced_prompting_enabled:
            print(f"Video context generation: {'enabled' if self.video_context_generation else 'disabled'}")
            print(f"Enhanced audio-visual template: {'enabled' if self.use_enhanced_template else 'disabled'}")

    def infer(self, video_path: str, question_or_data: Any, logger, traceid: str) -> str:
        """
        Enhanced inference with optional video context generation.
        """
        try:
            # Prepare initial inference data
            inference_data = self._prepare_inference_data(
                video_path, question_or_data, logger, traceid
            )

            # Check if enhanced prompting is enabled
            if not self.enhanced_prompting_enabled:
                # Use standard inference
                logger.info(f"trace_id: {traceid} - Using standard inference")
                return self._run_model_inference(inference_data, logger, traceid)

            # Enhanced prompting workflow
            logger.info(f"trace_id: {traceid} - Using enhanced prompting workflow")
            
            video_context = ""
            if self.video_context_generation:
                # Step 1: Generate video context description
                video_context = self._generate_video_context(inference_data, logger, traceid)
                logger.info(f"trace_id: {traceid} - Generated video context: {video_context[:100]}...")

            # Step 2: Create enhanced prompt and run final inference
            enhanced_result = self._run_enhanced_inference(
                inference_data, video_context, logger, traceid
            )

            logger.info(f"trace_id: {traceid} - Enhanced inference completed successfully")
            return enhanced_result

        except Exception as e:
            error_msg = f"Enhanced inference failed: {str(e)}"
            logger.error(f"trace_id: {traceid} - {error_msg}")
            logger.exception(e)
            return f"Error: {error_msg}"

    def _generate_video_context(self, inference_data: Dict[str, Any], logger, traceid: str) -> str:
        """
        Generate video context description using the model.
        """
        logger.info(f"trace_id: {traceid} - Generating video context description")
        
        # Create context generation prompt
        context_prompt = create_video_context_prompt()
        
        # Create temporary inference data for context generation
        context_inference_data = inference_data.copy()
        context_inference_data["prompt"] = context_prompt
        context_inference_data["question"] = "Describe this video"
        
        # Generate video context
        video_context = self._run_model_inference(context_inference_data, logger, f"{traceid}_context")
        
        return video_context.strip() if video_context else "No video context generated."

    def _run_enhanced_inference(
        self,
        inference_data: Dict[str, Any],
        video_context: str,
        logger,
        traceid: str
    ) -> str:
        """
        Run the final inference using enhanced prompt with video context.
        """
        logger.info(f"trace_id: {traceid} - Running enhanced inference with video context")
        
        # Extract data
        question = inference_data["question"]
        data = inference_data["data"]
        
        # Extract audio transcription
        speech_recognition = data.get("speech_recognition", {})
        speech_text = speech_recognition.get("recognized_text", "")
        audio_text = data.get("audio_transcription", "")
        audio_transcription = speech_text or audio_text or ""
        
        # Create enhanced prompt
        if audio_transcription and self.use_enhanced_template:
            # Use enhanced audio-visual prompt with video context
            enhanced_prompt = create_enhanced_audio_visual_prompt(
                question=question,
                video_context=video_context,
                audio_transcription=audio_transcription
            )
            logger.info(f"trace_id: {traceid} - Using enhanced audio-visual prompt")
        elif audio_transcription:
            # Use standard audio-visual prompt
            enhanced_prompt = create_audio_visual_prompt(
                question=question,
                visual_context=video_context,
                audio_transcription=audio_transcription
            )
            logger.info(f"trace_id: {traceid} - Using standard audio-visual prompt")
        else:
            # Use visual-only prompt with video context
            enhanced_prompt = create_visual_only_prompt(
                question=question,
                video_context=video_context
            )
            logger.info(f"trace_id: {traceid} - Using visual-only prompt with context")
        
        # Update inference data with enhanced prompt
        enhanced_inference_data = inference_data.copy()
        enhanced_inference_data["prompt"] = enhanced_prompt
        
        # Run final inference
        return self._run_model_inference(enhanced_inference_data, logger, traceid)
