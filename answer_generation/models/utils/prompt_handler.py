"""
Shared prompt handling logic for VQA models.
"""

from typing import Dict, Any, Tuple

try:
    from prompt_builder import (
        create_visual_only_prompt,
        create_audio_visual_prompt,
        create_enhanced_audio_visual_prompt,
        create_video_context_prompt,
        create_visual_only_prompt_with_previous,
        create_audio_visual_prompt_with_previous,
        create_enhanced_audio_visual_prompt_with_previous
    )
except ImportError:
    # Fallback if prompt_builder is not available
    create_visual_only_prompt = None
    create_audio_visual_prompt = None
    create_enhanced_audio_visual_prompt = None
    create_video_context_prompt = None
    create_visual_only_prompt_with_previous = None
    create_audio_visual_prompt_with_previous = None
    create_enhanced_audio_visual_prompt_with_previous = None

try:
    from .prompt_config_manager import PromptConfigManager
except ImportError:
    PromptConfigManager = None


class PromptHandler:
    """Enhanced prompt handler using YAML-based configuration templates."""

    def __init__(self, template_name: str = "audio_enhanced", config_path: str = None):
        """
        Initialize prompt handler with specified template.

        Args:
            template_name: Name of the template to use from YAML config
            config_path: Optional path to custom prompt config file
        """
        self.template_name = template_name
        self.prefer_audio_visual = True  # Keep for backward compatibility
        
        # Initialize the new configuration manager
        if PromptConfigManager:
            try:
                self.config_manager = PromptConfigManager(config_path)
                self.use_yaml_config = True
                print(f"✓ Using YAML prompt template: {template_name}")
            except Exception as e:
                print(f"⚠️ Could not load YAML config, falling back to legacy templates: {e}")
                self.config_manager = None
                self.use_yaml_config = False
        else:
            self.config_manager = None
            self.use_yaml_config = False

    def create_prompt(self, question: str, data: Dict[str, Any] = None) -> str:
        """
        Create a prompt using the configured template system.

        Args:
            question: The question to be answered
            data: Data dictionary containing audio, video context, etc.

        Returns:
            Formatted prompt string
        """
        data = data or {}
        
        if self.use_yaml_config and self.config_manager:
            # Use new YAML-based configuration
            try:
                # Extract audio transcription
                speech_recognition = data.get("speech_recognition") or {}
                speech_text = speech_recognition.get("recognized_text", "")
                audio_text = data.get("audio_transcription", "")
                audio_transcription = speech_text or audio_text

                # Extract video context
                video_context = ""
                if data.get("video_path"):
                    video_context = f"Video file: {data['video_path']}"
                elif data.get("video_context"):
                    video_context = data["video_context"]

                return self.config_manager.create_prompt(
                    question=question,
                    template_name=self.template_name,
                    audio_transcription=audio_transcription,
                    video_context=video_context
                )
            except Exception as e:
                print(f"⚠️ Error using YAML config, falling back to legacy: {e}")
        
        # Fallback to legacy prompt creation
        has_audio = self.has_audio(data)
        previous_answers = data.get("previous_answers", []) if data else []
        
        # Check if we should use enhanced prompting
        if self.template_name == "enhanced_audio_visual":
            return self.create_enhanced_audio_visual_prompt_text(question, data)
        elif has_audio and self.prefer_audio_visual:
            return self.create_audio_visual_prompt_text(question, data)
        else:
            return self.create_visual_only_prompt_text(question, data)

    def has_audio(self, data: Dict[str, Any]) -> bool:
        """Check if audio transcription is available and non-empty."""
        if isinstance(data, dict):
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            return bool(speech_text.strip() or audio_text.strip())
        return False

    def create_visual_only_prompt_text(
        self, question: str, data: Dict[str, Any] = None
    ) -> str:
        """Create a simple visual-only prompt."""
        previous_answers = data.get("previous_answers", []) if data else []
        
        if create_visual_only_prompt_with_previous and previous_answers:
            video_context = ""
            if data and isinstance(data, dict) and data.get("video_path"):
                video_context = f"Video file: {data['video_path']}"

            return create_visual_only_prompt_with_previous(
                question=question, video_context=video_context, previous_answers=previous_answers
            )
        elif create_visual_only_prompt:
            video_context = ""
            if data and isinstance(data, dict) and data.get("video_path"):
                video_context = f"Video file: {data['video_path']}"

            return create_visual_only_prompt(
                question=question, video_context=video_context
            )
        else:
            # Fallback template
            return f"Based on the video content, please answer the following question:\n\nQuestion: {question}\n\nAnswer:"

    def create_audio_visual_prompt_text(
        self, question: str, data: Dict[str, Any]
    ) -> str:
        """Create an audio-visual multimodal prompt."""
        previous_answers = data.get("previous_answers", []) if data else []
        
        if create_audio_visual_prompt_with_previous and previous_answers:
            # Extract audio transcription
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = (
                speech_text or audio_text or "No audio transcription available."
            )

            # Create visual context description
            visual_context = "Video content analysis"
            if data.get("video_path"):
                visual_context = f"Video file: {data['video_path']}"

            return create_audio_visual_prompt_with_previous(
                question=question,
                visual_context=visual_context,
                audio_transcription=audio_transcription,
                previous_answers=previous_answers
            )
        elif create_audio_visual_prompt:
            # Extract audio transcription
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = (
                speech_text or audio_text or "No audio transcription available."
            )

            # Create visual context description
            visual_context = "Video content analysis"
            if data.get("video_path"):
                visual_context = f"Video file: {data['video_path']}"

            return create_audio_visual_prompt(
                question=question,
                visual_context=visual_context,
                audio_transcription=audio_transcription,
            )
        else:
            # Fallback template
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = speech_text or audio_text or "No audio available."

            return f"""You are an expert multi-modal AI assistant. Analyze the video content and audio transcription to answer the question accurately.

### VIDEO CONTENT ###
Video content analysis

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###

"""

    def create_enhanced_audio_visual_prompt_text(
        self, question: str, data: Dict[str, Any]
    ) -> str:
        """Create an enhanced audio-visual prompt with video context."""
        
        previous_answers = data.get("previous_answers", []) if data else []
        
        # Debug logging
        print("DEBUG: create_enhanced_audio_visual_prompt_text called")
        print(f"DEBUG: data keys: {list(data.keys()) if data else 'None'}")
        print(f"DEBUG: video_context in data: {'video_context' in data if data else False}")
        if data and 'video_context' in data:
            print(f"DEBUG: video_context value: {data['video_context'][:100]}...")
        
        if create_enhanced_audio_visual_prompt_with_previous and previous_answers:
            # Extract audio transcription
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = (
                speech_text or audio_text or "No audio transcription available."
            )

            # Extract video context (will be populated by enhanced model)
            video_context = data.get("video_context", "")
            
            print(f"DEBUG: Checking video_context - Raw value: '{video_context}'")
            print(f"DEBUG: Video context type: {type(video_context)}, length: {len(video_context) if video_context else 0}")

            return create_enhanced_audio_visual_prompt_with_previous(
                question=question,
                video_context=video_context,
                audio_transcription=audio_transcription,
                previous_answers=previous_answers
            )
        elif create_enhanced_audio_visual_prompt:
            # Extract audio transcription
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = (
                speech_text or audio_text or "No audio transcription available."
            )

            # Extract video context (will be populated by enhanced model)
            video_context = data.get("video_context", "")
            
            print(f"DEBUG: Checking video_context - Raw value: '{video_context}'")
            print(f"DEBUG: Video context type: {type(video_context)}, length: {len(video_context) if video_context else 0}")
            
            # Check if we have actual video context
            if not video_context:
                print("⚠️  WARNING: No video_context key in data - enhanced prompting not configured properly!")
                video_context = "Video content analysis pending."
            elif video_context in ["Video content will be analyzed.", "Video content will be analyzed during processing."]:
                print("⚠️  WARNING: Default video context fallback detected - context generation may have failed!")
                print(f"⚠️  Received context: {video_context}")
                video_context = "Video content analysis pending."
            else:
                print(f"✅ Using generated video context: {video_context[:50]}...")

            print(f"🔧 DEBUG: Final video_context being used: '{video_context[:100]}...'")

            return create_enhanced_audio_visual_prompt(
                question=question,
                video_context=video_context,
                audio_transcription=audio_transcription,
            )
        else:
            # Fallback to enhanced template manually
            speech_recognition = data.get("speech_recognition") or {}
            speech_text = speech_recognition.get("recognized_text", "")
            audio_text = data.get("audio_transcription", "")
            audio_transcription = speech_text or audio_text or "No audio available."
            video_context = data.get("video_context", "Video content analysis pending.")

            return f"""Analyze the video content, audio transcription, and video context description, then respond with only the answer, concisely and without restating the question.

### VIDEO CONTEXT ###
{video_context}

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###

"""

    def process_question_data(
        self,
        question_or_data: Any,
        video_path: str,
        model_format: str,
        logger,
        traceid: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process question data and create appropriate prompt using the new template system.

        Args:
            question_or_data: Either string question or data dict
            video_path: Path to video file
            model_format: Model format for optimization (preserved for compatibility)
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Tuple of (question, prompt, data)
        """
        if isinstance(question_or_data, str):
            # Legacy format - just a question string
            question = question_or_data
            data = {"video_path": video_path, "video_exists": True}
            prompt = self.create_prompt(question, data)
            logger.info(
                f"trace_id: {traceid} - Using legacy format with template: {self.template_name}"
            )
        else:
            # New format - full data dictionary with potential audio information
            data = question_or_data
            question = data.get("question", "")
            
            # Debug: Check what data we received
            logger.info(f"trace_id: {traceid} - Received data keys: {list(data.keys())}")
            if 'video_context' in data:
                logger.info(f"trace_id: {traceid} - Video context found in data: {data['video_context'][:50]}...")
            else:
                logger.info(f"trace_id: {traceid} - No video_context in data")
            
            # Add video path to data if not present
            if "video_path" not in data:
                data["video_path"] = video_path

            # Create prompt using the new system
            prompt = self.create_prompt(question, data)
            
            # Log which type of prompt was used
            has_audio = self.has_audio(data)
            prompt_type = "audio-visual" if has_audio and self.prefer_audio_visual else "visual-only"
            logger.info(f"trace_id: {traceid} - Using {prompt_type} template: {self.template_name}")

            # Log audio information if available
            if has_audio:
                speech_recognition = data.get("speech_recognition") or {}
                speech_text = speech_recognition.get("recognized_text", "")
                audio_text = data.get("audio_transcription", "")
                audio_content = speech_text or audio_text
                logger.info(
                    f"trace_id: {traceid} - Audio transcript: '{audio_content[:50]}...'"
                )

        # Print the prompt used for each inference
        print(f"\n=== PROMPT USED (trace_id: {traceid}) ===")
        print(f"Template: {self.template_name}")
        print(prompt)
        print("=" * 50)

        logger.info(f"trace_id: {traceid} - Prompt created successfully using template: {self.template_name}")
        return question, prompt, data
