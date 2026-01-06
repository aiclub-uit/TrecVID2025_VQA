from vllm import LLM, SamplingParams
from .enhanced_vqa import EnhancedVQAModel
from typing import Dict, Any, List
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt_builder import create_video_context_prompt


class VLLMModel(EnhancedVQAModel):
    """
    A wrapper for vLLM models for text generation.
    """

    def __init__(self, config=None):
        # Set default model name before calling super().__init__
        if config is None:
            config = {}

        print("🔧 DEBUG: VLLMModel.__init__ called")
        print(f"🔧 DEBUG: Config type: {type(config)}")
        print(f"🔧 DEBUG: Config keys: {list(config.keys()) if config else []}")

        super().__init__(config)

        print(
            "🔧 DEBUG: After super().__init__(), enhanced_prompting_enabled =",
            getattr(self, "enhanced_prompting_enabled", "NOT_SET"),
        )

        # VLLM-specific configuration
        self.gpu_memory_utilization = self.config.get("gpu_memory_utilization", 0.8)
        self.max_model_len = self.config.get(
            "max_model_len", 30000
        )  # Reduced to match vLLM config

        print(f"Using model format: {self.model_format}")
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,  # Required for multi-modal models
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={
                "image": 256,
                "video": 5,
            },  # Support both image and video
            # tokenizer_mode="slow",  # Avoid CachedQwen2TokenizerFast type mismatch
            disable_log_stats=True,  # Reduce logging overhead
            # model_impl="transformers",  # Use transformers implementation
        )
        print("Model initialized successfully.")
        self.tokenizer = self.llm.get_tokenizer()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _get_aria_placeholders_for_videos(self, frames: List, timestamps=None):
        """
        Create placeholders for Aria model video processing.

        Args:
            frames: List of video frames
            timestamps: Optional list of timestamps

        Returns:
            List of content dictionaries for Aria format
        """
        contents = []
        if not timestamps:
            for i, _ in enumerate(frames):
                contents.append({"text": None, "type": "image"})
            contents.append({"text": "\n", "type": "text"})
        else:
            for i, (_, ts) in enumerate(zip(frames, timestamps)):
                contents.extend(
                    [
                        {
                            "text": f"[{int(ts) // 60:02d}:{int(ts) % 60:02d}]",
                            "type": "text",
                        },
                        {"text": None, "type": "image"},
                        {"text": "\n", "type": "text"},
                    ]
                )
        return contents

    def _create_qwen_prompt(self, inference_data: Dict[str, Any], logger, traceid: str):
        """
        Create prompt format for Qwen2.5-VL model.

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Tuple of (prompt_text, llm_inputs dict)
        """
        frames = inference_data["frames"]
        video_path = inference_data["video_path"]
        prompt = inference_data["prompt"]

        # Use all frames but limit to a reasonable number for stability
        # max_frames = min(32, len(frames))
        processed_frames = frames if frames else []
        logger.info(
            f"trace_id: {traceid} - Using {len(processed_frames)} frames (from {len(frames)} total)"
        )

        if processed_frames:
            content = []

            # Always use video format for multiple frames, image format for single frame
            if len(processed_frames) > 1:
                # Multiple frames - use video format with original video path
                content.append({"type": "text", "text": prompt})
                content.append(
                    {
                        "type": "video",
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28,
                        "fps": 1,
                    }
                )
            else:
                # Single frame - use image format
                content.append({"type": "text", "text": prompt})
                content.append({"type": "image", "image": processed_frames[0]})

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]

        # Apply chat template using processor
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info to get proper format
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare multi-modal data
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs
            logger.info(
                f"trace_id: {traceid} - Using video input with {len(video_inputs)} items"
            )
        elif image_inputs is not None:
            mm_data["image"] = image_inputs
            logger.info(
                f"trace_id: {traceid} - Using image input with {len(image_inputs)} items"
            )

        # Prepare the request
        llm_inputs = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
        }

        return prompt_text, llm_inputs

    def _create_aria_prompt(self, inference_data: Dict[str, Any], logger, traceid: str):
        """
        Create prompt format for Aria model.

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Tuple of (prompt_token_ids, llm_inputs dict)
        """
        frames = inference_data["frames"]
        prompt = inference_data["prompt"]

        # Use all frames but limit to a reasonable number
        max_frames = min(128, len(frames))  # Aria can handle more frames
        processed_frames = frames[:max_frames] if frames else []
        logger.info(
            f"trace_id: {traceid} - Using {len(processed_frames)} frames (from {len(frames)} total)"
        )

        # Create placeholders for frames
        contents = self._get_aria_placeholders_for_videos(processed_frames)

        # Add the actual prompt text
        contents.append({"text": prompt, "type": "text"})

        messages = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        # Apply chat template to get token IDs
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Prepare multi-modal data for Aria
        llm_inputs = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {
                "image": processed_frames,
                # "max_image_size": 490,  # The max image patch size
                # "split_image": False,  # whether to split the images
            },
        }

        return prompt_token_ids, llm_inputs

    def _run_model_inference(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ) -> str:
        """
        Run VLLM-specific inference with model-specific prompt formats.

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Model output string
        """
        frames = inference_data["frames"]
        video_path = inference_data["video_path"]

        # Debug: Log frame information
        logger.info(
            f"trace_id: {traceid} - Processing {len(frames)} frames from {video_path}"
        )
        logger.info(
            f"trace_id: {traceid} - Frame types: {[type(f).__name__ for f in frames[:3]]}..."
        )

        # Validate first few frames
        for i, frame in enumerate(frames[:3]):
            if hasattr(frame, "mode") and hasattr(frame, "size"):  # It's a PIL image
                logger.info(
                    f"trace_id: {traceid} - Frame {i}: {frame.mode}, {frame.size}"
                )
            else:
                logger.warning(
                    f"trace_id: {traceid} - Frame {i} is not a PIL image: {type(frame)}"
                )

        logger.info(f"trace_id: {traceid} - Using model format: {self.model_format}")

        # Create model-specific prompt format
        if self.model_format == "aria":
            logger.info(f"trace_id: {traceid} - Using Aria prompt format")
            prompt_data, llm_inputs = self._create_aria_prompt(
                inference_data, logger, traceid
            )

            # Check for custom sampling parameters or use defaults
            if "custom_sampling" in inference_data:
                custom_config = inference_data["custom_sampling"]
                logger.info(f"trace_id: {traceid} - Using custom sampling parameters for context generation")
            else:
                custom_config = self.config.get("sampling_params", {})
                
            sampling_params = SamplingParams(
                max_tokens=custom_config.get("max_tokens", 256),
                top_k=custom_config.get("top_k", 1),
                top_p=custom_config.get("top_p", 1.0),
                temperature=custom_config.get("temperature", 0.0),
                stop=custom_config.get("stop", ["<|im_end|>"]),
            )
        elif self.model_format == "qwen":
            logger.info(f"trace_id: {traceid} - Using Qwen2.5-VL prompt format")
            prompt_data, llm_inputs = self._create_qwen_prompt(
                inference_data, logger, traceid
            )

            # Check for custom sampling parameters or use defaults
            if "custom_sampling" in inference_data:
                custom_config = inference_data["custom_sampling"]
                logger.info(f"trace_id: {traceid} - Using custom sampling parameters for context generation")
            else:
                custom_config = self.config.get("sampling_params", {})
                
            sampling_params = SamplingParams(
                max_tokens=custom_config.get("max_tokens", 1024),
                temperature=custom_config.get("temperature", 0.1),
                top_p=custom_config.get("top_p", 0.95),
                stop=custom_config.get("stop", []),
            )
        else:
            logger.warning(
                f"trace_id: {traceid} - Unknown model format {self.model_format}, using Qwen format as fallback"
            )
            prompt_data, llm_inputs = self._create_qwen_prompt(
                inference_data, logger, traceid
            )

            # Check for custom sampling parameters or use defaults
            if "custom_sampling" in inference_data:
                custom_config = inference_data["custom_sampling"]
                logger.info(f"trace_id: {traceid} - Using custom sampling parameters for context generation")
            else:
                custom_config = self.config.get("sampling_params", {})
                
            sampling_params = SamplingParams(
                max_tokens=custom_config.get("max_tokens", 1024),
                temperature=custom_config.get("temperature", 0.1),
                top_p=custom_config.get("top_p", 0.95),
                stop=custom_config.get("stop", []),
            )

        logger.info(
            f"trace_id: {traceid} - Running inference with model {self.model_name}"
        )
        
        # Debug: Log sampling parameters
        if "custom_sampling" in inference_data:
            logger.info(f"trace_id: {traceid} - Custom sampling params: {inference_data['custom_sampling']}")
            logger.info(f"trace_id: {traceid} - Actual sampling_params: max_tokens={sampling_params.max_tokens}, temp={sampling_params.temperature}")

        # Run inference with model-specific format
        try:
            logger.info(f"trace_id: {traceid} - Calling llm.generate...")
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
            logger.info(f"trace_id: {traceid} - Generation successful, got {len(outputs)} outputs")

        except Exception as e:
            logger.error(f"trace_id: {traceid} - Multi-modal method failed: {str(e)}")
            try:
                # Text-only fallback
                logger.warning(
                    f"trace_id: {traceid} - Falling back to text-only generation"
                )

                # For text-only fallback, use the prompt data directly
                if self.model_format == "aria":
                    # For Aria, we have token IDs, convert back to text if needed
                    if isinstance(prompt_data, list):
                        fallback_prompt = self.tokenizer.decode(prompt_data)
                    else:
                        fallback_prompt = str(prompt_data)
                else:
                    # For Qwen, we have text prompt
                    fallback_prompt = prompt_data

                outputs = self.llm.generate(
                    fallback_prompt, sampling_params=sampling_params
                )
                logger.info(f"trace_id: {traceid} - Text-only generation successful")

            except Exception as e2:
                logger.error(f"trace_id: {traceid} - All methods failed: {str(e2)}")
                raise e2

        # Extract result from output
        try:
            result = outputs[0].outputs[0].text
            logger.info(f"trace_id: {traceid} - Extracted text result: '{result[:100]}...'")
        except (AttributeError, IndexError) as e:
            logger.error(f"trace_id: {traceid} - Text extraction error: {str(e)}")
            # Try alternative extraction method
            try:
                generated_tokens = outputs[0].outputs[0].token_ids
                result = self.tokenizer.decode(generated_tokens)
                logger.info(f"trace_id: {traceid} - Decoded from tokens: '{result[:100]}...'")
            except Exception as e2:
                logger.error(f"trace_id: {traceid} - Token decoding error: {str(e2)}")
                result = str(outputs)  # Fallback to string representation

        # Clean up the result
        original_result = result
        result = (
            result.replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .replace("###", "")
            .strip()
        )

        logger.info(f"trace_id: {traceid} - After cleanup: '{result[:100]}...'")

        # find the first occurrence of < and remove everything after it
        if "<" in result:
            result = result.split("<")[0].strip()
            logger.info(f"trace_id: {traceid} - After < split: '{result[:100]}...'")

        if "AUDIO TRANSCRIPTION" in result:
            result = result.split("AUDIO TRANSCRIPTION")[0].strip()
            logger.info(f"trace_id: {traceid} - After AUDIO split: '{result[:100]}...'")

        if "QUESTION" in result:
            result = result.split("QUESTION")[0].strip()
            logger.info(f"trace_id: {traceid} - After QUESTION split: '{result[:100]}...'")

        logger.info(f"trace_id: {traceid} - Final result: '{result}'")
        
        # Check if result is empty or too short
        if not result or len(result.strip()) < 2:
            logger.warning(f"trace_id: {traceid} - Result is empty or too short, original was: '{original_result[:200]}...'")

        return result

    def infer(
        self, video_path: str, question_or_data: Any, logger, traceid: str
    ) -> str:
        """
        Enhanced inference with video context generation for VLLMModel.
        """
        try:
            # Check if enhanced prompting is enabled
            enhanced_enabled = self.config.get("enhanced_prompting", {}).get(
                "enabled", False
            )
            video_context_gen = self.config.get("enhanced_prompting", {}).get(
                "video_context_generation", True
            )

            logger.info(
                f"trace_id: {traceid} - VLLM Enhanced prompting enabled: {enhanced_enabled}"
            )

            if not enhanced_enabled:
                # Use standard inference (call parent method)
                logger.info(f"trace_id: {traceid} - Using standard VLLM inference")
                return super().infer(video_path, question_or_data, logger, traceid)

            logger.info(
                f"trace_id: {traceid} - Using enhanced VLLM inference with video context generation"
            )

            # Step 1: Extract frames and prepare basic data (without creating prompt yet)
            logger.info(
                f"trace_id: {traceid} - Starting inference with model {self.model_name} on video {video_path}"
            )

            # Extract frames first
            frames = self._extract_frames(video_path, logger, traceid)
            
            # Prepare basic data structure without calling prompt handler yet
            if isinstance(question_or_data, str):
                # Legacy format - just a question string
                question = question_or_data
                basic_data = {"video_path": video_path, "video_exists": True}
            else:
                # New format - full data dictionary
                basic_data = question_or_data
                question = basic_data.get("question", "")
                # Add video path if not present
                if "video_path" not in basic_data:
                    basic_data["video_path"] = video_path

            # Create basic inference data structure for context generation
            inference_data = {
                "question": question,
                "prompt": "",  # Will be set later
                "data": basic_data,
                "frames": frames,
                "video_path": video_path
            }

            video_context = ""
            if video_context_gen:
                # Step 2: Generate video context description
                logger.info(
                    f"trace_id: {traceid} - Generating video context description"
                )

                try:
                    # Create context generation prompt
                    context_prompt = create_video_context_prompt()
                    logger.info(f"trace_id: {traceid} - Context generation prompt: {context_prompt[:100]}...")

                    # Create temporary inference data for context generation
                    context_inference_data = inference_data.copy()
                    context_inference_data["prompt"] = context_prompt
                    context_inference_data["question"] = "Describe this video"
                    
                    # Use different sampling parameters for context generation
                    context_sampling_config = {
                        "max_tokens": 150,  # More tokens for context description
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "stop": ["\n\n", "---", "###"]
                    }
                    context_inference_data["custom_sampling"] = context_sampling_config
                    logger.info(f"trace_id: {traceid} - Using custom sampling for context: max_tokens=150")

                    # Generate video context using VLLM
                    logger.info(f"trace_id: {traceid} - Starting video context generation...")
                    video_context = self._run_model_inference(
                        context_inference_data, logger, f"{traceid}_context"
                    )
                    logger.info(f"trace_id: {traceid} - Raw video context result: '{video_context}'")
                    
                    # Clean and validate video context
                    if video_context:
                        video_context = video_context.strip()
                        logger.info(f"trace_id: {traceid} - After stripping: '{video_context}' (length: {len(video_context)})")
                        
                        # More lenient validation - just check it's not empty and not an obvious error
                        if len(video_context) < 5:
                            logger.warning(f"trace_id: {traceid} - Video context too short: '{video_context}'")
                            video_context = ""
                        elif any(error_word in video_context.lower() for error_word in ['error', 'failed', 'cannot', 'unable']):
                            logger.warning(f"trace_id: {traceid} - Video context contains error indicators: {video_context}")
                            video_context = ""
                        else:
                            logger.info(f"trace_id: {traceid} - Video context validation passed")
                    else:
                        logger.warning(f"trace_id: {traceid} - Video context is None or empty")
                    
                    if not video_context:
                        logger.error(f"trace_id: {traceid} - No valid video context generated, using fallback")
                        video_context = "Video content will be analyzed during processing."
                    else:
                        logger.success(f"trace_id: {traceid} - Successfully generated video context: {video_context[:100]}...")

                except Exception as e:
                    logger.error(f"trace_id: {traceid} - Error generating video context: {str(e)}")
                    logger.exception(e)
                    video_context = "Video content will be analyzed during processing."

            # Step 3: Add video context to the data for final inference
            enhanced_data = basic_data.copy()
            enhanced_data["video_context"] = video_context

            logger.info(
                f"trace_id: {traceid} - video_context in enhanced_data: {'video_context' in enhanced_data}"
            )
            logger.info(
                f"trace_id: {traceid} - Enhanced data keys: {list(enhanced_data.keys())}"
            )
            logger.info(
                f"trace_id: {traceid} - Video context being passed: '{video_context[:50] if video_context else 'EMPTY'}...'"
            )

            # Step 4: Create enhanced prompt directly using the prompt handler
            # Don't call _prepare_inference_data again as it may lose the video_context
            logger.info(f"trace_id: {traceid} - Calling prompt handler with enhanced_data containing video_context")
            question, enhanced_prompt, final_data = self.prompt_handler.process_question_data(
                question_or_data=enhanced_data,
                video_path=video_path,
                model_format=self.model_format,
                logger=logger,
                traceid=f"{traceid}_final",
            )
            
            # Verify the video context made it through
            final_video_context = final_data.get("video_context", "NOT_FOUND")
            logger.info(f"trace_id: {traceid} - Final data video_context: '{final_video_context[:50] if final_video_context != 'NOT_FOUND' else final_video_context}...'")

            # Step 5: Create final inference data with enhanced prompt
            final_inference_data = inference_data.copy()
            final_inference_data["prompt"] = enhanced_prompt
            final_inference_data["question"] = question
            final_inference_data["data"] = final_data

            # Step 6: Run final inference with enhanced prompt
            logger.info(
                f"trace_id: {traceid} - Running final inference with video context"
            )
            result = self._run_model_inference(final_inference_data, logger, traceid)

            logger.info(
                f"trace_id: {traceid} - Enhanced VLLM inference completed successfully"
            )
            return result

        except Exception as e:
            error_msg = f"Enhanced VLLM inference failed: {str(e)}"
            logger.error(f"trace_id: {traceid} - {error_msg}")
            logger.exception(e)
            # Fallback to standard inference
            logger.info(f"trace_id: {traceid} - Falling back to standard inference")
            return super().infer(video_path, question_or_data, logger, traceid)
