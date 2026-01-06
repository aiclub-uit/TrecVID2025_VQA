import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from .enhanced_vqa import EnhancedVQAModel
from typing import Dict, Any


class TransformersModel(EnhancedVQAModel):
    """
    A wrapper for Hugging Face transformer models for image text understanding.
    """

    def __init__(self, config=None):
        # Set default model name before calling super().__init__
        if config is None:
            config = {}
        if "model_name" not in config:
            config["model_name"] = "Qwen/Qwen2.5-VL-7B-Instruct"

        super().__init__(config)

        # Transformers-specific configuration
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_class_name = self.config.get("model_class", "AutoModelForCausalLM")
        model_args = self.config.get("model_args", {})

        print(f"Using model format: {self.model_format}")

        # Check if this is VideoLLaMA3 model
        self.is_videollama3 = "videollama3" in self.model_name.lower()

        # Check if this is Qwen2.5-VL model
        self.is_qwen2_5_vl = (
            "qwen2.5-vl" in self.model_name.lower()
            or "qwen2_5-vl" in self.model_name.lower()
        )

        print(
            f"Initializing TransformersModel with model_name: {self.model_name}, "
            f"model_format: {self.model_format}, device: {self.device}, is_videollama3: {self.is_videollama3}, is qwen2_5_vl: {self.is_qwen2_5_vl}"
        )

        if self.is_videollama3:
            # VideoLLaMA3-specific initialization
            print("Initializing VideoLLaMA3 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=self.config.get(
                    "attn_implementation", "flash_attention_2"
                ),
                cache_dir=self.cache_dir,
                **model_args,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        elif self.is_qwen2_5_vl:
            # Qwen2.5-VL specific initialization
            print("Initializing Qwen2.5-VL model...")
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError(
                    "Qwen2_5_VLForConditionalGeneration not available. Please install the required transformers version."
                )

            # Prepare model loading arguments with fallbacks for compatibility
            model_load_args = {
                "torch_dtype": self.config.get("torch_dtype", torch.bfloat16),
                "device_map": self.config.get("device_map", "auto"),
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                **model_args,
            }

            # Handle attention implementation - fallback if flash_attention_2 causes issues
            attn_impl = self.config.get("attn_implementation", "flash_attention_2")
            model_load_args["attn_implementation"] = attn_impl
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_load_args
            )
            print(f"Successfully loaded with attention implementation: {attn_impl}")

            # Handle min_pixels and max_pixels for Qwen2.5-VL
            processor_args = {"trust_remote_code": True}
            if "min_pixels" in self.config:
                processor_args["min_pixels"] = self.config["min_pixels"]
            if "max_pixels" in self.config:
                processor_args["max_pixels"] = self.config["max_pixels"]

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, **processor_args
            )
        else:
            # Standard model initialization
            # Import and initialize model class
            model_class = globals().get(model_class_name)
            if not model_class:
                try:
                    # Try to import from transformers library
                    from transformers import __dict__ as transformers_dict

                    model_class = transformers_dict.get(model_class_name)
                except ImportError:
                    model_class = None

            if not model_class:
                raise ValueError(f"Model class {model_class_name} not found.")

            self.model = model_class.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, **model_args
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True, cache_dir=self.cache_dir
            )

    def _create_videollama3_prompt(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ):
        """
        Create prompt format for VideoLLaMA3 model.

        Args:
            inference_data: Prepared inference data containing frames and video path
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Conversation format for VideoLLaMA3
        """
        video_path = inference_data["video_path"]
        prompt = inference_data["prompt"]

        logger.info(
            f"trace_id: {traceid} - Creating VideoLLaMA3 conversation for video: {video_path}"
        )

        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": self.config.get("fps", 8),
                            "max_frames": self.config.get("max_frames", 180),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        return conversation

    def _create_qwen2_5_vl_prompt(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ):
        """
        Create prompt format for Qwen2.5-VL model.

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Messages format and processed vision info for Qwen2.5-VL
        """
        frames = inference_data["frames"]
        prompt = inference_data["prompt"]

        logger.info(
            f"trace_id: {traceid} - Creating Qwen2.5-VL conversation with {len(frames)} frames"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [{"type": "image", "image": frame} for frame in frames],
            }
        ]

        return messages

    def _create_standard_prompt(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ):
        """
        Create prompt format for standard models (Qwen, etc.).

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Messages format for standard models
        """
        frames = inference_data["frames"]
        prompt = inference_data["prompt"]

        logger.info(
            f"trace_id: {traceid} - Creating standard conversation with {len(frames)} frames"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [{"type": "image"} for _ in frames],
            }
        ]

        return messages, frames

    def _run_model_inference(
        self, inference_data: Dict[str, Any], logger, traceid: str
    ) -> str:
        """
        Run transformers-specific inference with support for VideoLLaMA3 and other models.

        Args:
            inference_data: Prepared inference data containing frames and prompt
            logger: Logger instance
            traceid: Trace ID for logging

        Returns:
            Model output string
        """
        logger.info(
            f"trace_id: {traceid} - Starting inference with model: {self.model_name}"
        )
        logger.info(f"trace_id: {traceid} - Model format: {self.model_format}")

        try:
            if self.is_videollama3:
                # VideoLLaMA3-specific inference
                logger.info(
                    f"trace_id: {traceid} - Using VideoLLaMA3 inference pathway"
                )

                conversation = self._create_videollama3_prompt(
                    inference_data, logger, traceid
                )

                logger.info(
                    f"trace_id: {traceid} - Processing conversation with processor"
                )
                inputs = self.processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                # Move inputs to device and convert to appropriate dtype
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                logger.info(
                    f"trace_id: {traceid} - Generating response with VideoLLaMA3"
                )

                # Get generation parameters from config
                max_new_tokens = self.config.get("max_new_tokens", 512)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )

                response = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()

                logger.info(f"trace_id: {traceid} - VideoLLaMA3 inference complete")
                return response

            elif self.is_qwen2_5_vl:
                # Qwen2.5-VL specific inference
                logger.info(f"trace_id: {traceid} - Using Qwen2.5-VL inference pathway")

                try:
                    messages = self._create_qwen2_5_vl_prompt(
                        inference_data, logger, traceid
                    )

                    logger.info(
                        f"trace_id: {traceid} - Applying chat template for Qwen2.5-VL"
                    )
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    logger.info(f"trace_id: {traceid} - Processing vision info")
                    if process_vision_info is None:
                        raise ImportError(
                            "qwen_vl_utils.process_vision_info not available. Please install qwen-vl-utils."
                        )

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Ensure all inputs are on the correct device and have compatible dtypes
                    logger.info(
                        f"trace_id: {traceid} - Moving inputs to device: {self.model.device}"
                    )

                    # Move inputs to model device with proper error handling
                    try:
                        inputs = inputs.to(self.model.device)

                        # Ensure pixel_values are in the correct dtype if they exist
                        if (
                            "pixel_values" in inputs
                            and inputs["pixel_values"] is not None
                        ):
                            # Use the same dtype as the model
                            model_dtype = next(self.model.parameters()).dtype
                            if model_dtype == torch.float32:
                                inputs["pixel_values"] = inputs["pixel_values"].to(
                                    torch.float32
                                )
                            else:
                                inputs["pixel_values"] = inputs["pixel_values"].to(
                                    torch.bfloat16
                                )
                            logger.info(
                                f"trace_id: {traceid} - Set pixel_values dtype to: {inputs['pixel_values'].dtype}"
                            )

                    except RuntimeError as device_error:
                        logger.warning(
                            f"trace_id: {traceid} - Device placement error: {device_error}"
                        )
                        # If there are device issues, try CPU fallback
                        if torch.cuda.is_available():
                            logger.info(
                                f"trace_id: {traceid} - Trying CPU fallback for inputs"
                            )
                            inputs = {
                                k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()
                            }
                            # Also move model to CPU if necessary
                            if str(self.model.device) != "cpu":
                                logger.info(
                                    f"trace_id: {traceid} - Moving model to CPU due to device issues"
                                )
                                self.model = self.model.cpu()
                        else:
                            raise device_error

                    logger.info(
                        f"trace_id: {traceid} - Generating response with Qwen2.5-VL"
                    )

                    # Get generation parameters from config
                    max_new_tokens = self.config.get("max_new_tokens", 512)

                    # Add error handling for generation
                    try:
                        logger.info(
                            f"trace_id: {traceid} - Starting generation with max_new_tokens: {max_new_tokens}"
                        )

                        # Use torch.no_grad() to save memory and avoid gradient computation
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs, max_new_tokens=max_new_tokens
                            )

                    except (AttributeError, RuntimeError) as generation_error:
                        error_str = str(generation_error)
                        logger.warning(
                            f"trace_id: {traceid} - Generation error: {error_str}"
                        )

                        if "is_causal" in error_str:
                            logger.info(
                                f"trace_id: {traceid} - Trying with explicit generation config for is_causal error"
                            )
                            with torch.no_grad():
                                generated_ids = self.model.generate(
                                    **inputs,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                    pad_token_id=self.processor.tokenizer.eos_token_id,
                                )
                        elif "Triton" in error_str or "cpu tensor" in error_str:
                            logger.info(
                                f"trace_id: {traceid} - Triton/device error, trying CPU generation"
                            )
                            # Move everything to CPU for generation
                            inputs_cpu = {
                                k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()
                            }
                            model_device = str(self.model.device)

                            # Temporarily move model to CPU
                            if model_device != "cpu":
                                self.model = self.model.cpu()

                            with torch.no_grad():
                                generated_ids = self.model.generate(
                                    **inputs_cpu,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                )

                            # Move model back to original device if it was on GPU
                            if model_device != "cpu" and torch.cuda.is_available():
                                self.model = self.model.to(model_device)
                                # Move generated_ids back to original device
                                generated_ids = generated_ids.to(model_device)
                        else:
                            raise generation_error

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    logger.info(f"trace_id: {traceid} - Qwen2.5-VL inference complete")
                    return output_text[0] if output_text else ""

                except Exception as qwen_error:
                    logger.error(
                        f"trace_id: {traceid} - Qwen2.5-VL specific error: {str(qwen_error)}"
                    )
                    logger.info(
                        f"trace_id: {traceid} - Falling back to standard inference pathway"
                    )
                    # Fall back to standard inference if Qwen2.5-VL specific path fails
                    messages, frames = self._create_standard_prompt(
                        inference_data, logger, traceid
                    )

                    logger.info(
                        f"trace_id: {traceid} - Applying chat template (fallback)"
                    )
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    inputs = self.processor(
                        text=[text], images=frames, return_tensors="pt"
                    )
                    inputs = inputs.to(self.model.device)

                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True
                    )

                    logger.info(f"trace_id: {traceid} - Fallback inference complete")
                    return output_text[0] if output_text else ""

            else:
                # Standard model inference (Qwen, etc.)
                logger.info(f"trace_id: {traceid} - Using standard inference pathway")

                messages, frames = self._create_standard_prompt(
                    inference_data, logger, traceid
                )

                logger.info(f"trace_id: {traceid} - Applying chat template")
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                logger.info(
                    f"trace_id: {traceid} - Text processed for model input: {text[:200]}..."
                )

                inputs = self.processor(text=[text], images=frames, return_tensors="pt")
                inputs = inputs.to(self.model.device)

                logger.info(
                    f"trace_id: {traceid} - Generating output with standard model"
                )

                # Get generation parameters from config
                max_new_tokens = self.config.get("max_new_tokens", 512)

                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                generated_ids = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                logger.info(
                    f"trace_id: {traceid} - Generated IDs: {generated_ids[:5] if generated_ids else 'None'}..."
                )

                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                logger.info(f"trace_id: {traceid} - Standard inference complete")
                return output_text

        except Exception as e:
            logger.error(f"trace_id: {traceid} - Inference failed: {str(e)}")
            logger.exception(f"trace_id: {traceid} - Detailed error traceback")
            raise e
