"""Simplified Transformers Model for Multiple Choice VQA."""

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    print(
        "Warning: Qwen2_5_VLForConditionalGeneration not available. Qwen2-VL models will not work properly."
    )
    pass

try:
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModel,
    )
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    import traceback

    traceback.print_exc()
    print("Warning: Transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

    # Create dummy classes for testing
    class AutoProcessor:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    class AutoModelForCausalLM:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    class AutoTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    class AutoModel:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    class Qwen2_5_VLForConditionalGeneration:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    import torch

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# InternVL3.5 specific imports and functions
try:
    import math
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    INTERNVL_AVAILABLE = True

    def build_transform(input_size):
        """Build transform for InternVL3.5"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    ):
        """Find closest aspect ratio for InternVL3.5"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        """Dynamic preprocessing for InternVL3.5"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_file, input_size=448, max_num=12):
        """Load and preprocess image for InternVL3.5"""
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        else:
            image = image_file
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

except ImportError:
    INTERNVL_AVAILABLE = False

    def load_image(*args, **kwargs):
        return None


from typing import Dict, Any, List
import cv2
import numpy as np
from PIL import Image


class TransformersModel:
    """Simplified Transformers model for multiple choice inference."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.model_format = config.get("model_format", "qwen")

        if not TRANSFORMERS_AVAILABLE:
            print(
                f"Warning: Transformers not available. Model {self.model_name} will not work properly."
            )
            self.model = None
            self.processor = None
            return

        # Initialize device
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize Transformers model
        transformers_config = config.get("transformers_config", {})

        # Check model type
        self.is_qwen2_vl = (
            "qwen2-vl" in self.model_name.lower()
            or "qwen2.5-vl" in self.model_name.lower()
        )
        self.is_internvl = (
            "internvl" in self.model_name.lower() or self.model_format == "internvl"
        )

        print(
            f"Initializing TransformersModel with model_name: {self.model_name}, is_qwen2_vl: {self.is_qwen2_vl}, is_internvl: {self.is_internvl}"
        )

        if self.is_internvl:
            # InternVL3.5 specific initialization
            print("Initializing InternVL3.5 model...")

            if not INTERNVL_AVAILABLE:
                print(
                    "Warning: InternVL dependencies not available. Install torchvision."
                )
                self.model = None
                self.tokenizer = None
                return

            model_kwargs = {
                "torch_dtype": getattr(
                    torch, transformers_config.get("torch_dtype", "bfloat16")
                ),
                "device_map": transformers_config.get("device_map", "auto"),
                "trust_remote_code": transformers_config.get("trust_remote_code", True),
                "low_cpu_mem_usage": transformers_config.get("low_cpu_mem_usage", True),
                "use_flash_attn": transformers_config.get("use_flash_attn", True),
                "load_in_8bit": transformers_config.get("load_in_8bit", False),
            }

            # For InternVL3, try AutoModelForCausalLM first, then fallback to AutoModel
            print(f"Loading InternVL3 model from: {self.model_name}")
            try:
                # Try AutoModelForCausalLM first (more common for chat models)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                ).eval()
                print(f"Model loaded with AutoModelForCausalLM. Type: {type(self.model)}")
            except Exception as e:
                print(f"AutoModelForCausalLM failed: {e}")
                print("Trying AutoModel...")
                self.model = AutoModel.from_pretrained(
                    self.model_name, **model_kwargs
                ).eval()
                print(f"Model loaded with AutoModel. Type: {type(self.model)}")

            print(f"Model has chat method: {hasattr(self.model, 'chat')}")
            print(f"Model has generate method: {hasattr(self.model, 'generate')}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )

            # InternVL3.5 specific parameters
            self.max_num_tiles = transformers_config.get("max_num_tiles", 12)
            self.image_size = transformers_config.get("image_size", 448)
            self.processor = None  # InternVL doesn't use processor
            
            # System message support
            self.system_message = config.get("system_message", "")
            print("image_size:", self.image_size)
            if self.system_message:
                print(f"System message set: {self.system_message[:100]}...")  # Preview first 100 chars

        elif self.is_qwen2_vl:
            # Qwen2-VL specific initialization
            print("Initializing Qwen2-VL model...")

            model_kwargs = {
                "torch_dtype": transformers_config.get("torch_dtype", "auto"),
                "device_map": transformers_config.get("device_map", "auto"),
                "trust_remote_code": transformers_config.get("trust_remote_code", True),
            }

            # Add attention implementation if specified
            if "attn_implementation" in transformers_config:
                model_kwargs["attn_implementation"] = transformers_config[
                    "attn_implementation"
                ]

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )

            processor_kwargs = {
                "trust_remote_code": True,
            }

            # Handle min_pixels and max_pixels for Qwen2-VL
            if "min_pixels" in transformers_config:
                processor_kwargs["min_pixels"] = transformers_config["min_pixels"]
            if "max_pixels" in transformers_config:
                processor_kwargs["max_pixels"] = transformers_config["max_pixels"]

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, **processor_kwargs
            )
        else:
            # Standard model initialization
            model_kwargs = {
                "torch_dtype": transformers_config.get("torch_dtype", "auto"),
                "device_map": transformers_config.get("device_map", "auto"),
                "trust_remote_code": transformers_config.get("trust_remote_code", True),
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

    def extract_frames(
        self, video_path: str, max_frames: int = 32
    ) -> List[Image.Image]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices to extract uniformly
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()
        return frames

    def create_prompt(self, question: str, options: List[str]) -> str:
        """Create multiple choice prompt (backward compatibility)."""
        approach = self.config.get("mc_config", {}).get("approach", "ranking")

        if approach == "iterative_scoring":
            # This shouldn't be called for iterative scoring, but provide fallback
            return self.create_ranking_prompt(question, options)
        else:
            return self.create_ranking_prompt(question, options)

    def create_ranking_prompt(
        self, question: str, options: List[str], recognized_text: str = ""
    ) -> str:
        """Create ranking prompt for original approach."""
        option_labels = ["A", "B", "C", "D"]

        template = self.config.get("mc_config", {}).get(
            "ranking_prompt_template",
            self.config.get("mc_config", {}).get(
                "prompt_template",
                "Watch the video and answer the following multiple choice question.\n\nQuestion: {question}\nAudio transcription: \"{recognized_text}\"\n\nOptions:\n{options}\n\nRank all options from most likely correct to least likely correct. Provide your ranking as letters separated by spaces (e.g., 'B A D C'):",
            ),
        )

        options_text = ""
        for i, option in enumerate(options):
            if i < len(option_labels):
                options_text += f"{option_labels[i]}. {option}\n"

        return template.format(
            question=question,
            recognized_text=recognized_text,
            options=options_text.strip(),
            option_a=options[0] if len(options) > 0 else "",
            option_b=options[1] if len(options) > 1 else "",
            option_c=options[2] if len(options) > 2 else "",
            option_d=options[3] if len(options) > 3 else "",
        )

    def create_scoring_prompt(
        self, question: str, option_text: str, recognized_text: str = ""
    ) -> str:
        """Create scoring prompt for iterative scoring approach."""
        template = self.config.get("mc_config", {}).get(
            "scoring_prompt_template",
            'You are given a video and a multiple-choice question about it.\nYour task: watch the video carefully and score how relevant and correct this specific option is to answering the question.\n\nQuestion: {question}\nAudio transcription: "{recognized_text}"\n\nOption to evaluate: {option_text}\n\nProvide a confidence score from 0.0 to 1.0, where:\n- 1.0 means you are very confident this is the correct answer\n- 0.5 means you are neutral/uncertain about this option\n- 0.0 means you are very confident this is incorrect\n\nBased on the video content, provide only a numerical score (e.g., 0.7):',
        )

        return template.format(
            question=question, option_text=option_text, recognized_text=recognized_text
        )

    def create_filter_prompt(
        self, question: str, options: List[str], recognized_text: str = ""
    ) -> str:
        """Create filter prompt for filter-then-rank approach."""
        template = self.config.get("mc_config", {}).get(
            "filter_prompt_template",
            'You are given a video and a multiple-choice question about it.\nYour task: watch the video carefully and identify which options could potentially be correct answers.\n\nQuestion: {question}\nAudio transcription: "{recognized_text}"\nOptions:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\nBased on what you see in the video, list the letters of options that could be possible answers.\nPossible options:',
        )

        return template.format(
            question=question,
            recognized_text=recognized_text,
            option_a=options[0] if len(options) > 0 else "",
            option_b=options[1] if len(options) > 1 else "",
            option_c=options[2] if len(options) > 2 else "",
            option_d=options[3] if len(options) > 3 else "",
        )

    def create_rank_filtered_prompt(
        self, question: str, filtered_options_text: str, recognized_text: str = ""
    ) -> str:
        """Create ranking prompt for filtered options."""
        template = self.config.get("mc_config", {}).get(
            "rank_filtered_prompt_template",
            'You are given a video and a multiple-choice question about it.\nYour task: rank the filtered options from most likely correct to least likely correct.\n\nQuestion: {question}\nAudio transcription: "{recognized_text}"\n\nFiltered options to rank:\n{filtered_options_text}\n\nRank these options from most likely correct to least likely correct. Provide your ranking as letters separated by spaces.\n\nRanking:',
        )

        return template.format(
            question=question,
            filtered_options_text=filtered_options_text,
            recognized_text=recognized_text,
        )

    def create_messages(self, prompt: str, frames: List[Image.Image]) -> List[Dict]:
        """Create messages format for the model."""
        if self.is_qwen2_vl:
            # Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ]
                    + [{"type": "image", "image": frame} for frame in frames],
                }
            ]
        else:
            # Standard format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ]
                    + [{"type": "image"} for _ in frames],
                }
            ]

        return messages

    def generate(self, prompt: str, frames: List[Image.Image], max_new_tokens: int = None) -> str:
        """Generate response for given prompt and frames."""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            print("Warning: Transformers not available or model not initialized.")
            return "A"  # Default response for testing

        try:
            if self.is_internvl:
                # InternVL3.5 specific processing
                if (
                    not INTERNVL_AVAILABLE
                    or self.model is None
                    or self.tokenizer is None
                ):
                    print("Warning: InternVL model not available or not initialized.")
                    return "A"

                # Verify we have the correct model type
                if not hasattr(self.model, 'chat'):
                    print(f"Error: Model type {type(self.model)} does not have 'chat' method.")
                    print("This suggests the model was loaded as the wrong class.")
                    print("Attempting to use generate method instead...")
                    
                    # Fallback to standard generation if chat method is not available
                    return self._generate_with_standard_method(prompt, frames, max_new_tokens)

                # Concatenate all frames into a single tensor for InternVL
                if frames:
                    # Use the load_image function for each frame and concatenate
                    pixel_values_list = []
                    for frame in frames:
                        pixel_values = load_image(
                            frame,
                            input_size=self.image_size,
                            max_num=self.max_num_tiles,
                        )
                        if pixel_values is not None:
                            pixel_values_list.append(pixel_values)

                    if pixel_values_list:
                        # Concatenate all frames
                        pixel_values = torch.cat(pixel_values_list, dim=0).to(
                            self.model.device, dtype=self.model.dtype
                        )
                    else:
                        pixel_values = None
                else:
                    pixel_values = None

                # Generation parameters
                generation_config = self.config.get("generation_config", {})
                generation_kwargs = {
                    "max_new_tokens": generation_config.get("max_new_tokens", max_new_tokens) if max_new_tokens is None else max_new_tokens,
                    "do_sample": generation_config.get("do_sample", False),
                }

                # Add temperature if do_sample is True
                if generation_kwargs["do_sample"]:
                    generation_kwargs["temperature"] = generation_config.get(
                        "temperature", 0.7
                    )

                print(f"Generation kwargs: {generation_kwargs}")
                
                # Apply system message if set
                if self.system_message:
                    formatted_prompt = f"{self.system_message}\n\n{prompt}"
                    print(f"Applied system message. Original prompt length: {len(prompt)}, formatted prompt length: {len(formatted_prompt)}")
                else:
                    formatted_prompt = prompt
                
                # Generate response using InternVL chat interface
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    formatted_prompt,
                    generation_kwargs,
                    history=None,
                    return_history=False,
                )

                print(f"InternVL3.5 response: {response}")
                
                # Extract final answer if using thinking format
                if self.system_message and "<think>" in str(response):
                    extracted_answer = self._extract_final_answer(response)
                    print(f"Extracted final answer: {extracted_answer}")
                    return extracted_answer if extracted_answer else response
                
                return response if isinstance(response, str) else ""
            else:
                # For Qwen2-VL and other models, create messages first
                messages = self.create_messages(prompt, frames)

                if self.is_qwen2_vl and process_vision_info is not None:
                    # Qwen2-VL specific processing
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.model.device)

                    # Generation parameters
                    generation_config = self.config.get("generation_config", {})
                    max_new_tokens = generation_config.get("max_new_tokens", max_new_tokens) if max_new_tokens is None else max_new_tokens

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, max_new_tokens=max_new_tokens, do_sample=False
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    print(output_text)
                    return output_text[0] if output_text else ""
                else:
                    # Standard processing (fallback)
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    inputs = self.processor(
                        text=[text], images=frames, return_tensors="pt"
                    )
                    inputs = inputs.to(self.model.device)

                    # Generation parameters
                    generation_config = self.config.get("generation_config", {})
                    max_new_tokens = generation_config.get("max_new_tokens", max_new_tokens) if max_new_tokens is None else max_new_tokens

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, max_new_tokens=max_new_tokens, do_sample=False
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True
                    )
                    print(output_text)

                    return output_text[0] if output_text else ""

        except Exception as e:
            print(f"Error in generation: {e}")
            return ""

    def _generate_with_standard_method(self, prompt: str, frames: List[Image.Image], max_new_tokens: int = None) -> str:
        """Fallback generation method for models without chat interface."""
        try:
            # Apply system message if set
            if self.system_message:
                formatted_prompt = f"{self.system_message}\n\n{prompt}"
            else:
                formatted_prompt = prompt

            # For models without chat method, try using standard generate
            if hasattr(self.model, 'generate') and self.tokenizer:
                # Tokenize the prompt
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                
                # Generation parameters
                generation_config = self.config.get("generation_config", {})
                generation_kwargs = {
                    "max_new_tokens": generation_config.get("max_new_tokens", max_new_tokens) if max_new_tokens is None else max_new_tokens,
                    "do_sample": generation_config.get("do_sample", False),
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                if generation_kwargs["do_sample"]:
                    generation_kwargs["temperature"] = generation_config.get("temperature", 0.7)
                
                print(f"Using standard generation with kwargs: {generation_kwargs}")
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the input prompt from the response
                if formatted_prompt in generated_text:
                    response = generated_text.replace(formatted_prompt, "").strip()
                else:
                    response = generated_text.strip()
                
                print(f"Standard generation response: {response}")
                
                # Extract final answer if using thinking format
                if self.system_message and "<think>" in str(response):
                    extracted_answer = self._extract_final_answer(response)
                    print(f"Extracted final answer: {extracted_answer}")
                    return extracted_answer if extracted_answer else response
                
                return response
            else:
                print("Error: Model does not have generate method either.")
                return "A"
                
        except Exception as e:
            print(f"Error in standard generation fallback: {e}")
            return "A"

    def infer_multiple_choice(
        self,
        video_path: str,
        question: str,
        options: List[str],
        recognized_text: str = "",
    ) -> str:
        """Run inference for multiple choice question."""
        # Extract frames
        max_frames = self.config.get("mc_config", {}).get("max_frames", 32)
        frames = self.extract_frames(video_path, max_frames)

        # Get approach from config
        approach = self.config.get("mc_config", {}).get("approach", "ranking")

        if approach == "iterative_scoring":
            return self._infer_iterative_scoring(
                question, options, frames, recognized_text
            )
        elif approach == "filter_then_rank":
            return self._infer_filter_then_rank(
                question, options, frames, recognized_text
            )
        elif approach == "context_ranking":
            return self._infer_context_ranking(
                question, options, frames, recognized_text
            )
        else:
            return self._infer_ranking(question, options, frames, recognized_text)

    def _infer_ranking(
        self, question: str, options: List[str], frames: List, recognized_text: str = ""
    ) -> str:
        """Original ranking approach."""
        # Create ranking prompt
        prompt = self.create_ranking_prompt(question, options, recognized_text)
        print(f"\n=== RANKING PROMPT ===\n{prompt}\n=== END PROMPT ===\n")
        # Generate response
        response = self.generate(prompt, frames)
        return response

    def _infer_iterative_scoring(
        self, question: str, options: List[str], frames: List, recognized_text: str = ""
    ) -> str:
        """New iterative scoring approach."""
        option_scores = []
        option_labels = ["A", "B", "C", "D"]

        for i, option in enumerate(options):
            if i >= len(option_labels):
                break

            # Create scoring prompt for this specific option
            prompt = self.create_scoring_prompt(question, option, recognized_text)
            print(
                f"\n=== SCORING PROMPT for Option {option_labels[i]} ===\n{prompt}\n=== END PROMPT ===\n"
            )

            # Generate score for this option
            response = self.generate(
                prompt, frames
            )  # Extract numerical score from response
            score = self._extract_score(response)
            option_scores.append((option_labels[i], score))

            print(f"Option {option_labels[i]}: '{option}' -> Score: {score:.3f}")

        # Sort options by score (highest first)
        option_scores.sort(key=lambda x: x[1], reverse=True)

        # Return ranking as space-separated letters
        ranking = " ".join([label for label, score in option_scores])
        return ranking

    def _infer_filter_then_rank(
        self, question: str, options: List[str], frames: List, recognized_text: str = ""
    ) -> str:
        """New filter-then-rank approach."""
        option_labels = ["A", "B", "C", "D"]

        # Step 1: Filter options
        filter_prompt = self.create_filter_prompt(question, options, recognized_text)
        print(f"\n=== FILTER PROMPT ===\n{filter_prompt}\n=== END PROMPT ===\n")
        filter_response = self.generate(filter_prompt, frames)

        print(f"Filter response: {filter_response}")

        # Extract filtered options
        filtered_letters = self._extract_filtered_options(filter_response, len(options))
        filtered_options = []
        filtered_options_text = ""

        for letter in filtered_letters:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(options):
                filtered_options.append((letter, options[idx]))
                filtered_options_text += f"{letter}. {options[idx]}\n"

        print(
            f"Filtered options: {[f'{letter}: {opt}' for letter, opt in filtered_options]}"
        )

        if not filtered_options:
            # If no options were filtered, use all options
            print("No options filtered, using all options")
            filtered_letters = option_labels[: len(options)]
            for i, option in enumerate(options):
                if i < len(option_labels):
                    filtered_options.append((option_labels[i], option))
                    filtered_options_text += f"{option_labels[i]}. {option}\n"

        # Step 2: Rank filtered options
        if len(filtered_options) > 1:
            rank_prompt = self.create_rank_filtered_prompt(
                question, filtered_options_text.strip(), recognized_text
            )
            print(f"\n=== RANKING PROMPT ===\n{rank_prompt}\n=== END PROMPT ===\n")
            rank_response = self.generate(rank_prompt, frames)

            print(f"Ranking response: {rank_response}")

            # Extract ranking from response
            ranked_letters = self._extract_ranking_from_response(
                rank_response, filtered_letters
            )
        else:
            # Only one filtered option, no need to rank
            ranked_letters = [filtered_options[0][0]] if filtered_options else []

        # Step 3: Append non-filtered options at the end
        all_letters = option_labels[: len(options)]
        final_ranking = []

        # Add ranked filtered options first
        for letter in ranked_letters:
            if letter in all_letters:
                final_ranking.append(letter)

        # Add remaining non-filtered options
        for letter in all_letters:
            if letter not in final_ranking:
                final_ranking.append(letter)

        # Ensure we have the right number of options
        final_ranking = final_ranking[: len(options)]

        print(f"Final ranking: {' '.join(final_ranking)}")
        return " ".join(final_ranking)

    def _infer_context_ranking(
        self, question: str, options: List[str], frames: List, recognized_text: str = ""
    ) -> str:
        """Context-aware ranking approach: First generate video context, then rank options."""

        # Step 1: Generate detailed video context
        context_generation_config = self.config.get("mc_config", {}).get("context_generation", {})
        context_prompt_template = context_generation_config.get("context_prompt_template", "")

        if context_prompt_template:
            context_prompt = context_prompt_template.strip()
        else:
            # Default context generation prompt
            context_prompt = """Please watch this video carefully and provide a detailed description of what you see and hear.
Focus on:
- Visual elements: objects, people, actions, scenes, settings
- Audio elements: speech, sounds, music, ambient noise
- Temporal sequence: what happens throughout the video
- Important details that might be relevant for answering questions

Provide a comprehensive and accurate description:"""

        print(f"\n=== CONTEXT GENERATION PROMPT ===\n{context_prompt}\n=== END PROMPT ===\n")

        # Generate video context using the model
        video_context = self.generate(context_prompt, frames, max_new_tokens=256).replace("\n", " ").strip()
        print(f"\n=== GENERATED VIDEO CONTEXT ===\n{video_context}\n=== END CONTEXT ===\n")

        # Step 2: Use the video context in ranking prompt
        prompt_template = self.config.get("mc_config", {}).get("prompt_template", "")

        if prompt_template:
            # Use the template from config with video context
            option_dict = {}
            option_labels = ['A', 'B', 'C', 'D']
            for i, option in enumerate(options):
                if i < len(option_labels):
                    option_dict[f'option_{option_labels[i].lower()}'] = option

            try:
                prompt = prompt_template.format(
                    question=question,
                    video_context=video_context,
                    recognized_text=recognized_text,
                    **option_dict
                )
            except KeyError as e:
                print(f"Warning: Template missing key {e}, falling back to default prompt")
                prompt = self.create_default_context_ranking_prompt(question, options, video_context, recognized_text)
        else:
            # Fallback to default context ranking prompt
            prompt = self.create_default_context_ranking_prompt(question, options, video_context, recognized_text)

        print(f"\n=== CONTEXT RANKING PROMPT ===\n{prompt}\n=== END PROMPT ===\n")

        # Generate final ranking response
        response = self.generate(prompt, frames)
        return response

    def create_default_context_ranking_prompt(
        self, question: str, options: List[str], video_context: str, recognized_text: str = ""
    ) -> str:
        """Create default context-aware ranking prompt."""
        option_labels = ['A', 'B', 'C', 'D']

        prompt = f"""You are given a video and a multiple-choice question about it.
Your task: use the detailed video context and audio information to answer the question.

For each question, rank all options from most likely correct to least likely correct. Provide your ranking as letters separated by spaces (e.g., B A D C).

Video Context: {video_context}

Audio transcription: "{recognized_text}"

Question: {question}

Options:
"""

        for i, option in enumerate(options):
            if i < len(option_labels):
                prompt += f"{option_labels[i]}. {option}\n"

        prompt += "\nAnswer:"
        return prompt

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from model response."""
        import re

        response = response.strip()
        print(f"Raw response for score extraction: '{response}'")

        # Look for "Score: X.X" pattern first
        score_pattern = re.search(r"[Ss]core:\s*([\d\.]+)", response)
        if score_pattern:
            try:
                score = float(score_pattern.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Look for decimal numbers between 0 and 1 (including edges)
        numbers = re.findall(r"\b(0?\.\d+|1\.0*|0\.0*|1)\b", response)

        if numbers:
            # Try each number found, prioritize ones that look like confidence scores
            for num_str in numbers:
                try:
                    score = float(num_str)
                    # Only accept values in reasonable range for confidence scores
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue

        # Look for any decimal number and normalize it
        all_numbers = re.findall(r"\b(\d+\.?\d*)\b", response)
        if all_numbers:
            try:
                score = float(all_numbers[0])
                # If score is > 1, assume it's out of 10 and normalize
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        print(f"Could not extract valid score from: '{response}', defaulting to 0.5")
        # If no valid score found, return neutral score
        return 0.5

    def _extract_filtered_options(self, response: str, num_options: int) -> List[str]:
        """Extract filtered option letters from response."""
        import re

        response = response.strip().upper()
        option_labels = ["A", "B", "C", "D"]
        valid_labels = option_labels[:num_options]

        # Look for letters in the response
        found_letters = re.findall(r"[ABCD]", response)

        # Keep only valid letters and preserve order, remove duplicates
        filtered_letters = []
        for letter in found_letters:
            if letter in valid_labels and letter not in filtered_letters:
                filtered_letters.append(letter)

        return filtered_letters

    def _extract_ranking_from_response(
        self, response: str, valid_letters: List[str]
    ) -> List[str]:
        """Extract ranking from response, focusing on valid letters."""
        import re

        response = response.strip().upper()

        # Look for space-separated letters pattern first
        space_pattern = re.search(r"\b([ABCD](?:\s+[ABCD])+)\b", response)
        if space_pattern:
            letters = space_pattern.group(1).split()
        else:
            # Fallback: extract all letters in order they appear
            letters = re.findall(r"[ABCD]", response)

        # Filter to only valid letters and remove duplicates
        ranked_letters = []
        for letter in letters:
            if letter in valid_letters and letter not in ranked_letters:
                ranked_letters.append(letter)

        # Add any missing valid letters at the end
        for letter in valid_letters:
            if letter not in ranked_letters:
                ranked_letters.append(letter)

        return ranked_letters

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response that includes thinking tags."""
        if not isinstance(response, str):
            return str(response)
        
        # Look for content after </think> tag
        think_end = response.find("</think>")
        if think_end != -1:
            # Get content after the closing think tag
            after_think = response[think_end + len("</think>"):].strip()
            
            # Look for "Ranking: " pattern first - this takes priority
            ranking_prefix = "Ranking: "
            ranking_start = after_think.find(ranking_prefix)
            if ranking_start != -1:
                # Extract content after "Ranking: "
                ranking_content = after_think[ranking_start + len(ranking_prefix):].strip()
                
                # Look for the ranking pattern in this content
                import re
                ranking_pattern = r'([A-D](?:\s+[A-D]){0,3})\b'
                ranking_match = re.search(ranking_pattern, ranking_content)
                
                if ranking_match:
                    return ranking_match.group(1).strip()
            
            # Look for ranking pattern anywhere in after_think (e.g., "A B C D" or "B A D C")
            import re
            ranking_pattern = r'\b([A-D](?:\s+[A-D]){0,3})\b'
            ranking_match = re.search(ranking_pattern, after_think)
            
            if ranking_match:
                return ranking_match.group(1).strip()
            
            # If no ranking pattern found, return the content after think tags
            # Remove extra whitespace and newlines
            lines = after_think.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            if non_empty_lines:
                return non_empty_lines[0]  # Return first non-empty line
        
        # Fallback: look for "Ranking: " pattern anywhere in the response
        import re
        ranking_prefix = "Ranking: "
        ranking_start = response.find(ranking_prefix)
        if ranking_start != -1:
            ranking_content = response[ranking_start + len(ranking_prefix):].strip()
            ranking_pattern = r'([A-D](?:\s+[A-D]){0,3})\b'
            ranking_match = re.search(ranking_pattern, ranking_content)
            if ranking_match:
                return ranking_match.group(1).strip()
        
        # Fallback: look for ranking pattern anywhere in the response
        ranking_pattern = r'\b([A-D](?:\s+[A-D]){0,3})\b'
        ranking_matches = re.findall(ranking_pattern, response)
        
        if ranking_matches:
            # Return the last ranking found (most likely the final answer)
            return ranking_matches[-1].strip()
        
        # If no pattern found, return original response
        return response.strip()
