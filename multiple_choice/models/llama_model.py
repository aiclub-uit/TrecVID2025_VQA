"""Llama.cpp HTTP Server Model for Multiple Choice VQA."""

import cv2
import re
import base64
import io
import math
import numpy as np
import torch
import torchvision.transforms as T
from typing import Dict, Any, List
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import requests

# InternVL3.5 Image Processing Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build image transform pipeline following InternVL3.5."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing following InternVL3.5 approach."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices for video processing."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


class LlamaCppModel:
    """Llama.cpp HTTP server model for multiple choice inference."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.model_format = config.get("model_format", "llama")
        
        # Server configuration
        server_config = config.get("server_config", {})
        self.host = server_config.get("host", "localhost")
        self.port = server_config.get("port", 8080)
        self.api_key = server_config.get("api_key", "no-key-required")
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Headers for HTTP requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Test server connection
        try:
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            if health_response.status_code == 200:
                print(f"Successfully connected to llama.cpp server at {self.base_url}")
            else:
                print(f"Warning: Server health check failed with status {health_response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to llama.cpp server at {self.base_url}: {e}")
        
        # Get model info
        try:
            models_response = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=10)
            if models_response.status_code == 200:
                models_data = models_response.json()
                if models_data.get("data"):
                    model_info = models_data["data"][0]
                    print(f"Connected to model: {model_info.get('id', 'Unknown')}")
                    
                    # Check for multimodal capabilities - look for 'multimodal' in meta or check capabilities
                    meta = model_info.get("meta", {})
                    self.supports_multimodal = meta.get("multimodal", False)
                    
                    # Also check if the server has multimodal endpoints available
                    if not self.supports_multimodal:
                        # Try to detect multimodal support by checking server capabilities
                        try:
                            props_response = requests.get(f"{self.base_url}/props", headers=self.headers, timeout=5)
                            if props_response.status_code == 200:
                                props_data = props_response.json()
                                modalities = props_data.get("modalities", {})
                                self.supports_multimodal = modalities.get("vision", False)
                        except Exception:
                            pass
                    
                    print(f"Multimodal support: {self.supports_multimodal}")
                    if meta:
                        print(f"  - Vocab size: {meta.get('n_vocab', 'unknown')}")
                        print(f"  - Context size: {meta.get('n_ctx_train', 'unknown')}")
                else:
                    self.supports_multimodal = False
            else:
                print(f"Warning: Could not get model info, status: {models_response.status_code}")
                self.supports_multimodal = False
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")
            self.supports_multimodal = False
        
        # Sampling parameters
        sampling_config = config.get("sampling_params", {})
        self.sampling_params = {
            "max_tokens": sampling_config.get("max_tokens", 2048),
            "temperature": sampling_config.get("temperature", 0.1),  # Lower for better instruction following
            "top_k": sampling_config.get("top_k", 50),  # Higher for more diverse responses
            "top_p": sampling_config.get("top_p", 0.95),  # Nucleus sampling
            "stop": sampling_config.get("stop", ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n"]),
            "stream": False,  # We'll handle streaming differently
            "cache_prompt": True,
            "repeat_penalty": 1.1,  # Prevent repetition
            "n_ctx": 32768,  # Use maximum context length
            "n_keep": -1,  # Keep entire prompt in context
        }
        
        # Image processing configuration (InternVL3.5 style)
        image_config = config.get("image_config", {})
        self.image_processing = {
            "input_size": image_config.get("input_size", 448),  # Image size for processing
            "max_num": image_config.get("max_num", 12),  # Max number of image tiles
            "use_thumbnail": image_config.get("use_thumbnail", True),  # Add thumbnail for context
            "min_num": image_config.get("min_num", 1),  # Minimum number of tiles
        }
        
        print(f"Image processing config: {self.image_processing}")

    def extract_frames(self, video_path: str, max_frames: int = 32) -> List[Image.Image]:
        """Extract frames from video using InternVL3.5 approach."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            return []

        # Use InternVL3.5 frame sampling approach
        frame_indices = get_index(
            bound=None,
            fps=fps,
            max_frame=total_frames - 1,
            first_idx=0,
            num_segments=max_frames
        )

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

    def process_image_internvl35(self, image: Image.Image, input_size: int = None, max_num: int = None) -> List[Image.Image]:
        """Process single image using InternVL3.5 dynamic preprocessing."""
        # Use instance config if parameters not provided
        if input_size is None:
            input_size = self.image_processing["input_size"]
        if max_num is None:
            max_num = self.image_processing["max_num"]
            
        # Apply dynamic preprocessing
        processed_images = dynamic_preprocess(
            image,
            min_num=self.image_processing["min_num"],
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=self.image_processing["use_thumbnail"]
        )
        return processed_images

    def process_images_batch_internvl35(self, images: List[Image.Image], input_size: int = 448, max_num: int = 12) -> List[Image.Image]:
        """Process multiple images using InternVL3.5 approach."""
        # Use instance config if parameters not provided
        if input_size is None:
            input_size = self.image_processing["input_size"]
        if max_num is None:
            max_num = self.image_processing["max_num"]
            
        all_processed_images = []
        for image in images:
            processed = self.process_image_internvl35(image, input_size, max_num)
            all_processed_images.extend(processed)
        return all_processed_images

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _get_image_token_for_model(self) -> str:
        """Get the appropriate image token for the model."""
        if "llama" in self.model_name.lower() and "vision" in self.model_name.lower():
            return "<image>"
        elif "qwen" in self.model_name.lower():
            return "<|vision_start|><|image_pad|><|vision_end|>"
        elif "internvl" in self.model_name.lower():
            return "<image>"
        elif "aria" in self.model_name.lower():
            return "<image>"
        else:
            return "<image>"

    def _format_multimodal_prompt(self, prompt: str, images: List[Image.Image]) -> Dict:
        """Format prompt for multimodal input with llama.cpp server using InternVL3.5 processing."""
        # Get system message from config
        system_message = None
        
        # Combine system message with user prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        if not images:
            return {"prompt": full_prompt}
        
        # Process images using InternVL3.5 approach
        # Use instance configuration
        processed_images = self.process_images_batch_internvl35(images)
        
        print(f"Original images: {len(images)}, Processed tiles: {len(processed_images)}")
        
        # For multimodal with llama.cpp server, we can use either:
        # 1. JSON object format with multimodal_data
        # 2. Chat completions format with image_url
        
        # Try the JSON object format first (more direct for llama.cpp)
        if len(processed_images) > 0:
            # Convert processed images to base64 (without data:image/jpeg;base64, prefix for multimodal_data)
            multimodal_data = []
            for image in processed_images:
                buffered = io.BytesIO()
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                multimodal_data.append(img_str)
            
            # Add media markers to the prompt
            # The marker <__media__> will be replaced by the server with the actual image
            media_markers = "<__media__>" * len(processed_images)
            prompt_with_media = f"{media_markers}\n{full_prompt}"
            
            return {
                "prompt": {
                    "prompt_string": prompt_with_media,
                    "multimodal_data": multimodal_data
                }
            }
        
        # Fallback: should not reach here if images exist
        return {"prompt": full_prompt}

    def _format_multimodal_prompt_chat(self, prompt: str, images: List[Image.Image]) -> Dict:
        """Alternative format using chat completions for multimodal input with InternVL3.5 processing."""
        # Get system message from config
        system_message = self.config.get("system_message", "")
        
        if not images:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            return {"messages": messages}
        
        # Process images using InternVL3.5 approach
        # Use instance configuration
        processed_images = self.process_images_batch_internvl35(images)
        
        # Chat completions format with image URLs
        image_contents = []
        for image in processed_images:
            image_b64 = self._image_to_base64(image)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": image_b64}
            })
        
        # Create messages format
        content = [{"type": "text", "text": prompt}] + image_contents
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})
        
        return {"messages": messages}

    def create_prompt(self, question: str, options: List[str]) -> str:
        """Create multiple choice prompt (backward compatibility)."""
        approach = self.config.get("mc_config", {}).get("approach", "ranking")

        if approach == "iterative_scoring":
            # For iterative scoring, this will be called per option
            return self.create_scoring_prompt(question, options[0] if options else "", "")
        else:
            return self.create_ranking_prompt(question, options, "")

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

    def generate(self, prompt: str, frames: List[Image.Image] = None) -> str:
        """Generate response for given prompt and frames."""
        frames = frames or []
        
        # If we have frames but no multimodal support detected, try anyway
        if frames and not self.supports_multimodal:
            print("Warning: Frames provided but multimodal support not detected. Trying anyway...")
        
        # Try the primary multimodal format first
        try:
            return self._try_generate_with_format(prompt, frames, use_chat_format=False)
        except Exception as e:
            print(f"Primary format failed: {e}")
            
            # Fallback to chat completions format if we have images
            if frames:
                try:
                    print("Trying fallback chat completions format...")
                    return self._try_generate_with_format(prompt, frames, use_chat_format=True)
                except Exception as e2:
                    print(f"Chat format also failed: {e2}")
            
            # Final fallback to text-only
            print("Falling back to text-only...")
            return self._try_generate_with_format(prompt, [])

    def _try_generate_with_format(self, prompt: str, frames: List[Image.Image], use_chat_format: bool = False) -> str:
        """Try to generate with a specific format."""
        try:
            # Format the request
            if use_chat_format and frames:
                request_data = self._format_multimodal_prompt_chat(prompt, frames)
            else:
                request_data = self._format_multimodal_prompt(prompt, frames)
            
            # Add sampling parameters
            request_data.update(self.sampling_params)

            # Choose endpoint based on request format
            if "messages" in request_data:
                # Use chat completions for multimodal (OpenAI-style)
                endpoint = f"{self.base_url}/v1/chat/completions"
                print(f"Using chat completions endpoint: {endpoint}")
            elif isinstance(request_data.get("prompt"), dict) and "multimodal_data" in request_data["prompt"]:
                # Use completions endpoint for JSON object format with multimodal_data
                endpoint = f"{self.base_url}/completion"  # Note: no /v1/ prefix for this endpoint
                print(f"Using completion endpoint with multimodal_data: {endpoint}")
            else:
                # Use completions for text-only
                endpoint = f"{self.base_url}/v1/completions"
                print(f"Using text completions endpoint: {endpoint}")
            
            # Make the request
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=request_data,
                timeout=120  # 2 minutes timeout
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"Raw response data: {response_data}")
                
                # Handle different response formats
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    # OpenAI-compatible format
                    choice = response_data["choices"][0]
                    
                    if "messages" in request_data:
                        # Chat completion response
                        content = choice.get("message", {}).get("content", "").strip()
                        print(f"Chat completion response: {content}")
                        return content
                    else:
                        # Text completion response
                        content = choice.get("text", choice.get("content", "")).strip()
                        print(f"Text completion response: {content}")
                        return content
                elif "content" in response_data:
                    # llama.cpp native format (direct content field)
                    content = response_data.get("content", "").strip()
                    print(f"Native llama.cpp response: {content}")
                    return content
                else:
                    print(f"Unexpected response format. Available keys: {list(response_data.keys())}")
                    # Try to extract any text-like content
                    for key in ["text", "response", "output", "result"]:
                        if key in response_data:
                            content = str(response_data[key]).strip()
                            print(f"Found content in '{key}': {content}")
                            return content
                    
                    print(f"No recognizable content found in response: {response_data}")
                    return ""
            else:
                print(f"HTTP error {response.status_code}: {response.text}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error in _try_generate_with_format: {e}")
            raise

    def infer_multiple_choice(
        self,
        video_path: str,
        question: str,
        options: List[str],
        recognized_text: str = "",
    ) -> str:
        """Run inference for multiple choice question."""
        # Extract frames for multimodal inference
        max_frames = self.config.get("mc_config", {}).get("max_frames", 32)
        frames = self.extract_frames(video_path, max_frames)

        # Get approach from config
        approach = self.config.get("mc_config", {}).get("approach", "ranking")

        if approach == "iterative_scoring":
            return self._infer_iterative_scoring(question, options, frames, recognized_text)
        elif approach == "filter_then_rank":
            return self._infer_filter_then_rank(question, options, frames, recognized_text)
        elif approach == "context_ranking":
            return self._infer_context_ranking(question, options, frames, recognized_text)
        else:
            return self._infer_ranking(question, options, frames, recognized_text)

    def _infer_ranking(
        self,
        question: str,
        options: List[str],
        frames: List[Image.Image] = None,
        recognized_text: str = "",
    ) -> str:
        """Original ranking approach."""
        # Create ranking prompt
        prompt = self.create_ranking_prompt(question, options, recognized_text)
        print(f"\n=== RANKING PROMPT ===\n{prompt}\n=== END PROMPT ===\n")
        # Generate response with frames
        response = self.generate(prompt, frames)
        return response

    def _infer_iterative_scoring(
        self,
        question: str,
        options: List[str],
        frames: List[Image.Image] = None,
        recognized_text: str = "",
    ) -> str:
        """New iterative scoring approach."""
        option_scores = []
        option_labels = ["A", "B", "C", "D"]

        for i, option in enumerate(options):
            if i >= len(option_labels):
                break
            
            # Create scoring prompt for this option
            scoring_prompt = self.create_scoring_prompt(question, option, recognized_text)
            print(f"\n=== SCORING PROMPT FOR OPTION {option_labels[i]} ===\n{scoring_prompt}\n=== END PROMPT ===\n")
            
            # Generate score
            response = self.generate(scoring_prompt, frames)
            score = self._extract_score(response)
            
            option_scores.append((option_labels[i], score))
            print(f"Option {option_labels[i]}: {option} -> Score: {score}")

        # Sort options by score (highest first)
        option_scores.sort(key=lambda x: x[1], reverse=True)

        # Return ranking as space-separated letters
        ranking = " ".join([label for label, score in option_scores])
        return ranking

    def _infer_filter_then_rank(
        self,
        question: str,
        options: List[str],
        frames: List[Image.Image] = None,
        recognized_text: str = "",
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
            if letter in option_labels:
                idx = option_labels.index(letter)
                if idx < len(options):
                    filtered_options.append((letter, options[idx]))
                    filtered_options_text += f"{letter}. {options[idx]}\n"

        print(
            f"Filtered options: {[f'{letter}: {opt}' for letter, opt in filtered_options]}"
        )

        if not filtered_options:
            # Fallback to all options if filtering failed
            print("No filtered options found, using all options")
            return self._infer_ranking(question, options, frames, recognized_text)

        # Step 2: Rank filtered options
        if len(filtered_options) > 1:
            rank_prompt = self.create_rank_filtered_prompt(question, filtered_options_text.strip(), recognized_text)
            print(f"\n=== RANK FILTERED PROMPT ===\n{rank_prompt}\n=== END PROMPT ===\n")
            rank_response = self.generate(rank_prompt, frames)
            ranked_letters = self._extract_ranking_from_response(rank_response, [letter for letter, _ in filtered_options])
        else:
            ranked_letters = [filtered_options[0][0]]

        # Step 3: Append non-filtered options at the end
        all_letters = option_labels[: len(options)]
        final_ranking = []

        # Add ranked filtered options first
        for letter in ranked_letters:
            if letter not in final_ranking:
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
        self,
        question: str,
        options: List[str],
        frames: List[Image.Image] = None,
        recognized_text: str = "",
    ) -> str:
        """Context-aware ranking approach: First generate video context, then rank options."""

        # Step 1: Generate detailed video context
        context_generation_config = self.config.get("mc_config", {}).get(
            "context_generation", {}
        )
        context_prompt_template = context_generation_config.get(
            "context_prompt_template", ""
        )

        if context_prompt_template:
            context_prompt = context_prompt_template.format(
                question=question, recognized_text=recognized_text
            )
        else:
            context_prompt = self.create_default_context_generation_prompt(
                question, recognized_text
            )

        print(
            f"\n=== CONTEXT GENERATION PROMPT ===\n{context_prompt}\n=== END PROMPT ===\n"
        )

        # Generate video context using the model
        video_context = self.generate(context_prompt, frames)
        print(
            f"\n=== GENERATED VIDEO CONTEXT ===\n{video_context}\n=== END CONTEXT ===\n"
        )

        # Step 2: Use the video context in ranking prompt
        prompt_template = self.config.get("mc_config", {}).get("prompt_template", "")

        if prompt_template:
            prompt = prompt_template.format(
                question=question,
                recognized_text=recognized_text,
                video_context=video_context,
                options="\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]),
                option_a=options[0] if len(options) > 0 else "",
                option_b=options[1] if len(options) > 1 else "",
                option_c=options[2] if len(options) > 2 else "",
                option_d=options[3] if len(options) > 3 else "",
            )
        else:
            prompt = self.create_default_context_ranking_prompt(
                question, options, video_context, recognized_text
            )

        print(f"\n=== CONTEXT RANKING PROMPT ===\n{prompt}\n=== END PROMPT ===\n")

        # Generate final ranking response
        response = self.generate(prompt, frames)
        return response

    def create_default_context_generation_prompt(
        self, question: str, recognized_text: str = ""
    ) -> str:
        """Create default context generation prompt."""
        return f"""Please watch this video carefully and provide a detailed description of what you see and hear.
        
Question context: {question}
Audio transcription: "{recognized_text}"

Describe the key visual elements, actions, objects, people, settings, and any other relevant details that might be important for answering the question. Be specific and comprehensive in your description."""

    def create_default_context_ranking_prompt(
        self,
        question: str,
        options: List[str],
        video_context: str,
        recognized_text: str = "",
    ) -> str:
        """Create default context-aware ranking prompt."""
        option_labels = ["A", "B", "C", "D"]

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
        response = response.strip()
        print(f"Raw response for score extraction: '{response}'")

        # Look for "Score: X.X" pattern first
        score_pattern = re.search(r"[Ss]core:\s*([\d\.]+)", response)
        if score_pattern:
            try:
                score = float(score_pattern.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                pass

        # Look for just a number at the start or end
        number_patterns = [
            r"^([\d\.]+)",  # Number at start
            r"([\d\.]+)$",  # Number at end
            r"\b(0\.\d+)\b",  # Decimal between 0 and 1
            r"\b(1\.0+)\b",  # 1.0
            r"\b([01])\b",  # Just 0 or 1
        ]

        for pattern in number_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # Clamp between 0 and 1
                except ValueError:
                    continue

        # Default fallback
        print(f"Could not extract score from: '{response}', defaulting to 0.5")
        return 0.5

    def _extract_filtered_options(self, response: str, num_options: int) -> List[str]:
        """Extract filtered option letters from response."""
        response = response.upper().strip()
        option_labels = ["A", "B", "C", "D"][:num_options]
        
        # Look for letters mentioned in the response
        found_letters = []
        for letter in option_labels:
            if letter in response:
                found_letters.append(letter)
        
        # If no letters found, return all as fallback
        if not found_letters:
            return option_labels
        
        return found_letters

    def _extract_ranking_from_response(
        self, response: str, valid_letters: List[str]
    ) -> List[str]:
        """Extract ranking from model response."""
        response = response.upper().strip()
        
        # Look for space-separated pattern first
        space_pattern = r"\b([A-D](?:\s+[A-D])*)\b"
        space_match = re.search(space_pattern, response)
        if space_match:
            letters = space_match.group(1).split()
            # Filter to only valid letters and remove duplicates while preserving order
            seen = set()
            filtered_letters = []
            for letter in letters:
                if letter in valid_letters and letter not in seen:
                    filtered_letters.append(letter)
                    seen.add(letter)
            if filtered_letters:
                return filtered_letters
        
        # Look for any sequence of letters
        letters_found = []
        for letter in valid_letters:
            if letter in response:
                letters_found.append(letter)
        
        if letters_found:
            return letters_found
        
        # Fallback to original order
        return valid_letters
