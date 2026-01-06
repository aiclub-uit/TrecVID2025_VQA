"""Simplified VLLM Model for Multiple Choice VQA with TTA support."""

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    import traceback

    traceback.print_exc()
    print("Warning: VLLM not available. Install with: pip install vllm")
    VLLM_AVAILABLE = False

    # Create dummy classes for testing
    class LLM:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return []

    class SamplingParams:
        def __init__(self, *args, **kwargs):
            pass


from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image
import random
from collections import Counter


class VLLMModel:
    """Simplified VLLM model for multiple choice inference with TTA support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.model_format = config.get("model_format", "aria")

        if not VLLM_AVAILABLE:
            print(
                f"Warning: VLLM not available. Model {self.model_name} will not work properly."
            )
            self.llm = None
            self.sampling_params = None
            return

        # Initialize VLLM
        vllm_config = config.get("vllm_config", {})
        
        # Special configuration for InternVL3
        if "internvl" in self.model_name.lower():
            # InternVL3 specific settings
            default_max_model_len = 8192  # InternVL3 typically supports longer sequences
            default_gpu_memory_utilization = 0.85  # Can be slightly higher for InternVL3
        else:
            default_max_model_len = 4096
            default_gpu_memory_utilization = 0.8
        
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
            max_model_len=vllm_config.get("max_model_len", default_max_model_len),
            enforce_eager=vllm_config.get("enforce_eager", True),
            trust_remote_code=vllm_config.get("trust_remote_code", True),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", default_gpu_memory_utilization),
            # Add additional parameters that might be needed for multimodal models
            disable_custom_all_reduce=vllm_config.get("disable_custom_all_reduce", False),
        )

        # Sampling parameters
        sampling_config = config.get("sampling_params", {})
        
        # Default stop tokens - add common InternVL3 stop tokens
        default_stop_tokens = ["<|im_end|>"]
        if "internvl" in self.model_name.lower():
            # InternVL3 might use different stop tokens
            default_stop_tokens.extend(["</s>", "<|endoftext|>"])
        
        self.sampling_params = SamplingParams(
            max_tokens=sampling_config.get("max_tokens", 10),
            temperature=sampling_config.get("temperature", 0.0),
            top_k=sampling_config.get("top_k", 1),
            top_p=sampling_config.get("top_p", 1.0),
            stop=sampling_config.get("stop", default_stop_tokens),
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

    def _get_image_token_for_model(self) -> str:
        """Get the appropriate image token for the model."""
        if "internvl" in self.model_name.lower():
            return "<image>"
        elif "qwen" in self.model_name.lower():
            return "<|image_pad|>"
        elif "aria" in self.model_name.lower():
            return "<image>"
        else:
            # Default fallback
            return "<image>"

    def _format_multimodal_prompt(self, prompt: str, num_images: int) -> str:
        """Format prompt with appropriate image tokens for multimodal input."""
        if num_images == 0:
            return prompt
        
        image_token = self._get_image_token_for_model()
        
        # For InternVL models, we typically put image tokens at the beginning
        if "internvl" in self.model_name.lower():
            image_tokens = image_token * num_images
            formatted_prompt = f"{image_tokens}\n{prompt}"
        else:
            # For other models, try the same format
            image_tokens = image_token * num_images
            formatted_prompt = f"{image_tokens}\n{prompt}"
        
        # Check if we need to apply chat template
        if self._should_apply_chat_template():
            return self._apply_chat_template(formatted_prompt)
        
        return formatted_prompt

    def _should_apply_chat_template(self) -> bool:
        """Check if the model requires chat template application."""
        # InternVL models often use chat templates
        if "internvl" in self.model_name.lower():
            return self.config.get("use_chat_template", True)
        return self.config.get("use_chat_template", False)

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to the prompt if needed."""
        # This is a simplified chat template application
        # In practice, you might want to use the model's tokenizer for this
        
        if "internvl" in self.model_name.lower():
            # InternVL3 chat template format
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic chat template
            return f"### Human: {prompt}\n### Assistant: "

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

    def generate(self, prompt: str, frames: List[Image.Image] = None) -> str:
        """Generate response for given prompt and frames."""
        if not VLLM_AVAILABLE or self.llm is None:
            return "A"  # Default response for testing

        if frames and len(frames) > 0:
            # Format prompt with appropriate image tokens
            multimodal_prompt = self._format_multimodal_prompt(prompt, len(frames))
            mm_data = {"image": frames}
            
            # Create multimodal input dict for vLLM
            inputs = {
                "prompt": multimodal_prompt,
                "multi_modal_data": mm_data
            }
            print(f"Multimodal inputs: original prompt length={len(prompt)}, formatted prompt length={len(multimodal_prompt)}, frames count={len(frames)}")
            print(f"Image token used: '{self._get_image_token_for_model()}'")
            outputs = self.llm.generate([inputs], self.sampling_params)
        else:
            # Text-only fallback
            outputs = self.llm.generate([prompt], self.sampling_params)

        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()

        return ""

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

        if approach == "ranking_tta":
            return self._infer_ranking_tta(
                question, options, frames, recognized_text
            )
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

    def _infer_ranking_tta(
        self,
        question: str,
        options: List[str],
        frames: List[Image.Image] = None,
        recognized_text: str = "",
    ) -> str:
        """Ranking approach with Test Time Augmentation (TTA)."""
        tta_config = self.config.get("mc_config", {}).get("tta_config", {})
        num_augmentations = tta_config.get("num_augmentations", 3)
        voting_strategy = tta_config.get("voting_strategy", "majority")
        seed = tta_config.get("seed", 42)
        
        print(f"Running TTA with {num_augmentations + 1} total inferences...")
        
        # Store all predictions and their mappings
        all_predictions = []
        all_mappings = []
        
        # Generate option permutations
        permutations = self._generate_option_permutations(options, num_augmentations, seed)
        
        for i, (permuted_options, mapping) in enumerate(permutations):
            print(f"\n--- TTA Inference {i + 1}/{len(permutations)} ---")
            print(f"Option mapping: {mapping}")
            
            # Create prompt with permuted options
            prompt = self.create_ranking_prompt(question, permuted_options, recognized_text)
            print(f"Permuted options: {[f'{chr(65 + j)}. {opt}' for j, opt in enumerate(permuted_options)]}")
            
            # Generate response
            response = self.generate(prompt, frames)
            print(f"Raw response: {response}")
            
            # Parse ranking from response
            ranking = self._parse_ranking_response(response, len(permuted_options))
            print(f"Parsed ranking: {ranking}")
            
            # Map back to original option indices
            original_ranking = self._map_ranking_to_original(ranking, mapping)
            print(f"Mapped to original: {original_ranking}")
            
            all_predictions.append(original_ranking)
            all_mappings.append(mapping)
        
        # Apply voting strategy
        if voting_strategy == "majority":
            final_ranking = self._majority_vote_ranking(all_predictions)
        else:
            # Default to majority voting
            final_ranking = self._majority_vote_ranking(all_predictions)
        
        print(f"\n--- TTA FINAL RESULT ---")
        print(f"All predictions: {all_predictions}")
        print(f"Final ranking: {final_ranking}")
        
        # Convert back to expected format (letters separated by spaces)
        result = " ".join(final_ranking)
        return result

    def _generate_option_permutations(
        self, options: List[str], num_augmentations: int, seed: int = 42
    ) -> List[Tuple[List[str], List[int]]]:
        """Generate option permutations for TTA.
        
        Returns:
            List of (permuted_options, mapping) where mapping[i] gives the 
            original index of the option now at position i.
        """
        random.seed(seed)
        permutations = []
        
        # Original order (no shuffling)
        original_mapping = list(range(len(options)))
        permutations.append((options.copy(), original_mapping))
        
        # Generate additional permutations
        for i in range(num_augmentations):
            # Create a new permutation
            indices = list(range(len(options)))
            random.shuffle(indices)
            
            # Ensure we don't duplicate the original order
            while indices == original_mapping:
                random.shuffle(indices)
            
            permuted_options = [options[idx] for idx in indices]
            permutations.append((permuted_options, indices))
        
        return permutations

    def _parse_ranking_response(self, response: str, num_options: int) -> List[str]:
        """Parse ranking letters from model response."""
        import re
        
        response = response.strip().upper()
        
        # Valid letters based on number of options
        valid_letters = [chr(65 + i) for i in range(num_options)]  # A, B, C, D...
        
        # First try to find a clear ranking pattern like "B A D C" (space-separated letters)
        space_pattern = re.search(r'\b([ABCD](?:\s+[ABCD])+)\b', response)
        if space_pattern:
            letters = space_pattern.group(1).split()
        else:
            # Fallback: extract all letters in order they appear
            letters = re.findall(r'[ABCD]', response)

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

    def _map_ranking_to_original(
        self, ranking: List[str], mapping: List[int]
    ) -> List[str]:
        """Map ranking letters back to original option positions.
        
        Args:
            ranking: Ranking in current permutation (e.g., ['B', 'A', 'C', 'D'])
            mapping: mapping[i] = original index of option at position i
            
        Returns:
            Ranking in terms of original option positions
        """
        # Convert letters to indices in current permutation
        ranking_indices = [ord(letter) - ord('A') for letter in ranking]
        
        # Map to original indices
        original_indices = [mapping[idx] for idx in ranking_indices]
        
        # Convert back to letters representing original positions
        original_ranking = [chr(65 + idx) for idx in original_indices]
        
        return original_ranking

    def _majority_vote_ranking(self, all_predictions: List[List[str]]) -> List[str]:
        """Apply majority voting to combine rankings.
        
        For each position in the ranking, count which option appears most
        frequently at that position across all predictions.
        """
        if not all_predictions:
            return ['A', 'B', 'C', 'D']
        
        num_options = len(all_predictions[0])
        final_ranking = []
        used_options = set()
        
        # For each ranking position (1st, 2nd, 3rd, 4th)
        for position in range(num_options):
            # Count votes for each option at this position
            votes = Counter()
            for prediction in all_predictions:
                if position < len(prediction):
                    option = prediction[position]
                    if option not in used_options:  # Only count if not already used
                        votes[option] += 1
            
            # Select most voted option (that hasn't been used)
            if votes:
                # Get the most common option that hasn't been used
                for option, count in votes.most_common():
                    if option not in used_options:
                        final_ranking.append(option)
                        used_options.add(option)
                        break
                else:
                    # Fallback: add any remaining option
                    all_options = set(chr(65 + i) for i in range(num_options))
                    remaining = all_options - used_options
                    if remaining:
                        final_ranking.append(remaining.pop())
            else:
                # Fallback: add remaining options in order
                all_options = [chr(65 + i) for i in range(num_options)]
                for option in all_options:
                    if option not in used_options:
                        final_ranking.append(option)
                        used_options.add(option)
                        break
        
        # Ensure we have all options
        all_options = [chr(65 + i) for i in range(num_options)]
        for option in all_options:
            if option not in final_ranking:
                final_ranking.append(option)
        
        return final_ranking[:num_options]
