"""
Prepare data for VQA inference by processing metadata and video folders.
Generates JSON files containing video paths, questions, and answers for evaluation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torchaudio
import moviepy.editor as mp
from transformers import AutoProcessor, SeamlessM4Tv2Model
import whisper


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_metadata_file(metadata_path: Path) -> Optional[Dict]:
    """Load and parse a metadata JSON file."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Failed to load metadata file {metadata_path}: {e}")
        return None


def find_video_file(video_folder: Path, stem: str) -> Optional[Path]:
    """Find corresponding video file for a given stem."""
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    for ext in video_extensions:
        video_path = video_folder / f"{stem}{ext}"
        if video_path.exists():
            return video_path

    return None


def create_prepared_data(
    metadata: Dict,
    video_path: Optional[Path],
    relative_video_path: str,
    speech_data: Optional[Dict] = None,
) -> Dict:
    """Create prepared data structure from metadata, video path, and speech recognition data."""
    prepared_data = {
        "video_path": relative_video_path if video_path else None,
        "video_url": metadata.get("video_url", ""),
        "question": metadata.get("question", ""),
        "correct_answer": metadata.get("correct_answer", ""),
        "incorrect_answers": metadata.get("incorrect_answers", []),
        "timestamp": metadata.get("timestamp", ""),
        "video_exists": video_path is not None,
        "speech_recognition": speech_data if speech_data else None,
    }

    return prepared_data


def process_metadata_and_videos(
    metadata_folder: Path,
    videos_folder: Path,
    output_folder: Path,
    enable_speech_recognition: bool = True,
    asr_model: str = "seamless-m4t",
    asr_language: str = "auto",
    whisper_model: str = "large-v3",
) -> Tuple[int, int, int]:
    """
    Process metadata and video folders to generate prepared JSON files with speech recognition.

    Returns:
        Tuple of (total_processed, successful_matches, missing_videos)
    """
    metadata_folder = Path(metadata_folder)
    videos_folder = Path(videos_folder)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize speech recognizer if enabled
    speech_recognizer = None
    if enable_speech_recognition:
        # Prepare ASR model parameters
        asr_kwargs = {
            "whisper_model": whisper_model,
            "language": asr_language if asr_language != "auto" else "en"
        }
        if asr_model in ["seamless-m4t", "seamless", "m4t"]:
            # Map language codes for Seamless M4T
            lang_mapping = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu"}
            asr_kwargs["target_lang"] = lang_mapping.get(asr_language, "eng")
        
        speech_recognizer = create_speech_recognizer(asr_model, **asr_kwargs)

    # Find all JSON files in metadata folder
    metadata_files = list(metadata_folder.glob("*.json"))

    if not metadata_files:
        logging.warning(f"No JSON files found in metadata folder: {metadata_folder}")
        return 0, 0, 0

    total_processed = 0
    successful_matches = 0
    missing_videos = 0

    logging.info(f"Processing {len(metadata_files)} metadata files...")
    if enable_speech_recognition:
        logging.info("Speech recognition is enabled")

    for metadata_file in metadata_files:
        total_processed += 1
        stem = metadata_file.stem

        # Load metadata
        metadata = load_metadata_file(metadata_file)
        if metadata is None:
            logging.error(f"Skipping {metadata_file} due to loading error")
            continue

        # Find corresponding video file
        video_path = find_video_file(videos_folder, stem)

        speech_data = None
        if video_path:
            # Create relative path for video
            relative_video_path = os.path.relpath(video_path, output_folder.parent)
            successful_matches += 1
            logging.debug(f"Found video for {stem}: {video_path}")

            # Perform speech recognition if enabled
            if speech_recognizer:
                logging.info(f"Recognizing speech from {stem}...")
                speech_data = speech_recognizer.recognize_speech(str(video_path))
                if speech_data:
                    logging.info(f"Speech recognition completed for {stem}")
                else:
                    logging.warning(f"Speech recognition failed for {stem}")
        else:
            relative_video_path = ""
            missing_videos += 1
            logging.warning(f"Missing video file for metadata: {stem}")

        print(metadata)
        # Create prepared data with speech recognition data
        prepared_data = create_prepared_data(
            metadata, video_path, relative_video_path, speech_data
        )

        # Save prepared data to output folder
        output_file = output_folder / f"{stem}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(prepared_data, f, indent=2, ensure_ascii=False)
            logging.debug(f"Created prepared file: {output_file}")
        except Exception as e:
            logging.error(f"Failed to write output file {output_file}: {e}")

    return total_processed, successful_matches, missing_videos


class SpeechRecognizer:
    """Speech recognition using Seamless M4T model."""

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        target_lang: str = "eng",
    ):
        """Initialize the speech recognition model."""
        self.model_name = model_name
        self.target_lang = target_lang
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self._initialized = False

        # Log device information
        if torch.cuda.is_available():
            logging.info(f"CUDA is available. Using device: {self.device}")
            logging.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            logging.info(f"CUDA is not available. Using device: {self.device}")

    def _lazy_init(self):
        """Lazy initialization of the model to save memory."""
        if not self._initialized:
            logging.info(f"Loading speech recognition model: {self.model_name}")
            # Initialize processor with explicit sampling rate to avoid silent errors
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                sampling_rate=self.sample_rate,  # Explicitly set sampling rate
            )
            self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name)

            # Move model to GPU if available
            self.model = self.model.to(self.device)
            logging.info(f"Speech recognition model loaded and moved to {self.device}")

            # Log model memory usage if on GPU
            if self.device.type == "cuda":
                logging.info(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )
                logging.info(
                    f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                )

            self._initialized = True
            logging.info("Speech recognition model loaded successfully")

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file and return temporary audio path."""
        try:
            video_stem = Path(video_path).stem
            temp_audio_path = f"/tmp/{video_stem}_temp_audio.wav"

            # Extract audio using moviepy
            video = mp.VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                logging.warning(f"No audio track found in video: {video_path}")
                return None

            # Save as WAV file
            audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            video.close()
            audio.close()

            return temp_audio_path

        except Exception as e:
            logging.error(f"Failed to extract audio from {video_path}: {e}")
            return None

    def recognize_speech(self, video_path: str) -> Optional[Dict]:
        """Recognize speech from video file using Seamless M4T."""
        if not os.path.exists(video_path):
            return None

        self._lazy_init()

        # Extract audio from video
        temp_audio_path = self.extract_audio_from_video(video_path)
        if not temp_audio_path:
            return None

        try:
            # Load and preprocess audio
            audio, orig_freq = torchaudio.load(temp_audio_path)

            # Resample to 16kHz if necessary (required for Seamless M4T)
            if orig_freq != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=orig_freq, new_freq=self.sample_rate
                )

            # Process audio for the model
            audio_inputs = self.processor(audios=audio, return_tensors="pt")

            # Move input tensors to the same device as model
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

            # Generate speech recognition with Seamless M4T
            with torch.no_grad():
                # Use the correct method for Seamless M4T v2 - specify generation parameters
                generated_outputs = self.model.generate(
                    **audio_inputs,
                    tgt_lang=self.target_lang,
                    generate_speech=False,  # We only want text, not speech generation
                    return_intermediate_token_ids=True,
                )

                # Extract text tokens from the output
                # Seamless M4T v2 returns a dictionary with text_sequences
                if hasattr(generated_outputs, "sequences"):
                    text_tokens = generated_outputs.sequences
                elif (
                    isinstance(generated_outputs, dict)
                    and "sequences" in generated_outputs
                ):
                    text_tokens = generated_outputs["sequences"]
                elif (
                    isinstance(generated_outputs, (list, tuple))
                    and len(generated_outputs) > 0
                ):
                    text_tokens = generated_outputs[0]
                else:
                    text_tokens = generated_outputs

                # Ensure we have integer token IDs
                if torch.is_tensor(text_tokens):
                    # Move to CPU and ensure integer type
                    text_tokens = text_tokens.cpu()
                    if text_tokens.dtype in [
                        torch.float32,
                        torch.float64,
                        torch.float16,
                    ]:
                        # Convert from logits/probabilities to token IDs
                        text_tokens = torch.argmax(text_tokens, dim=-1)
                    text_tokens = text_tokens.long()  # Ensure integer type
                elif isinstance(text_tokens, list):
                    # Convert list to tensor of integers
                    try:
                        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
                    except:
                        # If conversion fails, try to extract integers
                        text_tokens = torch.tensor(
                            [
                                int(token) if isinstance(token, (int, float)) else token
                                for token in text_tokens
                            ],
                            dtype=torch.long,
                        )

                # Handle batch dimension
                if text_tokens.dim() > 1:
                    tokens_to_decode = text_tokens[0]
                else:
                    tokens_to_decode = text_tokens

                # Ensure tokens are integers for decoding
                tokens_to_decode = tokens_to_decode.long()

                # Decode the tokens to text
                recognized_text = self.processor.decode(
                    tokens_to_decode, skip_special_tokens=True
                )

            speech_data = {
                "recognized_text": recognized_text,
                "audio_duration_seconds": float(audio.shape[1] / self.sample_rate),
                "target_language": self.target_lang,
                "model_name": self.model_name,
                "model_type": "seamless_m4t_v2",
                "device_used": str(self.device),
            }

            logging.debug(
                f"Speech recognized from {video_path}: {recognized_text[:100]}..."
            )
            return speech_data

        except Exception as e:
            logging.error(f"Failed to recognize speech from {video_path}: {e}")
            # Log more details about the error for debugging
            logging.debug(f"Error details: {str(e)}")
            if "generated_outputs" in locals():
                logging.debug(f"Generated outputs type: {type(generated_outputs)}")
                if hasattr(generated_outputs, "shape"):
                    logging.debug(f"Generated outputs shape: {generated_outputs.shape}")
                elif hasattr(generated_outputs, "sequences"):
                    logging.debug(
                        f"Generated outputs sequences shape: {generated_outputs.sequences.shape}"
                    )
            if "text_tokens" in locals() and torch.is_tensor(text_tokens):
                logging.debug(
                    f"Text tokens shape: {text_tokens.shape}, dtype: {text_tokens.dtype}"
                )
            return None

        finally:
            # Clean up temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass


class WhisperSpeechRecognizer:
    """Speech recognition using OpenAI Whisper model."""

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "en",
    ):
        """Initialize the Whisper speech recognition model."""
        self.model_name = model_name
        self.language = language
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False

        # Log device information
        if torch.cuda.is_available():
            logging.info(f"CUDA is available. Using device: {self.device}")
            logging.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            logging.info(f"CUDA is not available. Using device: {self.device}")

    def _lazy_init(self):
        """Lazy initialization of the Whisper model to save memory."""
        if not self._initialized:
            logging.info(f"Loading Whisper speech recognition model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            
            # Log model memory usage if on GPU
            if self.device.type == "cuda":
                logging.info(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                )
                logging.info(
                    f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                )

            self._initialized = True
            logging.info("Whisper speech recognition model loaded successfully")

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file and return temporary audio path."""
        try:
            video_stem = Path(video_path).stem
            temp_audio_path = f"/tmp/{video_stem}_temp_audio.wav"

            # Extract audio using moviepy
            video = mp.VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                logging.warning(f"No audio track found in video: {video_path}")
                return None

            # Save as WAV file
            audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            video.close()
            audio.close()

            return temp_audio_path

        except Exception as e:
            logging.error(f"Failed to extract audio from {video_path}: {e}")
            return None

    def recognize_speech(self, video_path: str) -> Optional[Dict]:
        """Recognize speech from video file using Whisper."""
        temp_audio_path = None
        try:
            # Initialize model if not already done
            self._lazy_init()

            # Extract audio from video
            temp_audio_path = self.extract_audio_from_video(video_path)
            if temp_audio_path is None:
                logging.error(f"Failed to extract audio from {video_path}")
                return None

            logging.info(f"Transcribing audio with Whisper model: {self.model_name}")

            # Transcribe using Whisper
            result = self.model.transcribe(
                temp_audio_path,
                language=self.language if self.language != "auto" else None,
                verbose=False
            )

            # Extract transcription text
            recognized_text = result.get("text", "").strip()
            
            if not recognized_text:
                logging.warning(f"No speech recognized from {video_path}")
                return None

            # Create speech data structure compatible with existing code
            speech_data = {
                "recognized_text": recognized_text,
                "language": result.get("language", self.language),
                "model_name": f"whisper-{self.model_name}",
                "device_used": str(self.device),
                "segments": result.get("segments", []),  # Whisper provides detailed segments
                "confidence": None  # Whisper doesn't provide overall confidence score
            }

            logging.debug(
                f"Speech recognized from {video_path}: {recognized_text[:100]}..."
            )
            return speech_data

        except Exception as e:
            logging.error(f"Failed to recognize speech from {video_path}: {e}")
            logging.debug(f"Error details: {str(e)}")
            return None

        finally:
            # Clean up temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass


def create_speech_recognizer(asr_model: str = "seamless-m4t", **kwargs):
    """Factory function to create appropriate speech recognizer."""
    if asr_model.lower() in ["whisper", "whisper-large-v3", "whisper-v3-large"]:
        # Extract whisper-specific parameters
        whisper_model = kwargs.get("whisper_model", "large-v3")
        language = kwargs.get("language", "en")
        return WhisperSpeechRecognizer(model_name=whisper_model, language=language)
    elif asr_model.lower() in ["seamless-m4t", "seamless", "m4t"]:
        # Extract seamless-m4t specific parameters
        model_name = kwargs.get("model_name", "facebook/seamless-m4t-v2-large")
        target_lang = kwargs.get("target_lang", "eng")
        return SpeechRecognizer(model_name=model_name, target_lang=target_lang)
    else:
        logging.warning(f"Unknown ASR model: {asr_model}, defaulting to Seamless M4T")
        return SpeechRecognizer()


def main():
    """Main function to parse arguments and process data."""
    parser = argparse.ArgumentParser(
        description="Prepare VQA data by processing metadata and video folders"
    )

    parser.add_argument(
        "--metadata-folder",
        type=str,
        required=True,
        help="Path to folder containing metadata JSON files",
    )

    parser.add_argument(
        "--videos-folder",
        type=str,
        required=True,
        help="Path to folder containing video files",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to output folder for prepared JSON files",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--asr-model",
        type=str,
        default="seamless-m4t",
        choices=["seamless-m4t", "whisper", "whisper-large-v3", "whisper-v3-large"],
        help="ASR model to use for speech recognition (default: seamless-m4t)",
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size when using whisper ASR (default: large-v3)",
    )

    parser.add_argument(
        "--asr-language",
        type=str,
        default="auto",
        help="Language code for ASR (e.g., 'en', 'es', 'fr', 'auto' for auto-detection)",
    )

    parser.add_argument(
        "--enable-speech-recognition",
        action="store_true",
        default=True,
        help="Enable speech recognition for videos (default: True)",
    )

    parser.add_argument(
        "--disable-speech-recognition",
        action="store_true",
        help="Disable speech recognition for videos",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate input folders
    metadata_folder = Path(args.metadata_folder)
    videos_folder = Path(args.videos_folder)

    if not metadata_folder.exists():
        logging.error(f"Metadata folder does not exist: {metadata_folder}")
        return 1

    if not videos_folder.exists():
        logging.error(f"Videos folder does not exist: {videos_folder}")
        return 1

    # Determine speech recognition setting
    enable_speech_recognition = not args.disable_speech_recognition

    logging.info(f"Metadata folder: {metadata_folder}")
    logging.info(f"Videos folder: {videos_folder}")
    logging.info(f"Output folder: {args.output_folder}")
    logging.info(
        f"Speech recognition: {'enabled' if enable_speech_recognition else 'disabled'}"
    )
    if enable_speech_recognition:
        logging.info(f"ASR model: {args.asr_model}")
        if args.asr_model.startswith("whisper"):
            logging.info(f"Whisper model size: {args.whisper_model}")
        logging.info(f"ASR language: {args.asr_language}")

    # Process data
    total_processed, successful_matches, missing_videos = process_metadata_and_videos(
        metadata_folder,
        videos_folder,
        Path(args.output_folder),
        enable_speech_recognition,
        args.asr_model,
        args.asr_language,
        args.whisper_model,
    )

    # Print summary
    logging.info("=" * 50)
    logging.info("PROCESSING SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total metadata files processed: {total_processed}")
    logging.info(f"Successful video matches: {successful_matches}")
    logging.info(f"Missing videos: {missing_videos}")

    if missing_videos > 0:
        logging.warning(f"Warning: {missing_videos} videos were missing")

    logging.info(f"Prepared JSON files saved to: {args.output_folder}")

    return 0


if __name__ == "__main__":
    exit(main())
