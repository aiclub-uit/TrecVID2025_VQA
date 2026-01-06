"""
Simple Prompt Definitions for VQA Answer Generation

This module provides two simple prompt templates that can be imported directly.
"""

# Direct prompt templates - can be imported and used immediately
VISUAL_ONLY_PROMPT = """You are an expert multi-modal AI assistant. Analyze the video content, then respond with only the answer, concisely and without restating the question.

### QUESTION ###
{question}

### ANSWER ###

"""

# AUDIO_VISUAL_PROMPT = """You are an expert multi-modal AI assistant.
# Analyze the video content and audio transcription, then answer in the shortest possible form in English.
# Do not restate the question, explain, or add extra details. Output only the direct answer.

# ### AUDIO TRANSCRIPTION ###
# {audio_transcription}

# ### QUESTION ###
# {question}

# ### ANSWER ###
# """

# AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

# {question}
# """


# AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

# ### AUDIO TRANSCRIPTION ###
# {audio_transcription}

# ### QUESTION ###
# {question}

# ### ANSWER ###

# """

AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

### VIDEO CONTEXT ###
{video_context}

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###
"""

# Enhanced multi-modal prompt that includes video context
# AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.
# ENHANCED_AUDIO_VISUAL_PROMPT = """Analyze the video content, audio transcription, and video context description, then respond with a detailed and thorough answer.
# ENHANCED_AUDIO_VISUAL_PROMPT = """Analyze the video content, audio transcription, and video context description, then respond with only the answer, concisely and without restating the question.
ENHANCED_AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

### VIDEO CONTEXT ###
{video_context}

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###
"""

# Prompt for generating video context description
VIDEO_CONTEXT_PROMPT = """You are an expert video analyst. Describe what is happening in this video in 2-3 concise sentences. Focus on the key visual elements, actions, objects, people, and setting that would be relevant for answering questions about the video.

### VIDEO DESCRIPTION ###
"""

# detailed prompt for audio-visual analysis
# AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with a detailed and thorough answer.

# ### AUDIO TRANSCRIPTION ###
# {audio_transcription}

# ### QUESTION ###
# {question}

# ### ANSWER ###

# """


# AUDIO_VISUAL_PROMPT = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

# ### AUDIO TRANSCRIPTION ###
# {audio_transcription}

# ### QUESTION ###
# {question}

# ### ANSWER ###

# """

# AUDIO_VISUAL_PROMPT = """Analyze the video content, then respond with only the answer, concisely and without restating the question.

# ### QUESTION ###
# {question}

# ### ANSWER ###
# """

# AUDIO_VISUAL_PROMPT = """You are given video content and its audio transcription.
# Your output must be ONLY the direct answer to the question, in as few words as possible.
# No additional explanation, no rephrasing of the question.

# AUDIO: {audio_transcription}
# QUESTION: {question}
# ANSWER:
# """

# AUDIO_VISUAL_PROMPT = """You are an expert multi-modal AI assistant. Analyze the video content and audio transcription to answer the question accurately.

# ### VIDEO CONTENT ###
# {visual_context}

# ### AUDIO TRANSCRIPTION ###
# {audio_transcription}

# ### QUESTION ###
# {question}

# ### ANSWER ###

# """


def create_visual_only_prompt(question: str, video_context: str = "") -> str:
    """
    Create a visual-only prompt for video analysis.

    Args:
        question: User's question
        video_context: Optional description of video content

    Returns:
        Formatted prompt string
    """
    return VISUAL_ONLY_PROMPT.format(
        question=question, video_context=video_context or "Video content analysis"
    ).strip()


def create_audio_visual_prompt(
    question: str, visual_context: str = "", audio_transcription: str = ""
) -> str:
    """
    Create an audio-visual prompt for multimodal analysis.

    Args:
        question: User's question
        visual_context: Description of visual content
        audio_transcription: Audio transcription text

    Returns:
        Formatted prompt string
    """
    return AUDIO_VISUAL_PROMPT.format(
        question=question,
        visual_context=visual_context or "Video content analysis",
        audio_transcription=audio_transcription or "No audio transcription available.",
    ).strip()


def create_enhanced_audio_visual_prompt(
    question: str, video_context: str = "", audio_transcription: str = ""
) -> str:
    """
    Create an enhanced audio-visual prompt that includes video context description.

    Args:
        question: User's question
        video_context: Generated description of video content
        audio_transcription: Audio transcription text

    Returns:
        Formatted prompt string
    """
    return ENHANCED_AUDIO_VISUAL_PROMPT.format(
        question=question,
        video_context=video_context or "No video context available.",
        audio_transcription=audio_transcription or "No audio transcription available.",
    ).strip()


def create_video_context_prompt() -> str:
    """
    Create a prompt for generating video context description.

    Returns:
        Formatted prompt string for video description generation
    """
    return VIDEO_CONTEXT_PROMPT.strip()


def create_visual_only_prompt_with_previous(question: str, video_context: str = "", previous_answers: list = None) -> str:
    """
    Create a visual-only prompt that includes context about previous incorrect answers.

    Args:
        question: User's question
        video_context: Optional description of video content
        previous_answers: List of previously given incorrect answers

    Returns:
        Formatted prompt string
    """
    if previous_answers:
        previous_answers_text = "\n".join([f"- {answer}" for answer in previous_answers])
        return VISUAL_ONLY_PROMPT_WITH_PREVIOUS.format(
            question=question,
            video_context=video_context or "Video content analysis",
            previous_answers=previous_answers_text
        ).strip()
    else:
        return create_visual_only_prompt(question, video_context)


def create_audio_visual_prompt_with_previous(
    question: str, visual_context: str = "", audio_transcription: str = "", previous_answers: list = None
) -> str:
    """
    Create an audio-visual prompt that includes context about previous incorrect answers.

    Args:
        question: User's question
        visual_context: Description of visual content
        audio_transcription: Audio transcription text
        previous_answers: List of previously given incorrect answers

    Returns:
        Formatted prompt string
    """
    if previous_answers:
        previous_answers_text = "\n".join([f"- {answer}" for answer in previous_answers])
        return AUDIO_VISUAL_PROMPT_WITH_PREVIOUS.format(
            question=question,
            video_context=visual_context or "Video content analysis",
            audio_transcription=audio_transcription or "No audio transcription available.",
            previous_answers=previous_answers_text
        ).strip()
    else:
        return create_audio_visual_prompt(question, visual_context, audio_transcription)


def create_enhanced_audio_visual_prompt_with_previous(
    question: str, video_context: str = "", audio_transcription: str = "", previous_answers: list = None
) -> str:
    """
    Create an enhanced audio-visual prompt that includes context about previous incorrect answers.

    Args:
        question: User's question
        video_context: Generated description of video content
        audio_transcription: Audio transcription text
        previous_answers: List of previously given incorrect answers

    Returns:
        Formatted prompt string
    """
    if previous_answers:
        previous_answers_text = "\n".join([f"- {answer}" for answer in previous_answers])
        return ENHANCED_AUDIO_VISUAL_PROMPT_WITH_PREVIOUS.format(
            question=question,
            audio_transcription=audio_transcription or "No audio transcription available.",
            previous_answers=previous_answers_text
        ).strip()
    else:
        return create_enhanced_audio_visual_prompt(question, video_context, audio_transcription)


# Prompt templates with previous answers context for generating different responses
VISUAL_ONLY_PROMPT_WITH_PREVIOUS = """You are an expert multi-modal AI assistant. Analyze the video content, then respond with only the answer, concisely and without restating the question.

IMPORTANT: The following answers have already been given for this question and are incorrect:
{previous_answers}

Please provide a DIFFERENT answer than the ones above. Think carefully and reconsider the video content to provide an alternative, more accurate response.

### QUESTION ###
{question}

### ANSWER ###

"""

AUDIO_VISUAL_PROMPT_WITH_PREVIOUS = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

IMPORTANT: The following answers have already been given for this question and are incorrect:
{previous_answers}

Please provide a DIFFERENT answer than the ones above. Think carefully and reconsider the video and audio content to provide an alternative, more accurate response.

### VIDEO CONTEXT ###
{video_context}

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###
"""

ENHANCED_AUDIO_VISUAL_PROMPT_WITH_PREVIOUS = """Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.

IMPORTANT: The following answers have already been given for this question and are incorrect:
{previous_answers}

Please provide a DIFFERENT answer than the ones above. Think carefully and reconsider the video and audio content to provide an alternative, more accurate response.

### AUDIO TRANSCRIPTION ###
{audio_transcription}

### QUESTION ###
{question}

### ANSWER ###
"""


# Example usage and testing
if __name__ == "__main__":
    # Test visual-only prompt
    print("=== VISUAL-ONLY PROMPT ===")
    visual_prompt = create_visual_only_prompt(
        question="What is happening in the video?",
        video_context="Video showing a cooking demonstration in a kitchen.",
    )
    print(visual_prompt)

    # Test audio-visual prompt
    print("\n=== AUDIO-VISUAL PROMPT ===")
    audio_visual_prompt = create_audio_visual_prompt(
        question="What assembly step is being demonstrated?",
        visual_context="Video of a person assembling furniture in a workshop.",
        audio_transcription="First, we need to attach the legs to the seat. Make sure the screws are tight.",
    )
    print(audio_visual_prompt)

    # Test enhanced audio-visual prompt
    print("\n=== ENHANCED AUDIO-VISUAL PROMPT ===")
    enhanced_prompt = create_enhanced_audio_visual_prompt(
        question="What cooking technique is being demonstrated?",
        video_context="The video shows a chef in a professional kitchen preparing pasta. The chef is boiling water in a large pot, adding salt, and then adding fresh linguine pasta. Steam rises from the pot as the chef stirs the pasta with a wooden spoon.",
        audio_transcription="Now we're going to add our fresh pasta to the boiling salted water. Make sure the water is at a rolling boil before adding the pasta. This will take about 3-4 minutes since it's fresh.",
    )
    print(enhanced_prompt)

    # Test video context prompt
    print("\n=== VIDEO CONTEXT PROMPT ===")
    context_prompt = create_video_context_prompt()
    print(context_prompt)

    print("\n=== Prompt templates ready for direct import ===")
    print(
        "Use: from prompt_builder import create_visual_only_prompt, create_audio_visual_prompt, create_enhanced_audio_visual_prompt"
    )
    print(
        "With previous answers: from prompt_builder import create_visual_only_prompt_with_previous, create_audio_visual_prompt_with_previous, create_enhanced_audio_visual_prompt_with_previous"
    )
    print(
        "Or import templates directly: from prompt_builder import VISUAL_ONLY_PROMPT, AUDIO_VISUAL_PROMPT, ENHANCED_AUDIO_VISUAL_PROMPT, VIDEO_CONTEXT_PROMPT"
    )
