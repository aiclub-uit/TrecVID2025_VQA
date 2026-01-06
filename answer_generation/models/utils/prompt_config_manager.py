"""
Prompt Configuration Manager

Manages loading and using YAML-based prompt templates for VQA models.
"""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class PromptConfigManager:
    """
    Manages prompt templates loaded from YAML configuration files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prompt configuration manager.

        Args:
            config_path: Path to the prompt templates YAML file.
                        If None, uses default path in configs directory.
        """
        if config_path is None:
            # Default to prompt_templates.yml in the configs directory
            current_dir = Path(__file__).parent
            config_path = current_dir / "configs" / "prompt_templates.yml"
        
        self.config_path = Path(config_path)
        self.templates = {}
        self.defaults = {}
        self.variables = {}
        self.validation = {}
        
        self.load_config()

    def load_config(self) -> None:
        """Load prompt templates from YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            self.templates = config.get('templates', {})
            self.defaults = config.get('defaults', {})
            self.variables = config.get('variables', {})
            self.validation = config.get('validation', {})

            print(f"✓ Loaded {len(self.templates)} prompt templates from {self.config_path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in prompt config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompt config: {e}")

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get information about a specific template.

        Args:
            template_name: Name of the template

        Returns:
            Dictionary with template information
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]
        return {
            'name': template.get('name', template_name),
            'description': template.get('description', 'No description available'),
            'type': template.get('type', 'unknown'),
            'template': template.get('template', '')
        }

    def get_template_by_type(self, template_type: str) -> str:
        """
        Get default template name for a specific type.

        Args:
            template_type: Type of template ('visual_only', 'multimodal')

        Returns:
            Template name
        """
        if template_type in self.defaults:
            return self.defaults[template_type]
        return self.defaults.get('fallback', 'basic_vqa')

    def format_prompt(
        self,
        template_name: str,
        question: str,
        audio_transcription: Optional[str] = None,
        video_context: Optional[str] = None,
        custom_variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format a prompt using the specified template.

        Args:
            template_name: Name of the template to use
            question: The question to be answered
            audio_transcription: Audio transcription text
            video_context: Video context description
            custom_variables: Additional variables for template formatting

        Returns:
            Formatted prompt string
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template_info = self.templates[template_name]
        template_text = template_info.get('template', '')

        # Prepare variables for formatting
        variables = {
            'question': question,
            'audio_transcription': audio_transcription or self._get_fallback('audio_transcription'),
            'video_context': video_context or self._get_fallback('video_context')
        }

        # Add custom variables if provided
        if custom_variables:
            variables.update(custom_variables)

        # Validate required variables
        self._validate_variables(variables)

        try:
            formatted_prompt = template_text.format(**variables)
            return formatted_prompt.strip()
        except KeyError as e:
            raise ValueError(f"Missing variable for template formatting: {e}")

    def _get_fallback(self, variable_name: str) -> str:
        """Get fallback value for a variable."""
        variable_info = self.variables.get(variable_name, {})
        return variable_info.get('fallback', '')

    def _validate_variables(self, variables: Dict[str, Any]) -> None:
        """Validate that required variables are provided."""
        required_vars = self.validation.get('required_variables', [])
        
        for var in required_vars:
            if var not in variables or not variables[var]:
                raise ValueError(f"Required variable '{var}' is missing or empty")

    def create_prompt(
        self,
        question: str,
        template_name: Optional[str] = None,
        audio_transcription: Optional[str] = None,
        video_context: Optional[str] = None,
        prefer_multimodal: bool = True,
        custom_variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a prompt with automatic template selection or explicit template.

        Args:
            question: The question to be answered
            template_name: Specific template to use (optional)
            audio_transcription: Audio transcription text
            video_context: Video context description
            prefer_multimodal: Whether to prefer multimodal templates when audio is available
            custom_variables: Additional variables for template formatting

        Returns:
            Formatted prompt string
        """
        # Auto-select template if not specified
        if template_name is None:
            has_audio = bool(audio_transcription and audio_transcription.strip())
            
            if has_audio and prefer_multimodal:
                template_name = self.get_template_by_type('multimodal')
            else:
                template_name = self.get_template_by_type('visual_only')

        return self.format_prompt(
            template_name=template_name,
            question=question,
            audio_transcription=audio_transcription,
            video_context=video_context,
            custom_variables=custom_variables
        )

    def list_templates_by_type(self, template_type: str) -> List[str]:
        """
        List all templates of a specific type.

        Args:
            template_type: Type to filter by ('visual_only', 'multimodal')

        Returns:
            List of template names
        """
        return [
            name for name, template in self.templates.items()
            if template.get('type') == template_type
        ]

    def print_template_summary(self) -> None:
        """Print a summary of all available templates."""
        print("\n" + "=" * 60)
        print("AVAILABLE PROMPT TEMPLATES")
        print("=" * 60)
        
        for name, template in self.templates.items():
            print(f"\n{name.upper()}")
            print(f"  Name: {template.get('name', name)}")
            print(f"  Type: {template.get('type', 'unknown')}")
            print(f"  Description: {template.get('description', 'No description')}")
        
        print(f"\n📁 Config file: {self.config_path}")
        print(f"📊 Total templates: {len(self.templates)}")
        print("=" * 60)

    def add_template(
        self,
        template_name: str,
        template_text: str,
        name: str,
        description: str,
        template_type: str = "multimodal"
    ) -> None:
        """
        Add a new template programmatically.

        Args:
            template_name: Unique identifier for the template
            template_text: The template string with format placeholders
            name: Human-readable name
            description: Template description
            template_type: Type of template ('visual_only', 'multimodal')
        """
        self.templates[template_name] = {
            'name': name,
            'description': description,
            'type': template_type,
            'template': template_text
        }

    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save the config. If None, saves to original location.
        """
        if output_path is None:
            output_path = self.config_path

        config = {
            'templates': self.templates,
            'defaults': self.defaults,
            'variables': self.variables,
            'validation': self.validation
        }

        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False, indent=2)

        print(f"✓ Prompt configuration saved to {output_path}")


# Convenience function for easy usage
def create_prompt_manager(config_path: Optional[str] = None) -> PromptConfigManager:
    """
    Create a prompt configuration manager instance.

    Args:
        config_path: Optional path to config file

    Returns:
        PromptConfigManager instance
    """
    return PromptConfigManager(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Test the prompt configuration manager
    try:
        manager = create_prompt_manager()
        
        # Print summary
        manager.print_template_summary()
        
        # Test template creation
        print("\n" + "=" * 60)
        print("TESTING TEMPLATE CREATION")
        print("=" * 60)
        
        # Test visual-only prompt
        prompt1 = manager.create_prompt(
            question="What is happening in this video?",
            template_name="basic_vqa"
        )
        print("BASIC VQA PROMPT:")
        print(prompt1)
        
        # Test multimodal prompt
        prompt2 = manager.create_prompt(
            question="What cooking technique is being demonstrated?",
            template_name="audio_enhanced",
            audio_transcription="First, we heat the oil in the pan, then add the vegetables.",
            video_context="Kitchen cooking demonstration"
        )
        print("\nAUDIO-ENHANCED PROMPT:")
        print(prompt2)
        
        # Test auto-selection
        prompt3 = manager.create_prompt(
            question="What is the main topic?",
            audio_transcription="Welcome to our tutorial on machine learning basics."
        )
        print("\nAUTO-SELECTED PROMPT:")
        print(prompt3)
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
