from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelPrompt:
    """Manages and organizes multiple prompts for system and user interactions."""

    def __init__(
        self,
        ref_sys_prompt: str,
        ref_usr_prompt: str,
        syn_sys_prompt: str,
        syn_usr_prompt: str,
    ):
        """
        Initializes ModelPrompt with structured prompts.

        Args:
            ref_sys_prompt (str): Reference system prompt.
            ref_usr_prompt (str): Reference user prompt.
            syn_sys_prompt (str): Synthesis system prompt.
            syn_usr_prompt (str): Synthesis user prompt.
        """
        self.ref_sys_prompt = ref_sys_prompt
        self.ref_usr_prompt = ref_usr_prompt
        self.syn_sys_prompt = syn_sys_prompt
        self.syn_usr_prompt = syn_usr_prompt

    def get_prompts(self) -> dict:
        """
        Returns the stored prompts as a dictionary.

        Returns:
            dict: System and user prompts.
        """
        return {
            "ref_sys_prompt": self.ref_sys_prompt,
            "ref_usr_prompt": self.ref_usr_prompt,
            "syn_sys_prompt": self.syn_sys_prompt,
            "syn_usr_prompt": self.syn_usr_prompt,
        }


class GenerationInput(BaseModel):
    """
    A model for validating generation inputs for a text generation model.
    """

    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    model: Optional[str] = None
    stream: Optional[bool] = None

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        base_dict = super().model_dump(exclude_unset=True, *args, **kwargs)

        # Special handling: if user sent nothing (empty dict), return only default model
        if not base_dict:
            return {"model": "openai/gpt-4o"}

        return {k: v for k, v in base_dict.items() if v is not None}
