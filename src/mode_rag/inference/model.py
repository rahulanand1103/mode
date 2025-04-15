from typing import List, Dict
from litellm import completion


class LiteLLMModel:
    """
    Handles interaction with language models through LiteLLM.
    """

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends a list of messages to the model and retrieves the response.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.

        Returns:
            str: The model-generated content.
        """
        print("Model-parameter", kwargs)
        response = completion(messages=messages, **kwargs)
        return response.choices[0].message.content
