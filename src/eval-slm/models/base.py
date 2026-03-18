from abc import ABC, abstractmethod

from backoff import on_exception, expo


class BaseS2SModel(ABC):
    def __init__(self, args):
        pass
    
    @abstractmethod
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt,
        user_instruction,
        max_new_tokens
    ):
        pass
