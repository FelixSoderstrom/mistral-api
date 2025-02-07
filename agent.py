from typing import Optional, Dict, Any, AsyncGenerator
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import settings


class MistralAgent:
    def __init__(self):
        """Initialize the Mistral agent with GGUF model."""
        self.model = None
        self.llm = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with optimized settings for GPU inference."""
        # Initialize the low-level model
        self.model = Llama(
            model_path=settings.MODEL_PATH,
            n_gpu_layers=settings.N_GPU_LAYERS,
            n_batch=settings.N_BATCH,
            n_threads=settings.N_THREADS,
            n_ctx=4096,
            verbose=False,
        )

        # Initialize LangChain wrapper with streaming support
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=settings.MODEL_PATH,
            n_gpu_layers=settings.N_GPU_LAYERS,
            n_batch=settings.N_BATCH,
            n_threads=settings.N_THREADS,
            callback_manager=callback_manager,
            verbose=False,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            streaming=True,
        )

    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate a complete response using the model."""
        try:
            # Use the low-level model for generation
            response = self.model.create_completion(
                prompt,
                max_tokens=max_tokens or settings.DEFAULT_MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                stream=False,  # No streaming needed
                **kwargs,
            )

            # Extract the complete text
            complete_text = response["choices"][0]["text"]

            # If the response starts with fragments before "Once upon a time",
            # clean it up
            if "Once upon a time" in complete_text:
                story_start_index = complete_text.find("Once upon a time")
                complete_text = complete_text[story_start_index:]

            # Clean up line breaks and multiple spaces
            complete_text = " ".join(complete_text.split())

            return complete_text

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def __del__(self):
        """Cleanup when the agent is destroyed."""
        if self.model is not None:
            del self.model
        if self.llm is not None:
            del self.llm
