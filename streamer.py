import json
from typing import Any, AsyncGenerator


class ResponseStreamer:
    """Handles streaming responses from the Mistral model."""

    @staticmethod
    async def stream_response(
        response_generator: AsyncGenerator,
    ) -> AsyncGenerator[str, None]:
        """Stream the generated text with proper JSON formatting."""
        try:
            current_chunk = ""
            async for response in response_generator:
                text = response["choices"][0]["text"]

                # Accumulate text until we have a natural break
                current_chunk += text

                # Send chunk if we have a sentence break or sufficient length
                if any(char in text for char in ".!?\n") or len(current_chunk) > 20:
                    chunk = {"text": current_chunk, "finished": False}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    current_chunk = ""

            # Send any remaining text
            if current_chunk:
                chunk = {"text": current_chunk, "finished": False}
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send final message
            yield f"data: {json.dumps({'text': '', 'finished': True})}\n\n"

        except Exception as e:
            error_response = {"error": str(e), "finished": True}
            yield f"data: {json.dumps(error_response)}\n\n"
