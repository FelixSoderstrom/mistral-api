from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from agent import MistralAgent
from streamer import ResponseStreamer

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
)

# Add rate limiter to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for AWS monitoring."""
    return {"status": "healthy"}


@app.post("/generate")
@limiter.limit(f"{settings.RATE_LIMIT_CALLS}/{settings.RATE_LIMIT_SECONDS}s")
async def generate(request_data: GenerateRequest, request: Request):
    """Generate text from the model."""
    try:
        # Create new agent instance for this request
        agent = MistralAgent()

        # Get the complete response
        response_text = await agent.generate_response(
            prompt=request_data.prompt,
            max_tokens=request_data.max_tokens,
        )

        return {"text": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # Use only 1 worker for GPU inference
    )
