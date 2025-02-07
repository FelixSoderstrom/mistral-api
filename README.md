# Mistral API

A FastAPI-based REST API for serving GGUF language models, with built-in rate limiting and GPU acceleration support. While developed and tested with the Mistral Dolphin 2.0 model, this API is designed to work with any GGUF model.

The main idea behind creating this API is to make our game project easier to use in the end.

## Features

- 🚀 GPU-accelerated inference using llama.cpp
- ⚡ Fast response times with optimized model loading
- 🔄 Rate limiting for API protection
- 🛡️ CORS support for frontend integration
- 📝 Clean JSON responses for easy frontend consumption
- 🔧 Configurable model parameters
- 🎯 Modular design for easy model swapping

## Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended)
- CUDA 11.8+ and appropriate drivers
- 8GB+ RAM (16GB+ recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd mistral-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your GGUF model in the `models` directory:
```bash
mkdir -p models
# Place your .gguf model file here
```
Model used: [The Bloke's Mistral Dolphin 2.0 7B](https://huggingface.co/TheBloke/dolphin-2.0-mistral-7B-GGUF)

## Configuration

The API can be configured through environment variables or the `config.py` file. Key configurations include:

```python
# Model Configuration
MODEL_PATH: Path to your GGUF model
MAX_NEW_TOKENS: Maximum tokens to generate (default: 2048)
DEFAULT_MAX_TOKENS: Default tokens if not specified (default: 512)
TEMPERATURE: Generation temperature (default: 0.7)
TOP_P: Top-p sampling (default: 0.95)

# GPU Configuration
N_GPU_LAYERS: Number of layers to offload to GPU (default: 35)
N_BATCH: Batch size for prompt processing (default: 512)
N_THREADS: Number of CPU threads (default: 8)

# API Configuration
RATE_LIMIT_CALLS: Number of allowed calls per time window (default: 10)
RATE_LIMIT_SECONDS: Time window for rate limiting in seconds (default: 60)
```

## Usage

1. Start the API:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

### Endpoints

#### Health Check
```http
GET /health
```

#### Generate Text
```http
POST /generate
Content-Type: application/json

{
    "prompt": "Once upon a time",
    "max_tokens": 200
}
```

Response:
```json
{
    "text": "Once upon a time in a small village..."
}
```

## Docker Support

Build and run with Docker:

```bash
docker build -t mistral-api .
docker run -p 8000:8000 --gpus all mistral-api
```

## Model Compatibility

While this API has been tested primarily with the Mistral Dolphin 2.0 model, it is designed to work with any GGUF format model. To use a different model:

1. Place your .gguf model in the `models` directory
2. Update the `MODEL_PATH` in `config.py` or set it via environment variable
3. Adjust model parameters as needed for your specific model

## Performance Notes

- GPU acceleration is recommended for optimal performance
- Model loading time depends on the model size and GPU memory
- Response generation speed depends on the requested token count and model configuration

## License

Free to use and modify.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or send me a message.

## Contact

Bjorn or Felix
