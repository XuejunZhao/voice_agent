# Voice AI with Deep Research Integration

This voice agent integrates deep research capabilities directly, eliminating the need for a separate HTTP server.

## Prerequisites

1. **Python 3.8+**
2. **SearxNG** search engine running on `http://127.0.0.1:38000`
   - Install via Docker: `docker run -d -p 38000:8080 searxng/searxng`
   - Or install locally following [SearxNG documentation](https://docs.searxng.org/)

3. **API Keys** (set as environment variables):
   - `DASHSCOPE_API_KEY` (required) - For voice AI (ASR/TTS/LLM)
   - `ALI_API_KEY` (optional, falls back to DASHSCOPE_API_KEY) - For deep research LLM
   - `SEARX_HOST` (optional, defaults to `http://127.0.0.1:38000`) - SearxNG host

## Installation

1. **Navigate to the voice-agent directory:**
   ```bash
   cd /Users/xuejun/code/voice-agent
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Voice Agent

1. **Set environment variables:**
   ```bash
   export DASHSCOPE_API_KEY="your-dashscope-api-key"
   export ALI_API_KEY="your-ali-api-key"  # Optional
   export SEARX_HOST="http://127.0.0.1:38000"  # Optional
   ```

2. **Make sure SearxNG is running:**
   ```bash
   # Check if SearxNG is accessible
   curl http://127.0.0.1:38000
   ```

3. **Run the voice agent:**
   ```bash
   cd voice-ai-by-cursor
   python voice_ai_realtime.py
   ```

4. **Follow the prompts:**
   - Choose mode: `1` for Traditional or `2` for Omni
   - Enter a brand name when prompted (e.g., "Apple", "Tesla")
   - The agent will run deep research on the brand and load it as RAG context
   - Press Enter to start speaking
   - Say "quit" or "exit" to stop

## How It Works

1. **Brand Input**: When you enter a brand name, the agent triggers deep research
2. **Deep Research**: Uses LangGraph agent to search and compile brand background
3. **RAG Injection**: Research results are injected as context into the voice assistant
4. **Voice Interaction**: You can ask questions about the brand, and the assistant uses the research context

## Troubleshooting

- **"SearxNG connection failed"**: Make sure SearxNG is running on port 38000
- **"API key not set"**: Check your environment variables
- **"No module named 'deep_research'"**: Make sure you're running from the voice-agent directory or add it to PYTHONPATH
- **Audio issues**: Check microphone permissions and PyAudio installation

## Project Structure

```
voice-agent/
├── deep_research/          # Deep research agent code (copied from deep-research-agent repo)
│   ├── agent.py            # LangGraph workflow
│   ├── service.py          # Service wrapper
│   ├── common/             # Agent state definitions
│   ├── node/               # LangGraph nodes
│   └── ...
├── voice-ai-by-cursor/
│   └── voice_ai_realtime.py  # Main voice agent script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```
