# MCP Audio Server

A Model Context Protocol (MCP) server that provides audio transcription, intelligent splitting, and meeting analysis tools. This server exposes audio processing capabilities to MCP-compatible clients like Claude Desktop, enabling seamless integration of audio workflows into AI conversations.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that enables AI assistants to securely connect to external data sources and tools. This MCP server provides audio processing tools that can be used by any MCP-compatible client, allowing AI assistants to:

- Transcribe audio files directly in conversations
- Split large audio files for processing
- Generate meeting summaries and insights
- Analyze multiple audio transcripts simultaneously

## Features

- 🎙️ **Audio Transcription**: High-quality transcription using Groq's Whisper models
- ✂️ **Smart Audio Splitting**: Automatically split large audio files by size or duration
- 📝 **Transcript Summarization**: Generate comprehensive meeting summaries with context
- 📁 **Multi-file Analysis**: Chat with multiple transcript files simultaneously
- 🔄 **Format Fallbacks**: Robust export with MP3 → AAC → WAV fallback chain
- 📊 **Size Management**: Automatic handling of 25MB Groq API limits
- 🎯 **Intelligent Break Points**: Uses silence detection for optimal split points

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (for audio processing)
- Groq API key

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd mcp-audio-server
```

2. **Install with uv:**
```bash
uv sync
```

3. **Set up your Groq API key:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

4. **Verify installation:**
```bash
uv run python setup.py
```

## Usage

### With Claude Desktop

Add this server to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-audio-server": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_audio_server.server"],
      "cwd": "/path/to/mcp-audio-server",
      "env": {
        "GROQ_API_KEY": "your-groq-api-key-here"
      }
    }
  }
}
```

### With Other MCP Clients

Use the provided configuration file:

```bash
# Copy and edit the config
cp mcp-config.json my-config.json
# Edit my-config.json with your API key and paths
```

Then connect your MCP client using the configuration.

### Standalone Testing

Test the server directly:

```bash
# Start the MCP server
uv run python -m mcp_audio_server.server

# Or test with the CLI client
uv run mcp-audio-client transcribe path/to/audio.mp3
```

## MCP Tools

This server exposes the following tools to MCP clients:

### `transcribe_audio`
Transcribe audio files using Groq's Whisper API.

**Parameters:**
- `file_path` (string): Path to audio file
- `model` (string, optional): Groq model (default: "whisper-large-v3")
- `language` (string, optional): Audio language

**Example in Claude:**
> "Please transcribe the audio file at `/path/to/meeting.mp3`"

### `split_audio`
Split audio files with multiple strategies.

**Parameters:**
- `file_path` (string): Path to audio file
- `splits` (array, optional): Manual split points
- `output_dir` (string, optional): Output directory
- `max_size_mb` (number, optional): Max size for auto-splitting (default: 24MB)
- `max_duration_minutes` (number, optional): Max duration for auto-splitting
- `auto_split_by_size` (boolean): Enable size-based splitting (default: true)
- `auto_split_by_duration` (boolean): Enable duration-based splitting

**Example in Claude:**
> "Please split the large audio file at `/path/to/long_meeting.mp3` into segments under 25MB"

### `summarize_transcript`
Generate summaries from transcripts.

**Parameters:**
- `transcript` (string): Transcript text
- `context` (string, optional): Additional context
- `custom_prompt` (string, optional): Custom system prompt
- `model` (string, optional): Groq model (default: "llama3-8b-8192")

**Example in Claude:**
> "Please summarize this meeting transcript with context about our quarterly planning session"

### `multi_file_chat`
Analyze multiple files simultaneously.

**Parameters:**
- `file_paths` (array): List of file paths
- `question` (string): Question to ask
- `system_prompt` (string, optional): Custom system prompt
- `model` (string, optional): Groq model (default: "llama3-8b-8192")

**Example in Claude:**
> "Please analyze these three meeting transcripts and tell me what the common themes were"

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key (required for transcription/summarization)

### Supported Audio Formats
- MP3, M4A, WAV, FLAC, OGG, and more (via pydub)

### Export Formats
- **Primary**: MP3 (most compatible)
- **Fallback**: AAC (ADTS format)
- **Final Fallback**: WAV (uncompressed)

## Example Workflows

### Complete Meeting Processing
1. **Split large recording**: "Please split this 2-hour meeting recording into manageable segments"
2. **Transcribe segments**: "Now transcribe each segment"
3. **Generate summary**: "Create a comprehensive summary of all segments with action items"

### Multi-Meeting Analysis
1. **Transcribe multiple meetings**: Process several meeting recordings
2. **Cross-meeting analysis**: "What are the recurring themes across these three meetings?"
3. **Action item tracking**: "What action items were mentioned and who owns them?"

## Development

### Setup Development Environment
```bash
uv sync --dev
```

### Run Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black src/
uv run ruff check src/
```

### Testing the MCP Server
```bash
# Test server startup
uv run python -m mcp_audio_server.server

# Test with example client
uv run python examples/basic_usage.py
```

## Requirements

- Python 3.13+
- uv package manager
- ffmpeg (for audio processing)
- Groq API key

### Installing ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not set"**
   - Ensure your API key is exported: `export GROQ_API_KEY="your-key"`
   - For Claude Desktop, add it to the MCP configuration

2. **"ffmpeg not found"**
   - Install ffmpeg using the instructions above
   - Ensure it's in your system PATH

3. **"File too large" errors**
   - Use the `split_audio` tool first to break large files into <25MB segments
   - Then transcribe each segment individually

4. **MCP connection issues**
   - Verify the server path in your MCP configuration
   - Check that uv is installed and accessible
   - Ensure the working directory is correct

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with uv: `uv sync --dev`
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the MCP documentation: https://modelcontextprotocol.io/
- Review the examples in this repository

---

**Note**: This is an MCP server that requires a compatible MCP client (like Claude Desktop) to use. The server provides audio processing tools that integrate seamlessly into AI conversations.
