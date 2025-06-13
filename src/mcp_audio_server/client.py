#!/usr/bin/env python3

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from fastmcp import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-audio-client")

class AudioMCPClient:
    def __init__(self, server_command: List[str]):
        """Initialize client with server command"""
        # FastMCP Client can infer transport from command list
        # For stdio servers using the MCP config format
        config = {
            "mcpServers": {
                "audio-server": {
                    "command": server_command[0],
                    "args": server_command[1:] if len(server_command) > 1 else []
                }
            }
        }
        logger.info(f"Config: {config}")
        self.client = Client(config)
        self._connected = False
    
    async def connect(self):
        """Connect to the MCP server using async context manager"""
        try:
            logger.info(f"Connecting to MCP server...")
            # FastMCP Client uses async context manager for connection
            self._connected = True
            logger.info("Successfully connected to MCP server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self._connected = False
            return False
    
    def is_connected(self):
        """Check if client is connected"""
        return self._connected
    
    async def list_tools(self):
        """List available tools from the server"""
        async with self.client:
            try:
                tools = await self.client.list_tools()
                return tools
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                return []
    
    async def transcribe_audio(self, file_path: str, model: str = "whisper-large-v3", language: Optional[str] = None):
        """Transcribe audio file"""
        async with self.client:
            arguments = {
                "file_path": file_path,
                "model": model
            }
            
            if language:
                arguments["language"] = language
            
            try:
                result = await self.client.call_tool("transcribe_audio", arguments)
                # FastMCP returns content objects, get the text from first content
                if result and len(result) > 0 and hasattr(result[0], 'text'):
                    return json.loads(result[0].text)
                return {"error": "No result returned"}
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                return {"error": str(e)}
    
    async def split_audio(self, file_path: str, splits: List[Dict[str, Any]], output_dir: Optional[str] = None):
        """Split audio file based on timestamps"""
        async with self.client:
            arguments = {
                "file_path": file_path,
                "splits": splits
            }
            
            if output_dir:
                arguments["output_dir"] = output_dir
            
            try:
                result = await self.client.call_tool("split_audio", arguments)
                if result and len(result) > 0 and hasattr(result[0], 'text'):
                    return json.loads(result[0].text)
                return {"error": "No result returned"}
            except Exception as e:
                logger.error(f"Error splitting audio: {e}")
                return {"error": str(e)}
    
    async def summarize_transcript(self, transcript: str, context: str = "", custom_prompt: Optional[str] = None, model: str = "llama3-8b-8192"):
        """Summarize transcript with custom context"""
        async with self.client:
            arguments = {
                "transcript": transcript,
                "context": context,
                "model": model
            }
            
            if custom_prompt:
                arguments["custom_prompt"] = custom_prompt
            
            try:
                result = await self.client.call_tool("summarize_transcript", arguments)
                if result and len(result) > 0 and hasattr(result[0], 'text'):
                    return json.loads(result[0].text)
                return {"error": "No result returned"}
            except Exception as e:
                logger.error(f"Error summarizing transcript: {e}")
                return {"error": str(e)}
    
    async def multi_file_chat(self, file_paths: List[str], question: str, system_prompt: Optional[str] = None, model: str = "llama3-8b-8192"):
        """Chat with multiple files"""
        async with self.client:
            arguments = {
                "file_paths": file_paths,
                "question": question,
                "model": model
            }
            
            if system_prompt:
                arguments["system_prompt"] = system_prompt
            
            try:
                result = await self.client.call_tool("multi_file_chat", arguments)
                if result and len(result) > 0 and hasattr(result[0], 'text'):
                    return json.loads(result[0].text)
                return {"error": "No result returned"}
            except Exception as e:
                logger.error(f"Error in multi-file chat: {e}")
                return {"error": str(e)}
    
    async def close(self):
        """Close connection to server"""
        # FastMCP Client handles cleanup automatically in context manager
        self._connected = False

# CLI Commands
@click.group()
@click.option('--server-command', default=None, help='Command to start the MCP server')
@click.pass_context
def cli(ctx, server_command):
    """MCP Audio Client CLI"""
    ctx.ensure_object(dict)
    ctx.obj['server_command'] = server_command or ['python', '-m', 'mcp_audio_server.server']
    logger.info(f"Server command: {ctx.obj['server_command']}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--model', default='whisper-large-v3', help='Groq model to use for transcription')
@click.option('--language', default=None, help='Language of the audio')
@click.option('--output', '-o', type=click.Path(), help='Output file for transcript')
@click.pass_context
def transcribe(ctx, file_path, model, language, output):
    """Transcribe an audio file using Groq"""
    async def run():
        client = AudioMCPClient(ctx.obj['server_command'])
        try:
            success = await client.connect()
            if not success:
                click.echo("Failed to connect to MCP server", err=True)
                return
            
            click.echo(f"Transcribing {file_path}...")
            result = await client.transcribe_audio(file_path, model, language)
            
            if "error" in result:
                click.echo(f"Error: {result['error']}", err=True)
                return
            
            click.echo("\n=== Transcription Result ===")
            click.echo(f"File: {result['file_path']}")
            click.echo(f"Model: {result['model_used']}")
            click.echo(f"Language: {result['language']}")
            click.echo("\n=== Transcript ===")
            click.echo(result['transcript'])
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"\nTranscript saved to {output}")
            
        finally:
            await client.close()
    
    asyncio.run(run())

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--splits', required=True, help='JSON string with split information: [{"start": 0, "end": 2400, "name": "part1"}, ...]')
@click.option('--output-dir', type=click.Path(), help='Output directory for split files')
@click.pass_context
def split(ctx, file_path, splits, output_dir):
    """Split an audio file based on timestamps"""
    async def run():
        client = AudioMCPClient(ctx.obj['server_command'])
        try:
            success = await client.connect()
            if not success:
                click.echo("Failed to connect to MCP server", err=True)
                return
            
            try:
                splits_data = json.loads(splits)
            except json.JSONDecodeError as e:
                click.echo(f"Error parsing splits JSON: {e}", err=True)
                return
            
            click.echo(f"Splitting {file_path}...")
            result = await client.split_audio(file_path, splits_data, output_dir)
            
            if "error" in result:
                click.echo(f"Error: {result['error']}", err=True)
                return
            
            click.echo("\n=== Split Result ===")
            click.echo(f"Original file: {result['original_file']}")
            click.echo(f"Original duration: {result['original_duration']:.2f} seconds")
            click.echo(f"Output directory: {result['output_directory']}")
            click.echo("\n=== Split Files ===")
            for split_info in result['splits']:
                click.echo(f"- {split_info['name']}: {split_info['file_path']}")
                click.echo(f"  Duration: {split_info['duration']:.2f}s ({split_info['start_time']:.2f}s - {split_info['end_time']:.2f}s)")
            
        finally:
            await client.close()
    
    asyncio.run(run())

@cli.command()
@click.option('--transcript-file', type=click.Path(exists=True), help='File containing the transcript')
@click.option('--transcript-text', help='Transcript text directly')
@click.option('--context', default='', help='Additional context about the transcript')
@click.option('--custom-prompt', help='Custom system prompt for summarization')
@click.option('--model', default='llama3-8b-8192', help='Groq model to use')
@click.option('--output', '-o', type=click.Path(), help='Output file for summary')
@click.pass_context
def summarize(ctx, transcript_file, transcript_text, context, custom_prompt, model, output):
    """Summarize a transcript with custom context"""
    async def run():
        client = AudioMCPClient(ctx.obj['server_command'])
        try:
            success = await client.connect()
            if not success:
                click.echo("Failed to connect to MCP server", err=True)
                return
            
            # Get transcript text
            if transcript_file:
                with open(transcript_file, 'r') as f:
                    transcript_content = f.read()
            elif transcript_text:
                transcript_content = transcript_text
            else:
                click.echo("Either --transcript-file or --transcript-text must be provided", err=True)
                return
            
            click.echo("Summarizing transcript...")
            result = await client.summarize_transcript(transcript_content, context, custom_prompt, model)
            
            if "error" in result:
                click.echo(f"Error: {result['error']}", err=True)
                return
            
            click.echo("\n=== Summary Result ===")
            click.echo(f"Model: {result['model_used']}")
            click.echo(f"Context: {result['context_used']}")
            click.echo(f"Custom prompt used: {result['custom_prompt_used']}")
            click.echo("\n=== Summary ===")
            click.echo(result['summary'])
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"\nSummary saved to {output}")
            
        finally:
            await client.close()
    
    asyncio.run(run())

@cli.command()
@click.option('--files', required=True, help='Comma-separated list of file paths')
@click.option('--question', required=True, help='Question to ask about the files')
@click.option('--system-prompt', help='Custom system prompt')
@click.option('--model', default='llama3-8b-8192', help='Groq model to use')
@click.option('--output', '-o', type=click.Path(), help='Output file for response')
@click.pass_context
def chat(ctx, files, question, system_prompt, model, output):
    """Chat with multiple files"""
    async def run():
        client = AudioMCPClient(ctx.obj['server_command'])
        try:
            success = await client.connect()
            if not success:
                click.echo("Failed to connect to MCP server", err=True)
                return
            
            file_paths = [f.strip() for f in files.split(',')]
            
            # Validate files exist
            for file_path in file_paths:
                if not Path(file_path).exists():
                    click.echo(f"File not found: {file_path}", err=True)
                    return
            
            click.echo(f"Analyzing {len(file_paths)} files...")
            result = await client.multi_file_chat(file_paths, question, system_prompt, model)
            
            if "error" in result:
                click.echo(f"Error: {result['error']}", err=True)
                return
            
            click.echo("\n=== Chat Result ===")
            click.echo(f"Files analyzed: {', '.join(result['files_analyzed'])}")
            click.echo(f"Model: {result['model_used']}")
            click.echo(f"Question: {result['question']}")
            click.echo(f"Custom system prompt used: {result['custom_system_prompt_used']}")
            click.echo("\n=== Answer ===")
            click.echo(result['answer'])
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"\nResponse saved to {output}")
            
        finally:
            await client.close()
    
    asyncio.run(run())

@cli.command()
@click.pass_context
def list_tools(ctx):
    """List available tools from the MCP server"""
    async def run():
        client = AudioMCPClient(ctx.obj['server_command'])
        try:
            success = await client.connect()
            if not success:
                click.echo("Failed to connect to MCP server", err=True)
                return
            
            tools = await client.list_tools()
            
            click.echo("\n=== Available Tools ===")
            for tool in tools:
                click.echo(f"\n- {tool.name}")
                click.echo(f"  Description: {tool.description}")
                if hasattr(tool, 'inputSchema'):
                    click.echo(f"  Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
                elif hasattr(tool, 'input_schema'):
                    click.echo(f"  Input Schema: {json.dumps(tool.input_schema, indent=2)}")
                else:
                    click.echo("  Input Schema: Not available")
            
        finally:
            await client.close()
    
    asyncio.run(run())

def main():
    """Main entry point"""
    cli()

if __name__ == "__main__":
    main() 