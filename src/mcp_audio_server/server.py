#!/usr/bin/env python3

import json
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
from fastmcp import FastMCP, Context
from pydub import AudioSegment

from .enums import AudioModels
from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-audio-server")

# Create FastMCP instance
mcp = FastMCP("mcp-audio-server")

@mcp.tool
async def transcribe_audio(
    file_path: str,
    model: str = AudioModels.WHISPER_LARGE_V3.value,
    language: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Transcribe audio file using Groq Whisper API
    
    Args:
        file_path: Path to the audio file to transcribe
        model: Groq model to use (default: whisper-large-v3)
        language: Language of the audio (optional)
    """
    if ctx:
        await ctx.info("Starting audio transcription...")
    
    if not settings.groq_client:
        if ctx:
            await ctx.error("Groq client not configured. Please set GROQ_API_KEY environment variable.")
        return json.dumps({
            "error": "Groq client not configured. Please set GROQ_API_KEY environment variable."
        })
    
    if not file_path:
        if ctx:
            await ctx.error("file_path parameter is required")
        return json.dumps({"error": "file_path is required"})
    
    try:
        audio_file_path = Path(file_path)
        if not audio_file_path.exists():
            if ctx:
                await ctx.error(f"File not found: {file_path}")
            return json.dumps({"error": f"File not found: {file_path}"})

        # Check file size (25MB limit for Groq)
        file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            error_msg = (
                f"Audio file is too large ({file_size_mb:.2f} MB). "
                "The file should be less than 25 MB. "
                "Please consider splitting the audio file and then generate transcript per split."
            )
            logger.error(error_msg)
            if ctx:
                await ctx.error(error_msg)
            return json.dumps({"error": error_msg})
        
        if ctx:
            await ctx.info(f"📁 File: {audio_file_path.name}")
            await ctx.info(f"📊 Size: {file_size_mb:.2f} MB")
            await ctx.info(f"🤖 Model: {model}")
            await ctx.info("🎙️ Starting transcription...")
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        # Prepare transcription parameters
        transcription_params = {
            "file": audio_file_path,
            "model": model,
            "response_format": "text"
        }

        if language:
            transcription_params["language"] = language
            if ctx:
                await ctx.debug(f"Using specified language: {language}")

        # Make async call to Groq API
        if ctx:
            await ctx.info("🔄 Calling Groq API for transcription...")
        transcription = await settings.groq_client.audio.transcriptions.create(**transcription_params)
            
        if ctx:
            await ctx.info("✅ Transcription completed successfully!")
            
        result = {
            "transcript": transcription,
            "file_path": str(file_path),
            "model_used": model,
            "language": language or "auto-detected"
        }
        
        if ctx:
            transcript_length = len(transcription) if transcription else 0
            await ctx.info(f"📝 Transcript length: {transcript_length} characters")
            
        return json.dumps(result, indent=2)
        
    except Exception:
        error_msg = f"Error transcribing audio: {traceback.format_exc()}"
        logger.error(error_msg)
        if ctx:
            await ctx.error(f"❌ Transcription failed: {str(error_msg)}")
        return json.dumps({"error": error_msg})

@mcp.tool
async def split_audio(
    file_path: str,
    splits: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    max_size_mb: Optional[float] = 24.0,  # Keep under 25MB for Groq
    max_duration_minutes: Optional[float] = None,
    auto_split_by_size: bool = True,
    auto_split_by_duration: bool = False,
    ctx: Context = None
) -> str:
    """
    Split audio file based on timestamps, file size limits, or duration limits using pydub
    
    Args:
        file_path: Path to the audio file to split
        splits: Optional list of manual split points in format [{"start": 0, "end": 2400, "name": "part1"}, ...]
               Times in seconds. If no end specified, splits to end of file.
        output_dir: Directory to save split files (optional, uses temp dir if not specified)
        max_size_mb: Maximum file size in MB for auto-splitting (default: 24MB for Groq compatibility)
        max_duration_minutes: Maximum duration in minutes for auto-splitting
        auto_split_by_size: Enable automatic splitting by file size
        auto_split_by_duration: Enable automatic splitting by duration
    """
    if ctx:
        await ctx.info("Starting enhanced audio splitting process...")
    
    if not file_path:
        if ctx:
            await ctx.error("file_path parameter is required")
        return json.dumps({"error": "file_path is required"})
    
    try:
        audio_file_path = Path(file_path)
        if not audio_file_path.exists():
            if ctx:
                await ctx.error(f"File not found: {file_path}")
            return json.dumps({"error": f"File not found: {file_path}"})
        
        # Get original file size
        original_size_mb = audio_file_path.stat().st_size / (1024 * 1024)
        
        if ctx:
            await ctx.info(f"📁 Loading audio file: {audio_file_path.name}")
            await ctx.info(f"📊 Original file size: {original_size_mb:.2f} MB")
        
        # Load audio file
        audio = AudioSegment.from_file(str(audio_file_path))
        total_duration = len(audio) / 1000.0  # Convert to seconds
        total_duration_min = total_duration / 60.0
        
        if ctx:
            await ctx.info(f"⏱️ Total audio duration: {total_duration:.2f} seconds ({total_duration_min:.2f} minutes)")
        
        # Determine splitting strategy
        split_results = []
        
        if splits:
            # Manual splits provided
            if ctx:
                await ctx.info(f"🔪 Using manual splits: {len(splits)} segments specified")
            split_results = await _process_manual_splits(audio, splits, total_duration, ctx)
            
        elif auto_split_by_size and original_size_mb > max_size_mb:
            # Auto-split by file size
            if ctx:
                await ctx.info(f"📏 Auto-splitting by size: {original_size_mb:.2f}MB > {max_size_mb}MB limit")
            split_results = await _process_auto_split_by_size(audio, max_size_mb, audio_file_path, ctx)
            
        elif auto_split_by_duration and max_duration_minutes and total_duration_min > max_duration_minutes:
            # Auto-split by duration
            if ctx:
                await ctx.info(f"⏲️ Auto-splitting by duration: {total_duration_min:.2f}min > {max_duration_minutes}min limit")
            split_results = await _process_auto_split_by_duration(audio, max_duration_minutes, ctx)
            
        else:
            # No splitting needed or requested
            if ctx:
                await ctx.info("✅ No splitting required - file meets all criteria")
            return json.dumps({
                "message": "No splitting required",
                "original_file": str(file_path),
                "original_duration": total_duration,
                "original_size_mb": original_size_mb,
                "splits": []
            }, indent=2)
        
        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            if ctx:
                await ctx.info(f"📂 Using specified output directory: {output_dir}")
        else:
            output_path = Path(tempfile.mkdtemp())
            if ctx:
                await ctx.info(f"📂 Using temporary output directory: {output_path}")
        
        # Export all segments with format fallbacks
        exported_splits = []
        for i, split_info in enumerate(split_results):
            segment = split_info["segment"]
            name = split_info["name"]
            start_time = split_info["start_time"]
            end_time = split_info["end_time"]
            
            if ctx:
                await ctx.info(f"💾 Exporting segment {i+1}/{len(split_results)}: {name}")
            
            # Try to export with format fallbacks
            output_file = await _export_segment_with_fallbacks(
                segment, name, output_path, start_time, end_time, ctx
            )
            
            if not output_file:
                error_msg = f"Failed to export segment {name}"
                if ctx:
                    await ctx.error(f"❌ {error_msg}")
                return json.dumps({"error": error_msg})
            
            # Get exported file size
            exported_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            if ctx:
                await ctx.debug(f"Saved {name}: {exported_size_mb:.2f}MB, {end_time - start_time:.2f}s")
            
            exported_splits.append({
                "name": name,
                "file_path": str(output_file),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "size_mb": exported_size_mb
            })
        
        if ctx:
            total_exported_size = sum(s["size_mb"] for s in exported_splits)
            await ctx.info(f"✅ Audio splitting completed! Created {len(exported_splits)} segments")
            await ctx.info(f"📊 Total exported size: {total_exported_size:.2f}MB")
            
        result = {
            "original_file": str(file_path),
            "original_duration": total_duration,
            "original_size_mb": original_size_mb,
            "output_directory": str(output_path),
            "split_strategy": _determine_split_strategy(splits, auto_split_by_size, auto_split_by_duration, original_size_mb, max_size_mb, total_duration_min, max_duration_minutes),
            "splits": exported_splits
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error splitting audio: {str(e)}"
        logger.error(error_msg)
        if ctx:
            await ctx.error(f"❌ Audio splitting failed: {error_msg}")
        return json.dumps({"error": error_msg})

@mcp.tool
async def summarize_transcript(
    transcript: str,
    context: str = "",
    custom_prompt: Optional[str] = None,
    model: str = "llama3-8b-8192",
    ctx: Context = None
) -> str:
    """
    Summarize transcript with custom context and system prompts
    
    Args:
        transcript: The transcript text to summarize
        context: Additional context about the transcript (e.g., "Call between lead data engineer and junior engineer about database optimization")
        custom_prompt: Custom system prompt for summarization (optional)
        model: Groq model to use for summarization (default: llama3-8b-8192)
    """
    if ctx:
        await ctx.info("Starting transcript summarization...")
    
    if not settings.groq_client:
        if ctx:
            await ctx.error("Groq API key not configured. Please set GROQ_API_KEY environment variable.")
        return json.dumps({
            "error": "Groq API key not configured. Please set GROQ_API_KEY environment variable."
        })
    
    if not transcript:
        if ctx:
            await ctx.error("transcript parameter is required")
        return json.dumps({"error": "transcript is required"})
    
    try:
        transcript_length = len(transcript)
        if ctx:
            await ctx.info(f"📝 Transcript length: {transcript_length} characters")
            await ctx.info(f"🤖 Using model: {model}")
            if context:
                await ctx.info(f"📋 Context provided: {context[:100]}...")
            if custom_prompt:
                await ctx.info("🎯 Using custom prompt")
        
        # Default system prompt
        default_system_prompt = """You are an expert at analyzing and summarizing meeting transcripts. 
        Please provide a comprehensive summary that includes:
        1. Key topics discussed
        2. Important decisions made
        3. Action items and next steps
        4. Key insights and takeaways
        5. Notable quotes or important statements
        
        Format your response in a clear, structured manner.
        and add transcript source in the end of the summary.
        """
        
        system_prompt = custom_prompt if custom_prompt else default_system_prompt
        
        # Add context to the user message if provided
        user_message = f"Context: {context}\n\nTranscript to summarize:\n{transcript}" if context else f"Transcript to summarize:\n{transcript}"
        
        if ctx:
            await ctx.info("🔄 Calling Groq API for summarization...")
        
        # Create chat completion
        chat_completion = await settings.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=model,
            temperature=0.3,
        )
        
        summary = chat_completion.choices[0].message.content
        
        if ctx:
            summary_length = len(summary) if summary else 0
            await ctx.info(f"✅ Summarization completed! Summary length: {summary_length} characters")
        
        result = {
            "summary": summary,
            "context_used": context,
            "model_used": model,
            "custom_prompt_used": bool(custom_prompt)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error summarizing transcript: {str(e)}"
        logger.error(error_msg)
        if ctx:
            await ctx.error(f"❌ Summarization failed: {error_msg}")
        return json.dumps({"error": error_msg})

@mcp.tool
async def multi_file_chat(
    file_paths: List[str],
    question: str,
    system_prompt: Optional[str] = None,
    model: str = "llama3-8b-8192",
    ctx: Context = None
) -> str:
    """
    Chat with multiple files using Groq
    
    Args:
        file_paths: List of file paths to include in the chat
        question: Question to ask about the files
        system_prompt: Custom system prompt (optional)
        model: Groq model to use (default: llama3-8b-8192)
    """
    if ctx:
        await ctx.info("Starting multi-file chat analysis...")
    
    if not settings.groq_client:
        if ctx:
            await ctx.error("Groq API key not configured. Please set GROQ_API_KEY environment variable.")
        return json.dumps({
            "error": "Groq API key not configured. Please set GROQ_API_KEY environment variable."
        })
    
    if not file_paths:
        if ctx:
            await ctx.error("file_paths list parameter is required")
        return json.dumps({"error": "file_paths list is required"})
    
    if not question:
        if ctx:
            await ctx.error("question parameter is required")
        return json.dumps({"error": "question is required"})
    
    try:
        if ctx:
            await ctx.info(f"📁 Processing {len(file_paths)} files")
            await ctx.info(f"❓ Question: {question[:100]}...")
            await ctx.info(f"🤖 Using model: {model}")
            if system_prompt:
                await ctx.info("🎯 Using custom system prompt")
        
        # Read all files
        file_contents = []
        for i, file_path in enumerate(file_paths):
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                if ctx:
                    await ctx.error(f"File not found: {file_path}")
                return json.dumps({"error": f"File not found: {file_path}"})
            
            if ctx:
                await ctx.info(f"📖 Reading file {i+1}/{len(file_paths)}: {file_path_obj.name}")
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                if ctx:
                    await ctx.debug(f"File {file_path_obj.name}: {len(content)} characters")
                file_contents.append({
                    "file_path": file_path,
                    "content": content
                })
        
        # Default system prompt for multi-file analysis
        default_system_prompt = """You are an expert analyst capable of reading and understanding multiple documents simultaneously. 
        Analyze the provided files and answer questions based on their contents. 
        Always cite which file(s) your information comes from when providing answers.
        Be comprehensive and accurate in your responses."""
        
        system_message = system_prompt if system_prompt else default_system_prompt
        
        # Construct user message with file contents
        user_message = f"Question: {question}\n\nFiles to analyze:\n"
        for file_info in file_contents:
            user_message += f"\n--- File: {file_info['file_path']} ---\n"
            user_message += file_info['content']
            user_message += "\n--- End of file ---\n"
        
        if ctx:
            total_chars = sum(len(f['content']) for f in file_contents)
            await ctx.info(f"📊 Total content: {total_chars} characters across {len(file_contents)} files")
            await ctx.info("🔄 Calling Groq API for analysis...")
        
        # Create chat completion
        chat_completion = await settings.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model=model,
            temperature=0.3,
            max_tokens=2048
        )
        
        answer = chat_completion.choices[0].message.content
        
        if ctx:
            answer_length = len(answer) if answer else 0
            await ctx.info(f"✅ Multi-file analysis completed! Answer length: {answer_length} characters")
        
        result = {
            "answer": answer,
            "files_analyzed": [f["file_path"] for f in file_contents],
            "question": question,
            "model_used": model,
            "custom_system_prompt_used": bool(system_prompt)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error in multi-file chat: {str(e)}"
        logger.error(error_msg)
        if ctx:
            await ctx.error(f"❌ Multi-file chat failed: {error_msg}")
        return json.dumps({"error": error_msg})


# Helper functions for audio splitting

async def _process_manual_splits(audio: AudioSegment, splits: List[Dict[str, Any]], total_duration: float, ctx: Context = None) -> List[Dict[str, Any]]:
    """Process manually specified splits"""
    split_results = []
    
    for i, split_info in enumerate(splits):
        start_sec = split_info.get("start", 0)
        end_sec = split_info.get("end", total_duration)
        name = split_info.get("name", f"manual_split_{i+1}")
        
        if ctx:
            await ctx.info(f"✂️ Processing manual split {i+1}/{len(splits)}: {name} ({start_sec}s - {end_sec}s)")
        
        # Convert seconds to milliseconds
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        split_results.append({
            "segment": segment,
            "name": name,
            "start_time": start_sec,
            "end_time": end_sec
        })
    
    return split_results


async def _process_auto_split_by_size(audio: AudioSegment, max_size_mb: float, original_file_path: Path, ctx: Context = None) -> List[Dict[str, Any]]:
    """Auto-split audio to keep segments under specified size"""
    split_results = []
    
    # Estimate bitrate from original file
    original_size_bytes = original_file_path.stat().st_size
    audio_duration_sec = len(audio) / 1000.0
    estimated_bitrate = (original_size_bytes * 8) / audio_duration_sec  # bits per second
    
    # Calculate target duration for each segment to stay under size limit
    max_size_bytes = max_size_mb * 1024 * 1024
    target_duration_sec = (max_size_bytes * 8) / estimated_bitrate
    target_duration_sec = min(target_duration_sec * 0.9, audio_duration_sec)  # 90% safety margin
    
    if ctx:
        await ctx.debug(f"Estimated bitrate: {estimated_bitrate/1000:.2f} kbps")
        await ctx.debug(f"Target segment duration: {target_duration_sec:.2f} seconds")
    
    current_start = 0
    segment_num = 1
    
    while current_start < audio_duration_sec:
        # Calculate end time for this segment
        current_end = min(current_start + target_duration_sec, audio_duration_sec)
        
        # Try to find a good break point (silence) near the target end
        if current_end < audio_duration_sec:
            current_end = await _find_good_break_point(audio, current_start, current_end, ctx)
        
        name = f"auto_size_part_{segment_num:03d}"
        
        if ctx:
            await ctx.info(f"🔄 Creating size-based segment {segment_num}: {current_start:.2f}s - {current_end:.2f}s")
        
        # Convert to milliseconds and extract segment
        start_ms = int(current_start * 1000)
        end_ms = int(current_end * 1000)
        segment = audio[start_ms:end_ms]
        
        split_results.append({
            "segment": segment,
            "name": name,
            "start_time": current_start,
            "end_time": current_end
        })
        
        current_start = current_end
        segment_num += 1
    
    return split_results


async def _process_auto_split_by_duration(audio: AudioSegment, max_duration_minutes: float, ctx: Context = None) -> List[Dict[str, Any]]:
    """Auto-split audio by duration"""
    split_results = []
    
    max_duration_sec = max_duration_minutes * 60
    audio_duration_sec = len(audio) / 1000.0
    
    current_start = 0
    segment_num = 1
    
    while current_start < audio_duration_sec:
        current_end = min(current_start + max_duration_sec, audio_duration_sec)
        
        # Try to find a good break point near the target end
        if current_end < audio_duration_sec:
            current_end = await _find_good_break_point(audio, current_start, current_end, ctx)
        
        name = f"auto_duration_part_{segment_num:03d}"
        
        if ctx:
            await ctx.info(f"🕐 Creating duration-based segment {segment_num}: {current_start:.2f}s - {current_end:.2f}s")
        
        # Convert to milliseconds and extract segment
        start_ms = int(current_start * 1000)
        end_ms = int(current_end * 1000)
        segment = audio[start_ms:end_ms]
        
        split_results.append({
            "segment": segment,
            "name": name,
            "start_time": current_start,
            "end_time": current_end
        })
        
        current_start = current_end
        segment_num += 1
    
    return split_results


async def _export_segment_with_fallbacks(
    segment: AudioSegment, 
    name: str, 
    output_path: Path, 
    start_time: float, 
    end_time: float, 
    ctx: Context = None
) -> Optional[Path]:
    """
    Export audio segment with format fallbacks for maximum compatibility
    
    Args:
        segment: AudioSegment to export
        name: Base name for the file
        output_path: Directory to save the file
        start_time: Start time in seconds (for bitrate calculation)
        end_time: End time in seconds (for bitrate calculation)
        ctx: Context for logging
    
    Returns:
        Path to exported file or None if all formats failed
    """
    duration_minutes = (end_time - start_time) / 60.0
    target_size_mb = 24.0  # Keep under 25MB
    
    # Calculate target bitrate to stay under size limit
    target_bitrate = min(int((target_size_mb * 8 * 1024) / duration_minutes), 128)  # Cap at 128kbps for quality
    
    # Format fallback chain: MP3 -> AAC -> WAV
    export_formats = [
        {"format": "mp3", "extension": "mp3", "params": {"bitrate": f"{target_bitrate}k"}},
        {"format": "adts", "extension": "aac", "params": {"bitrate": f"{target_bitrate}k"}},  # AAC format
        {"format": "wav", "extension": "wav", "params": {}}  # Uncompressed fallback
    ]
    
    for format_info in export_formats:
        output_file = output_path / f"{name}.{format_info['extension']}"
        
        try:
            if ctx:
                await ctx.debug(f"Trying {format_info['format']} format for {name}")
            
            # Export with specified format and parameters
            segment.export(str(output_file), format=format_info["format"], **format_info["params"])
            
            # Verify the file was created successfully
            if output_file.exists() and output_file.stat().st_size > 0:
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                if ctx:
                    await ctx.info(f"✅ Exported as {format_info['format'].upper()}: {file_size_mb:.2f}MB")
                return output_file
            else:
                if ctx:
                    await ctx.debug(f"Failed to create {format_info['format']} file")
                
        except Exception as e:
            if ctx:
                await ctx.debug(f"Export failed with {format_info['format']}: {str(e)}")
            
            # Clean up failed file if it exists
            if output_file.exists():
                try:
                    output_file.unlink()
                except:
                    pass
    
    # All formats failed
    if ctx:
        await ctx.error(f"All export formats failed for segment {name}")
    return None


async def _find_good_break_point(audio: AudioSegment, start_time: float, target_end: float, ctx: Context = None, search_window: float = 10.0) -> float:
    """
    Find a good break point near the target end time by looking for silence
    
    Args:
        audio: AudioSegment to analyze
        start_time: Start time in seconds
        target_end: Target end time in seconds
        search_window: Window in seconds to search for silence around target_end
    """
    # Search window around target end
    search_start = max(target_end - search_window/2, start_time)
    search_end = min(target_end + search_window/2, len(audio)/1000.0)
    
    # Convert to milliseconds
    search_start_ms = int(search_start * 1000)
    search_end_ms = int(search_end * 1000)
    target_end_ms = int(target_end * 1000)
    
    # Extract search segment
    search_segment = audio[search_start_ms:search_end_ms]
    
    # Find the quietest point in the search window
    min_volume = float('inf')
    best_break_ms = target_end_ms
    
    # Sample every 100ms within the search window
    for ms in range(0, len(search_segment), 100):
        if ms + 100 <= len(search_segment):
            sample = search_segment[ms:ms+100]
            volume = sample.dBFS
            if volume < min_volume:
                min_volume = volume
                best_break_ms = search_start_ms + ms
    
    best_break_sec = best_break_ms / 1000.0
    
    if ctx and abs(best_break_sec - target_end) > 1.0:  # Only log if we found a significantly different break point
        await ctx.debug(f"Found better break point at {best_break_sec:.2f}s (vs target {target_end:.2f}s)")
    
    return best_break_sec


def _determine_split_strategy(splits, auto_split_by_size, auto_split_by_duration, original_size_mb, max_size_mb, total_duration_min, max_duration_minutes):
    """Determine which splitting strategy was used"""
    if splits:
        return "manual"
    elif auto_split_by_size and original_size_mb > max_size_mb:
        return f"auto_size_{max_size_mb}MB"
    elif auto_split_by_duration and max_duration_minutes and total_duration_min > max_duration_minutes:
        return f"auto_duration_{max_duration_minutes}min"
    else:
        return "none_required"


def main():
    """Main entry point"""
    logger.info("Starting FastMCP Audio Server...")
    mcp.run()


if __name__ == "__main__":
    main() 