#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
import logging
from pyannote.audio.pipelines.utils.hook import ProgressHook
from typing import List, Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def extract_word_timings(
    json_file_path: str, file_name: str = None, output_file_path: str = None
) -> list:
    """
    Extract word-level timing information from a transcription JSON file.

    This function reads a JSON file containing transcription data with word-level
    timing information and formats it according to the specified requirements.

    Args:
        json_file_path (str): Path to the transcription JSON file
        file_name (str, optional): Specific file name to extract data for
        output_file_path (str, optional): Path to save the formatted data

    Returns:
        list: List of dictionaries with text and timestamp information
    """
    try:
        # Read the JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize the result list
        result = []

        # Process each record in the file
        for record in data:
            # Skip if file_name is specified and doesn't match
            if file_name and record.get("file_name") != file_name:
                continue

            # Get the transcription data
            transcription = record.get("transcription", [{}])[0]

            # Extract words and timing information
            words = transcription.get("words", [])
            start_times = transcription.get("start_times", [])
            end_times = transcription.get("end_times", [])

            # Ensure we have the same number of words and timings
            if len(words) != len(start_times) or len(words) != len(end_times):
                logging.warning(
                    f"Mismatch in word and timing counts for {record.get('file_name', 'unknown')}"
                )
                continue

            # Process each word and its timing
            for i in range(len(words)):
                # Format the word with a leading space if needed
                word = words[i]
                if i > 0 and not word.startswith((".", ",", "!", "?", ":", ";")):
                    word = " " + word

                # Create the timestamp tuple
                timestamp = (start_times[i], end_times[i])

                # Add to the result
                result.append({"text": word, "timestamp": timestamp})

        # Save to file if output path is provided
        if output_file_path:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    except Exception as e:
        logging.error(f"Error processing file {json_file_path}: {str(e)}")
        return []


def find_best_match(
    diarization, start_time: float, end_time: float
) -> Optional[Tuple[float, float, str]]:
    """
    Find the best matching speaker segment for a given time range.

    Args:
        diarization: Pyannote diarization result
        start_time (float): Start time of the segment
        end_time (float): End time of the segment

    Returns:
        Optional[Tuple[float, float, str]]: Best matching segment (start, end, speaker) or None
    """
    best_match = None
    max_intersection = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_start = turn.start
        turn_end = turn.end

        # Calculate intersection manually
        intersection_start = max(start_time, turn_start)
        intersection_end = min(end_time, turn_end)

        if intersection_start < intersection_end:
            intersection_length = intersection_end - intersection_start
            if intersection_length > max_intersection:
                max_intersection = intersection_length
                best_match = (turn_start, turn_end, speaker)

    return best_match


def merge_consecutive_segments(
    segments: List[Tuple[str, float, float, str]],
) -> List[Tuple[str, float, float, str]]:
    """
    Merge consecutive segments from the same speaker.

    Args:
        segments (List[Tuple[str, float, float, str]]): List of segments (speaker, start, end, text)

    Returns:
        List[Tuple[str, float, float, str]]: Merged segments
    """
    merged_segments = []
    previous_segment = None

    for segment in segments:
        if previous_segment is None:
            previous_segment = segment
        else:
            if segment[0] == previous_segment[0]:
                # Merge segments of the same speaker that are consecutive
                previous_segment = (
                    previous_segment[0],
                    previous_segment[1],
                    segment[2],
                    previous_segment[3] + segment[3],
                )
            else:
                merged_segments.append(previous_segment)
                previous_segment = segment

    if previous_segment:
        merged_segments.append(previous_segment)

    return merged_segments


def get_last_segment(annotation) -> Any:
    """
    Get the last segment from a pyannote annotation.

    Args:
        annotation: Pyannote annotation

    Returns:
        Any: Last segment
    """
    last_segment = None
    for segment in annotation.itersegments():
        last_segment = segment
    return last_segment


def align(
    timings: List[Dict[str, Any]], diarization
) -> List[Tuple[str, float, float, str]]:
    """
    Align word timings with speaker diarization.

    Args:
        timings (List[Dict[str, Any]]): List of word timing dictionaries
        diarization: Pyannote diarization result

    Returns:
        List[Tuple[str, float, float, str]]: Aligned segments (speaker, start, end, text)
    """
    speaker_transcriptions = []

    # Find the end time of the last segment in diarization
    last_diarization_end = get_last_segment(diarization).end

    for chunk in timings:
        chunk_start = chunk["timestamp"][0]
        chunk_end = chunk["timestamp"][1]
        segment_text = chunk["text"]

        # Handle the case where chunk_end is None
        if chunk_end is None:
            # Use the end of the last diarization segment as the default end time
            chunk_end = (
                last_diarization_end
                if last_diarization_end is not None
                else chunk_start
            )

        # Find the best matching speaker segment
        best_match = find_best_match(diarization, chunk_start, chunk_end)
        if best_match:
            speaker = best_match[2]  # Extract the speaker label
            speaker_transcriptions.append(
                (speaker, chunk_start, chunk_end, segment_text)
            )

    # Merge consecutive segments of the same speaker
    speaker_transcriptions = merge_consecutive_segments(speaker_transcriptions)
    return speaker_transcriptions


def process_audio_file(
    audio_file: str,
    diarization_pipeline,
    transcription_file: str,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    Process a single audio file with diarization and alignment.

    Args:
        audio_file (str): Path to the audio file
        diarization_pipeline: Pyannote diarization pipeline
        transcription_file (str): Path to the transcription JSON file
        output_file (str, optional): Path to save the output JSON

    Returns:
        Dict[str, Any]: Processed file data
    """
    # Get file name from path
    file_name = os.path.basename(audio_file)

    # Run diarization
    with ProgressHook() as hook:
        diarization = diarization_pipeline(
            audio_file,
            hook=hook,
            num_speakers=2,  # Default to 2 speakers, can be adjusted
        )

    # Extract word timings from transcription
    word_timings = extract_word_timings(transcription_file, file_name)

    if not word_timings:
        print(f"Warning: No word timings found for {file_name}")
        return None

    # Align word timings with diarization
    aligned_segments = align(word_timings, diarization)

    # Convert aligned segments to the desired format
    transcription = []
    for speaker, start, end, text in aligned_segments:
        transcription.append(
            {"text": text, "author": speaker, "start": float(start), "end": float(end)}
        )

    # Create the output data structure
    result = {
        "file_path": audio_file,
        "file_name": file_name,
        "duration": get_last_segment(diarization).end,
        "transcription": transcription,
    }

    return result


def process_directory(
    audio_dir: str,
    transcription_file: str,
    output_file: str,
    auth_token: str,
    device: str = "cuda",
) -> None:
    """
    Process all audio files in a directory with diarization and alignment.

    Args:
        audio_dir (str): Directory containing audio files
        transcription_file (str): Path to the transcription JSON file
        output_file (str): Path to save the output JSON
        auth_token (str): Hugging Face authentication token for pyannote
        device (str, optional): Device to run diarization on (cuda or cpu)
    """
    # Initialize diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token,
    )
    diarization_pipeline.to(torch.device(device))

    # Get all audio files in the directory
    audio_files = [
        f for f in os.listdir(audio_dir) if f.endswith((".mp3", ".wav", ".ogg"))
    ]

    # Process each audio file
    results = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        full_path = os.path.join(audio_dir, audio_file)
        result = process_audio_file(full_path, diarization_pipeline, transcription_file)
        if result:
            results.append(result)

    # Save results to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)} files. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process audio files with speaker diarization and alignment"
    )
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument(
        "transcription_file", help="Path to the transcription JSON file"
    )
    parser.add_argument("output_file", help="Path to save the output JSON")
    parser.add_argument(
        "--auth_token",
        required=True,
        help="Hugging Face authentication token for pyannote",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run diarization on (cuda or cpu)"
    )

    args = parser.parse_args()

    process_directory(
        args.audio_dir,
        args.transcription_file,
        args.output_file,
        args.auth_token,
        args.device,
    )


if __name__ == "__main__":
    main()
