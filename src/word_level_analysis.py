import json
import re
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import pandas as pd
from tqdm import tqdm
import argparse

# Regex pattern for word tokenization (only word characters)
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def flatten(jd: Dict) -> Tuple[List[str], List[str]]:
    """Extract lowercase tokens and speakers from transcription data.

    Args:
        jd: Dictionary containing transcription data with 'transcription' key

    Returns:
        Tuple of (words, speakers) where:
        - words: List of lowercase tokens with punctuation removed
        - speakers: List of corresponding speaker labels
    """
    words, speakers = [], []

    for turn in jd["transcription"]:
        spk = turn["author"]
        # Convert text to lowercase and remove punctuation
        text = turn["text"].lower()
        for tok in WORD_PATTERN.findall(text):
            if not tok.strip():
                continue
            words.append(tok)
            speakers.append(spk)
    return words, speakers


def levenshtein_align(
    a: List[str], b: List[str]
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Compute alignment between two lists of words using Levenshtein distance.

    Args:
        a: First list of words (reference)
        b: Second list of words (hypothesis)

    Returns:
        List of (idx_a, idx_b) tuples representing alignment, where None indicates
        insertion/deletion
    """
    n, m = len(a), len(b)
    # dp cost matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    # Initialize first row and column
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    # Fill dp matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if a[i - 1] == b[j - 1] else 1
            best = dp[i - 1][j - 1] + sub_cost
            op = "M" if sub_cost == 0 else "S"

            if dp[i - 1][j] + 1 < best:
                best = dp[i - 1][j] + 1
                op = "D"
            if dp[i][j - 1] + 1 < best:
                best = dp[i][j - 1] + 1
                op = "I"

            dp[i][j] = best
            back[i][j] = op

    # Backtrack to get alignment
    align = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            align.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            align.append((i - 1, None))
            i -= 1
        elif op == "I":
            align.append((None, j - 1))
            j -= 1

    return list(reversed(align))


def analyze_word_differences(ref_json: Dict, hyp_json: Dict) -> Dict[str, Any]:
    """Analyze word-by-word differences between reference and hypothesis.

    Args:
        ref_json: Reference (ground truth) JSON data
        hyp_json: Hypothesis JSON data

    Returns:
        Dictionary containing:
        - alignment_details: List of dictionaries with word-by-word comparison
        - error_summary: Summary of substitution, deletion, and insertion errors
        - wer: Word Error Rate
    """
    # Extract words and speakers
    ref_w, ref_s = flatten(ref_json)
    hyp_w, hyp_s = flatten(hyp_json)

    # Skip empty files
    if not ref_w or not hyp_w:
        raise ValueError(
            f"Cannot analyze file: reference words: {len(ref_w)}, "
            f"hypothesis words: {len(hyp_w)}"
        )

    # Get alignment
    alignment = levenshtein_align(ref_w, hyp_w)

    # Analyze alignment
    alignment_details = []
    subs = dels = ins = 0

    for ia, ib in alignment:
        if ia is None:
            # Insertion
            alignment_details.append(
                {
                    "type": "INSERTION",
                    "ref_word": None,
                    "ref_speaker": None,
                    "hyp_word": hyp_w[ib],
                    "hyp_speaker": hyp_s[ib],
                }
            )
            ins += 1
        elif ib is None:
            # Deletion
            alignment_details.append(
                {
                    "type": "DELETION",
                    "ref_word": ref_w[ia],
                    "ref_speaker": ref_s[ia],
                    "hyp_word": None,
                    "hyp_speaker": None,
                }
            )
            dels += 1
        elif ref_w[ia] != hyp_w[ib]:
            # Substitution
            alignment_details.append(
                {
                    "type": "SUBSTITUTION",
                    "ref_word": ref_w[ia],
                    "ref_speaker": ref_s[ia],
                    "hyp_word": hyp_w[ib],
                    "hyp_speaker": hyp_s[ib],
                }
            )
            subs += 1
        else:
            # Match
            alignment_details.append(
                {
                    "type": "MATCH",
                    "ref_word": ref_w[ia],
                    "ref_speaker": ref_s[ia],
                    "hyp_word": hyp_w[ib],
                    "hyp_speaker": hyp_s[ib],
                }
            )

    # Calculate WER
    wer = (subs + dels + ins) / len(ref_w) if ref_w else 0.0

    # Create error summary
    error_summary = {
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "total_words": len(ref_w),
        "wer": wer,
    }

    return {"alignment_details": alignment_details, "error_summary": error_summary}


def analyze_files(stereo_file: str, mono_file: str) -> Dict[str, Any]:
    """Analyze differences between stereo and mono files.

    Args:
        stereo_file: Path to stereo (ground truth) JSON file
        mono_file: Path to mono (hypothesis) JSON file

    Returns:
        Dictionary containing analysis results for each file
    """
    # Load JSON files
    with open(stereo_file, "r", encoding="utf-8") as f:
        stereo_data = json.load(f)
    with open(mono_file, "r", encoding="utf-8") as f:
        mono_data = json.load(f)

    # Convert to dictionary format if input is list
    if isinstance(stereo_data, list):
        stereo_dict = {item["file_name"]: item for item in stereo_data}
    else:
        stereo_dict = stereo_data

    if isinstance(mono_data, list):
        mono_dict = {item["file_name"]: item for item in mono_data}
    else:
        mono_dict = mono_data

    # Analyze each file
    results = {}
    for file_name in tqdm(stereo_dict.keys()):
        if file_name in mono_dict:
            try:
                analysis = analyze_word_differences(
                    stereo_dict[file_name], mono_dict[file_name]
                )
                results[file_name] = analysis
            except ValueError as e:
                print(f"Skipping {file_name}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error analyzing {file_name}: {str(e)}")
                continue

    return results


def print_analysis(
    analysis_results: Dict[str, Any],
    file_name: Optional[str] = None,
    show_matches: bool = False,
):
    """Print detailed analysis results.

    Args:
        analysis_results: Results from analyze_files or analyze_word_differences
        file_name: Optional specific file to print results for
        show_matches: Whether to show matched words in the output
    """
    if file_name:
        if file_name not in analysis_results:
            print(f"File {file_name} not found in results")
            return
        results = {file_name: analysis_results[file_name]}
    else:
        results = analysis_results

    for fname, result in results.items():
        print(f"\nAnalysis for {fname}:")
        print("\nError Summary:")
        summary = result["error_summary"]
        print(f"Total Words: {summary['total_words']}")
        print(f"Substitutions: {summary['substitutions']}")
        print(f"Deletions: {summary['deletions']}")
        print(f"Insertions: {summary['insertions']}")
        print(f"WER: {summary['wer']:.4f}")

        print("\nDetailed Word-by-Word Analysis:")
        print("Type\t\tReference\t\tHypothesis")
        print("-" * 60)
        for detail in result["alignment_details"]:
            if not show_matches and detail["type"] == "MATCH":
                continue  # Skip matches unless show_matches is True
            ref = (
                f"{detail['ref_word']} ({detail['ref_speaker']})"
                if detail["ref_word"]
                else "N/A"
            )
            hyp = (
                f"{detail['hyp_word']} ({detail['hyp_speaker']})"
                if detail["hyp_word"]
                else "N/A"
            )
            print(f"{detail['type']:<12}\t{ref:<20}\t{hyp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze word-level differences between reference and hypothesis files"
    )
    parser.add_argument("reference", help="Path to reference JSON file")
    parser.add_argument("hypothesis", help="Path to hypothesis JSON file")
    parser.add_argument(
        "--show-matches", action="store_true", help="Show matched words in the output"
    )
    parser.add_argument("--file", help="Analyze specific file only")

    args = parser.parse_args()

    results = analyze_files(args.reference, args.hypothesis)
    print_analysis(results, file_name=args.file, show_matches=args.show_matches)
