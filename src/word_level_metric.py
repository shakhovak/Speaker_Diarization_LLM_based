import json
import re
import sys
from collections import Counter
from typing import List, Tuple, Optional, Dict
import pandas as pd
import logging
import asyncio
import nest_asyncio
from gpt_engine_evaluation import gpt_engine_diarization_eval
from tqdm import tqdm

# Try to import scipy for advanced metrics
try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# Import DER calculation functions
try:
    from timestamp_metric import calculate_der as calculate_der_metric

    DER_AVAILABLE = True
except ImportError:
    DER_AVAILABLE = False

# Regex pattern for word tokenization (only word characters)
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def micro(avg_df: pd.DataFrame, metric: str) -> float:
    """Calculate micro average for a given metric.

    Args:
        avg_df: DataFrame containing metrics and word counts
        metric: Name of the metric column to average

    Returns:
        Micro averaged metric value
    """
    return (avg_df[metric] * avg_df["n_words"]).sum() / avg_df["n_words"].sum()


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
        List of (idx_a, idx_b) tuples representing alignment, where None
        indicates insertion/deletion
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


def calculate_wer(ref_w: List[str], hyp_w: List[str], alignment: List[Tuple]) -> float:
    """Calculate Word Error Rate between reference and hypothesis words.

    Args:
        ref_w: List of reference words
        hyp_w: List of hypothesis words
        alignment: List of (ref_idx, hyp_idx) tuples from levenshtein_align

    Returns:
        Word Error Rate as a float between 0 and 1
    """
    subs = dels = ins = 0
    for ia, ib in alignment:
        if ia is None:
            ins += 1
        elif ib is None:
            dels += 1
        elif ref_w[ia] != hyp_w[ib]:
            subs += 1
    return (subs + dels + ins) / len(ref_w) if ref_w else 0.0


def sort_utterances_by_time(data: List[Dict]) -> List[Dict]:
    """Sort utterances in transcription data by start time.

    Args:
        data: List of transcription items, each containing 'start' time

    Returns:
        List of transcription items sorted by start time
    """
    return sorted(data, key=lambda x: x.get("start", float("inf")))


def swap_speakers(hyp_json: List[Dict]) -> List[Dict]:
    """Swap speaker labels in hypothesis JSON.

    Args:
        hyp_json: Hypothesis JSON data

    Returns:
        New JSON with swapped speaker labels
    """
    swapped_json = hyp_json.copy()
    for item in swapped_json:
        if "transcription" in item:
            for turn in item["transcription"]:
                if turn["author"] == "SPEAKER_00":
                    turn["author"] = "SPEAKER_01"
                elif turn["author"] == "SPEAKER_01":
                    turn["author"] = "SPEAKER_00"
    return swapped_json


async def apply_rework_dial(ground_truth: List[Dict], prediction: List[Dict]) -> Tuple:
    """Apply GPT-based diarization evaluation.

    Args:
        ground_truth: Ground truth transcription
        prediction: Predicted transcription

    Returns:
        Tuple of (evaluation result, cost, tokens)
    """
    return await gpt_engine_diarization_eval(ground_truth, prediction)


def apply_async_function(ground_truth: List[Dict], prediction: List[Dict]) -> Tuple:
    """Run async GPT evaluation function.

    Args:
        ground_truth: Ground truth transcription
        prediction: Predicted transcription

    Returns:
        Tuple of (evaluation result, cost, tokens)
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(apply_rework_dial(ground_truth, prediction))
    if len(result) == 3:
        return result
    else:
        raise ValueError(
            "Unexpected result format from gpt_engine_speaker_classification"
        )


def transfer_speakers(
    ref_w: List[str],
    ref_s: List[str],
    hyp_w: List[str],
    hyp_s: List[str],
    alignment: List[Tuple],
) -> Tuple[List[Optional[str]], Dict[str, str]]:
    """Transfer speaker labels from reference to hypothesis timeline.

    Args:
        ref_w: Reference words
        ref_s: Reference speakers
        hyp_w: Hypothesis words
        hyp_s: Hypothesis speakers
        alignment: Word alignment between reference and hypothesis

    Returns:
        Tuple of (gold_on_hyp, mapping) where:
        - gold_on_hyp: List of gold speaker labels on hypothesis timeline
        - mapping: Dictionary mapping reference speakers to hypothesis speakers
    """
    pair_cnt = Counter()
    for ia, ib in alignment:
        if ia is not None and ib is not None and ref_w[ia] == hyp_w[ib]:
            pair_cnt[(ref_s[ia], hyp_s[ib])] += 1

    mapping = {}
    for rs in set(ref_s):
        opts = [(c, hs) for (r, hs), c in pair_cnt.items() if r == rs]
        mapping[rs] = max(opts)[1] if opts else None

    gold_on_hyp = []
    for ia, ib in alignment:
        if ib is None:  # deletion in hyp → no token
            continue
        gold_on_hyp.append(None if ia is None else mapping.get(ref_s[ia]))

    return gold_on_hyp, mapping


def speaker_error_rate(
    gold_spk_hyp: List[Optional[str]],
    hyp_spk: List[str],
    hyp_w: List[str],
    ref_w: List[str],
    alignment: List[Tuple],
) -> float:
    """Calculate Speaker Error Rate.

    Args:
        gold_spk_hyp: Gold speaker labels on hypothesis timeline
        hyp_spk: Hypothesis speaker labels
        hyp_w: Hypothesis words
        ref_w: Reference words
        alignment: Word alignment between reference and hypothesis

    Returns:
        Speaker Error Rate as a float between 0 and 1
    """
    tok_err = tok_total = 0
    for (ia, ib), g in zip(alignment, gold_spk_hyp):
        if ia is None or ib is None:  # insertion / deletion: skip
            continue
        if ref_w[ia] != hyp_w[ib]:  # word is wrong: skip
            continue
        tok_total += 1
        tok_err += (g is None) or (g != hyp_spk[ib])
    return 0 if tok_total == 0 else tok_err / tok_total


def boundary_f1(
    gold_spk_hyp: List[Optional[str]], hyp_spk: List[str]
) -> Tuple[float, float, float]:
    """Calculate Boundary F1 Score.

    Args:
        gold_spk_hyp: Gold speaker labels on hypothesis timeline
        hyp_spk: Hypothesis speaker labels

    Returns:
        Tuple of (F1 score, precision, recall)
    """
    gold_b = [
        (
            g != gold_spk_hyp[i - 1]
            if g is not None and gold_spk_hyp[i - 1] is not None
            else False
        )
        for i, g in enumerate(gold_spk_hyp)
    ][1:]
    hyp_b = [hyp_spk[i] != hyp_spk[i - 1] for i in range(1, len(hyp_spk))]

    tp = sum(g and h for g, h in zip(gold_b, hyp_b))
    fp = sum(h and not g for g, h in zip(gold_b, hyp_b))
    fn = sum(g and not h for g, h in zip(gold_b, hyp_b))

    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    return f1, prec, rec


def calculate_metrics(
    stereo_data: List[Dict], mono_data: List[Dict]
) -> Dict[str, float]:
    """Calculate metrics for given stereo and mono data.

    Args:
        stereo_data: Ground truth JSON data
        mono_data: Hypothesis JSON data

    Returns:
        Dictionary containing micro-averaged metrics
    """
    # Group transcriptions by audio file
    stereo_by_file = {}
    mono_by_file = {}

    # Process stereo (ground truth) file
    for item in stereo_data:
        file_name = item.get("file_name", "unknown")
        if "transcription" in item:
            item["transcription"] = sort_utterances_by_time(item["transcription"])
        stereo_by_file[file_name] = item

    # Process mono (hypothesis) file
    for item in mono_data:
        file_name = item.get("file_name", "unknown")
        mono_by_file[file_name] = item

    # Calculate metrics for each file pair
    results = []
    missing_files = []
    all_gpt_scores = []

    for file_name in tqdm(stereo_by_file):
        if file_name in mono_by_file:
            # Get basic metrics
            ref_json = stereo_by_file[file_name]
            hyp_json = mono_by_file[file_name]
            ref_w, ref_s = flatten(ref_json)
            hyp_w, hyp_s = flatten(hyp_json)

            # Get alignment and speaker mapping
            aln = levenshtein_align(ref_w, hyp_w)
            gold_on_hyp, mapping = transfer_speakers(ref_w, ref_s, hyp_w, hyp_s, aln)

            # Calculate basic metrics
            stats = {
                "file_name": file_name,
                "n_words": len(ref_w),
                "wer": calculate_wer(ref_w, hyp_w, aln),
                "ser": speaker_error_rate(gold_on_hyp, hyp_s, hyp_w, ref_w, aln),
            }
            bf1, _, _ = boundary_f1(gold_on_hyp, hyp_s)
            stats["boundary_f1"] = bf1

            # Calculate GPT evaluation score
            try:
                gt_trans = ref_json.get("transcription", [])
                pred_trans = hyp_json.get("transcription", [])

                if gt_trans and pred_trans:
                    eval_result, cost, tokens = apply_async_function(
                        gt_trans, pred_trans
                    )
                    gpt_score = eval_result.get("diarization_score", 0)
                    all_gpt_scores.append(gpt_score)
                    stats["gpt_score"] = gpt_score
            except Exception as e:
                logging.error(f"Error calculating GPT score for {file_name}: {str(e)}")
                stats["gpt_score"] = 0.0

            results.append(stats)
        else:
            missing_files.append(file_name)

    # Create DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        return {
            "wer": 0.0,
            "ser": 0.0,
            "boundary_f1": 0.0,
            "gpt_score": 0.0,
        }

    # Calculate micro averages
    micro_avg = {
        "wer": micro(df, "wer"),
        "ser": micro(df, "ser"),
        "boundary_f1": micro(df, "boundary_f1"),
        "gpt_score": (
            sum(all_gpt_scores) / len(all_gpt_scores) if all_gpt_scores else 0.0
        ),
    }

    return micro_avg


def process_files(
    stereo_file: str, mono_file: str, der_metric: bool = False
) -> Dict[str, float]:
    """Process stereo and mono files and calculate metrics.

    Args:
        stereo_file: Path to stereo (ground truth) JSON file
        mono_file: Path to mono (hypothesis) JSON file
        der_metric: Whether to calculate DER metric (default: False)

    Returns:
        Dictionary containing best micro-averaged metrics
    """
    nest_asyncio.apply()
    # Load JSON files
    with open(stereo_file, "r", encoding="utf-8") as f:
        stereo_data = json.load(f)
    with open(mono_file, "r", encoding="utf-8") as f:
        mono_data = json.load(f)

    # Calculate metrics for both speaker orderings
    results_original = calculate_metrics(stereo_data, mono_data)
    results_swapped = calculate_metrics(stereo_data, swap_speakers(mono_data))

    # Select best metrics
    best_metrics = {
        "wer": float(round(min(results_original["wer"], results_swapped["wer"]), 4)),
        "ser": float(round(min(results_original["ser"], results_swapped["ser"]), 4)),
        "boundary_f1": float(
            round(
                max(results_original["boundary_f1"], results_swapped["boundary_f1"]), 4
            )
        ),
        "gpt_score": float(
            round(max(results_original["gpt_score"], results_swapped["gpt_score"]), 4)
        ),
    }

    # Calculate DER if requested
    if der_metric:
        if not DER_AVAILABLE:
            logging.warning(
                "timestamp_metric.py not available, skipping DER calculation"
            )
            best_metrics["der"] = 0.0
        else:
            try:
                der_values = calculate_der_metric(mono_file, stereo_file)
                if der_values:
                    best_metrics["der"] = float(
                        round(sum(der_values) / len(der_values), 4)
                    )
                else:
                    best_metrics["der"] = 0.0
            except Exception as e:
                logging.error(f"Error calculating DER: {str(e)}")
                best_metrics["der"] = 0.0

    return best_metrics


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("USAGE: python wder.py reference.json hypothesis.json")

    # Apply nest_asyncio for Jupyter compatibility
    nest_asyncio.apply()

    results = process_files(sys.argv[1], sys.argv[2], der_metric=True)
    print("Best metrics:")
    print(f"WER: {results['wer']:.4f}")
    print(f"SER: {results['ser']:.4f}")
    print(f"Boundary F1: {results['boundary_f1']:.4f}")
    if "der" in results:
        print(f"DER: {results['der']:.4f}")
    print(f"GPT Score: {results['gpt_score']:.4f}")
