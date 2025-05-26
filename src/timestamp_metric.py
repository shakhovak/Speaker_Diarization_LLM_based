import json
import os
import logging
import warnings
from typing import List, Dict
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from pyannote.core import Timeline


def normalize_path(path: str) -> str:
    """
    Normalize path separators to forward slashes.

    Args:
        path (str): Path to normalize

    Returns:
        str: Normalized path
    """
    return path.replace("\\", "/")


def create_rttm_file(record_file_name: str, record_data: dict, output_path: str) -> str:
    """
    Create RTTM file for a single audio file.

    Args:
        record_file_name (str): Name of the audio file
        record_data (dict): Dictionary containing transcription data for this file
        output_path (str): Base path for output RTTM file

    Returns:
        str: Path to created RTTM file
    """
    # Normalize record file name
    record_file_name = normalize_path(record_file_name)

    # Create RTTM filename based on the audio filename
    rttm_path = f"{output_path.split('.')[0]}_{record_file_name}.rttm"

    transcription = record_data.get("transcription", [])

    if not transcription:
        return rttm_path

    with open(rttm_path, "wb") as f:
        for turn in transcription:
            # Convert milliseconds to seconds and normalize start time
            start = turn["start"]
            end = turn["end"]
            duration = end - start

            fields = [
                "SPEAKER",
                record_file_name,
                "1",
                f"{start:.3f}",
                f"{duration:.3f}",
                "<NA>",
                "<NA>",
                turn["author"],
                "<NA>",
                "<NA>",
            ]
            line = " ".join(fields)
            f.write(line.encode("utf-8"))
            f.write(b"\n")

    return rttm_path


def sort_utterances_by_time(data: List[Dict]) -> List[Dict]:
    """Sort utterances in transcription data by start time.

    Args:
        data: List of transcription items, each containing 'start' time

    Returns:
        List of transcription items sorted by start time
    """
    return sorted(data, key=lambda x: x.get("start", float("inf")))


def calculate_der(path_pred: str, path_gt: str) -> List[float]:
    """
    Calculate Diarization Error Rate (DER) for all records.
    Creates separate RTTM files for each audio file.

    Args:
        path_pred (str): Path to prediction JSON file
        path_gt (str): Path to ground truth JSON file

    Returns:
        List[float]: List of DER values for each record
    """
    # Suppress pyannote warnings about UEM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
        all_ders = []

        # Load both prediction and ground truth data
        with open(path_pred, "r", encoding="utf-8") as file:
            pred_data = json.load(file)
        with open(path_gt, "r", encoding="utf-8") as file:
            gt_data = json.load(file)

        # Convert to dictionary format if input is list
        if isinstance(pred_data, list):
            pred_dict = {normalize_path(item["file_name"]): item for item in pred_data}
        else:
            pred_dict = {normalize_path(k): v for k, v in pred_data.items()}

        if isinstance(gt_data, list):
            gt_dict = {normalize_path(item["file_name"]): item for item in gt_data}
        else:
            gt_dict = {normalize_path(k): v for k, v in gt_data.items()}

        # Process each audio file
        for file_name in pred_dict.keys():
            try:
                if file_name not in gt_dict:
                    logging.warning(f"File {file_name} not found in ground truth data")
                    continue

                # Sort ground truth utterances by start time
                if "transcription" in gt_dict[file_name]:
                    gt_dict[file_name]["transcription"] = sort_utterances_by_time(
                        gt_dict[file_name]["transcription"]
                    )

                # Create RTTM files for this audio file
                gt_rttm = create_rttm_file(
                    file_name, gt_dict[file_name], f"{path_gt}_gt"
                )
                pred_rttm = create_rttm_file(
                    file_name, pred_dict[file_name], f"{path_pred}_pred"
                )

                # Load RTTM files
                _, gt = load_rttm(gt_rttm).popitem()
                _, pred = load_rttm(pred_rttm).popitem()

                # Create UEM from the union of reference and hypothesis
                gt_timeline = Timeline(segments=gt.get_timeline())
                pred_timeline = Timeline(segments=pred.get_timeline())
                uem = gt_timeline.union(pred_timeline)

                # Try both possible speaker label mappings
                der1 = metric(gt, pred, uem=uem)  # Original mapping
                der2 = metric(
                    gt,
                    pred.rename_labels(
                        {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
                    ),
                    uem=uem,
                )  # Swapped mapping

                # Use the lower DER value
                der = min(der1, der2)
                all_ders.append(der * 100)

                # Clean up RTTM files after processing
                os.remove(gt_rttm)
                os.remove(pred_rttm)

            except Exception as e:
                logging.error(f"Error calculating DER for {file_name}: {str(e)}")
                continue

        return all_ders


def save_der_statistics(
    der_values: List[float], experiment_name: str, output_path: str
) -> None:
    """
    Save DER statistics to a JSON file.

    Args:
        der_values (List[float]): List of DER values
        experiment_name (str): Name of the experiment
        output_path (str): Path to save the statistics JSON file
    """
    if not der_values:
        logging.warning("No DER values to save")
        return

    statistics = {
        "experiment_name": experiment_name,
        "average_der": sum(der_values) / len(der_values),
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        logging.info(f"DER statistics saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving DER statistics: {str(e)}")
