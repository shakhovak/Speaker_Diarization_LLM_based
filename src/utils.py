import json
import os
from datetime import datetime
from typing import Dict, List, Optional


def save_metrics_to_json(
    metrics: Dict[str, float],
    experiment_name: str,
    reference_file: str,
    hypothesis_file: str,
    output_dir: str = "metrics",
) -> str:
    """Save metrics dictionary to a JSON file with experiment metadata.
    All experiments are saved in a single file.

    Args:
        metrics: Dictionary containing the metrics (WER, SER, etc.)
        experiment_name: Name of the experiment
        reference_file: Path to the reference/ground truth file
        hypothesis_file: Path to the hypothesis file
        output_dir: Directory to save the metrics file (default: "metrics")

    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use a fixed filename for all experiments
    output_path = os.path.join(output_dir, "experiments.json")

    # Create metadata dictionary for this experiment
    experiment_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "reference_file": reference_file,
        "hypothesis_file": hypothesis_file,
        "metrics": metrics,
    }

    # Load existing data if file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    # Convert single experiment to list format
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new experiment data
    data.append(experiment_data)

    # Save updated data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


def get_experiment_results(output_file: str = "metrics/experiments.json") -> List[Dict]:
    """Load experiment results from the JSON file.

    Args:
        output_file: Path to the JSON file containing experiment results
                   (default: "metrics/experiments.json")

    Returns:
        List of experiment results
    """
    if not os.path.exists(output_file):
        return []

    with open(output_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return []


def get_latest_experiment(
    output_file: str = "metrics/experiments.json",
) -> Optional[Dict]:
    """Get the most recent experiment result from the JSON file.

    Args:
        output_file: Path to the JSON file containing experiment results
                   (default: "metrics/experiments.json")

    Returns:
        Most recent experiment result or None if file is empty
    """
    results = get_experiment_results(output_file)
    if not results:
        return None

    # Sort by timestamp in descending order
    sorted_results = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)
    return sorted_results[0]
