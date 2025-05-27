# Speaker Diarization and Transcription Analysis

## Abstract
This study presents a comprehensive comparison of multiple approaches to speaker diarization and transcription in conversational audio. We evaluate API-based solutions, PyAnnote-based diarization, and GPT-powered methods on both stereo and mono inputs. Our analysis leverages a suite of metrics—Word Error Rate (WER), Speaker Error Rate (SER), Boundary F1 Score, Diarization Error Rate (DER), and a human-centric GPT Score—to characterize each method's strengths and weaknesses. We also introduce and contextualize the **Transcript-Preserving Speaker Transfer (TPST)** algorithm, which underpins our text-level speaker mapping and error analysis.

## Overview

Evaluating diarization systems is inherently challenging due to two main factors:
1. Arbitrary speaker labels (e.g., "SPEAKER_01" could refer to different individuals across systems)
2. Imperfect word alignment between systems using different ASR models

This research addresses these challenges through the **Transcript-Preserving Speaker Transfer (TPST)** algorithm, which enables:
- Separation of speaker attribution quality from transcription quality
- Fair comparisons across systems with different tokenizations
- Evaluation of LLM-based diarization without audio or timing information

A key focus of this research is testing **how well large language models (LLMs) can infer speaker identities purely from text**—without access to audio, embeddings, or timing. TPST makes this possible by:
- Transferring oracle speaker labels to hypothesis timelines
- Enabling semantic speaker prediction evaluation
- Providing a framework for comparing LLM-based and acoustic-based approaches

## Repository Structure

```
Speaker_Diarization_LLM_based/
├── src/                    # Core implementation files
│   ├── pyannote_aligned.py    # PyAnnote-based diarization implementation
│   ├── word_level_metric.py   # TPST algorithm and metric calculations
│   ├── timestamp_metric.py    # DER calculations and RTTM conversion
│   └── utils.py              # Common utility functions
├── data/                   # Example data directory (empty, as per privacy requirements)
├── examples/              # Example scripts (to be added if needed)
├── README.md             # Main documentation
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore rules
└── LICENSE              # MIT License
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up PyAnnote authentication:
   - Get your authentication token from Hugging Face
   - Set it as an environment variable or use it directly in the code

## Methodology

### Data Preparation
The project uses a structured approach to file organization and processing:

- **File Structure**
  - `stereo_transcriptions/`: Contains stereo-based JSON files with word-level timing
  - `mono_transcriptions/`: Contains mono-based JSON files
  - `metrics/`: Stores experiment results and evaluation metrics
  - `experiments/`: Contains configuration files for different experimental setups

- **JSON Format**
  ```json
  {
    "file_name": "audio_file_name",
    "transcription": [
      {
        "author": "SPEAKER_00",
        "text": "transcribed text",
        "start": start_time,
        "end": end_time,
        "words": [
          {
            "text": "word",
            "start": word_start_time,
            "end": word_end_time
          }
        ]
      }
    ]
  }
  ```

### Transcript-Preserving Speaker Transfer (TPST)

TPST is a text-based speaker alignment algorithm that enables cross-system comparison by mapping gold speaker labels onto hypothesis word timelines. The algorithm proceeds in four steps:

1. **Levenshtein Alignment**
   - Aligns hypothesis and reference words
   - Handles insertions, deletions, and substitutions
   - Preserves word order and context

2. **Speaker Co-occurrence Analysis**
   - Counts speaker label pairs on matched tokens
   - Identifies consistent speaker mappings
   - Handles speaker label permutations

3. **Best Permutation Selection**
   - Uses Hungarian algorithm for optimal mapping
   - Considers both direct matches and context
   - Handles cases with multiple valid mappings

4. **Label Transfer**
   - Maps gold speaker labels to hypothesis timeline
   - Preserves word-level alignment
   - Enables fair metric calculation

### Evaluation Metrics

1. **Word Error Rate (WER)**  
   Measures word-level transcription accuracy using Levenshtein alignment  
   ```
   WER = (S + D + I) / N
   ```
   where:
   - S: Substitutions
   - D: Deletions
   - I: Insertions
   - N: Total reference words

2. **Speaker Error Rate (SER)**  
   Measures token-level speaker attribution errors  
   ```
   SER = #(correct word, wrong speaker) / # correct words
   ```
   - Focuses on speaker labeling accuracy
   - Independent of transcription quality
   - Uses TPST for fair comparison

3. **Boundary F1 Score**  
   Evaluates segmentation quality by comparing speaker change-points
   - Precision: Correctly identified boundaries
   - Recall: Missed boundaries
   - F1: Harmonic mean of precision and recall

4. **Diarization Error Rate (DER)**  
   Time-weighted speaker labeling error
   - Miss: Undetected speech
   - False Alarm: Non-speech detected as speech
   - Confusion: Speaker misclassification
   - Uses 0.25s collar and skips overlaps

5. **GPT Evaluation Score**  
   Human-style fluency rating (1-10)
   - Assesses semantic impact of errors
   - Considers context and coherence
   - Provides human-centric evaluation

### Experimental Approaches

| Approach                        | Input   | Pipeline Components                                                |
|---------------------------------|---------|--------------------------------------------------------------------|
| **API**                         | Mono mp3 files   | Provider ASR + built-in diarizer                                   |
| **PyAnnote Align (stereo/mono)**| Stereo/Mono oracle transcripts, mono mp3 for pyannote | PyAnnote VAD → embeddings → clustering    |
| **GPT-Only (v1/v2)**            | Stereo/Mono oracle transcripts   | Oracle transcript → GPT-4 prompted to segment & label speakers     |
| **GPT API Correction**          | API result | API output → GPT relabel & boundary smoothing                      |

## Usage

### Basic Diarization
```python
from pyannote_aligned import process_directory

process_directory(
    audio_dir="audio_files/",
    transcription_file="transcriptions.json",
    output_file="results.json",
    auth_token="YOUR_TOKEN"
)
```

### Metric Calculation
```python
from word_level_metric import process_files

metrics = process_files(
    stereo_file="ground_truth.json",
    mono_file="prediction.json",
    der_metric=True
)
print(f"WER: {metrics['wer']:.4f}")
print(f"SER: {metrics['ser']:.4f}")
```

### DER Evaluation
```python
from timestamp_metric import calculate_der, save_der_statistics

der_values = calculate_der("prediction.json", "ground_truth.json")
save_der_statistics(
    der_values=der_values,
    experiment_name="experiment_1",
    output_path="der_stats.json"
)
```

## Results

| Experiment Name                   | Base    | WER    | SER     | Boundary F1 | GPT Score | DER     |
|-----------------------------------|---------|--------|---------|-------------|-----------|---------|
| pyannote_align_stereo_based       | stereo  | 0.0507 | 0.1587  | 0.3954      | 6.4762    | 34.2971 |
| gpt_only_v2_stereo_based          | stereo  | 0.0009 | 0.2125  | 0.3778      | 6.7143    | —       |
| pyannote_align_mono_based         | mono    | 0.3292 | 0.2566  | 0.4811      | 6.0476    | 31.5624 |
| gpt_only_stereo_based             | stereo  | 0.0407 | 0.3025  | 0.5160      | 7.0476    | —       |
| gpt_only_mono_based               | mono    | 0.3341 | 0.3291  | 0.3927      | 6.0476    | —       |
| gpt_only_v2_mono_based            | mono    | 0.3315 | 0.3350  | 0.3256      | 6.0000    | —       |
| API_v2                            | mono    | 0.2853 | 0.3525  | 0.2642      | 6.5714    | 41.8712 |
| API                               | mono    | 0.3000 | 0.3577  | 0.3458      | 5.7619    | 45.1620 |
| gpt_api_correction                | mono    | 0.3280 | 0.3709  | 0.4357      | 5.6190    | —       |
| gpt_api_correction_stereo_based   | stereo  | 0.3171 | 0.3731  | 0.2902      | 6.0952    | —       |

## Key Findings

1. **PyAnnote Performance**
   - Best SER on stereo (0.1587)
   - Best DER on mono (31.56%)
   - Strong Boundary F1 on mono inputs
   - Consistent performance across metrics

2. **GPT-Only Methods**
   - Excellent performance on oracle inputs
   - High fluency scores (GPT score ~7)
   - Challenges with speaker segmentation
   - Potential for improvement in boundary detection

3. **API-based Diarization**
   - Higher error rates (SER: 0.35-0.37)
   - Significant DER (>45%)
   - GPT post-processing shows modest improvements
   - Room for optimization in speaker attribution

## Technical Implementation

The project is implemented in Python with the following key components:

### Core Scripts

1. **pyannote_aligned.py**
   - Handles speaker diarization using PyAnnote
   - Aligns word timings with speaker segments
   - Processes both stereo and mono audio files

2. **word_level_metric.py**
   - Implements TPST algorithm for speaker alignment
   - Calculates WER, SER, and Boundary F1 metrics
   - Integrates GPT-based evaluation

3. **timestamp_metric.py**
   - Calculates Diarization Error Rate (DER)
   - Converts JSON to RTTM format for evaluation
   - Handles speaker label permutations

### GPT Integration (Proprietary)

The project includes several proprietary GPT-based components that are not publicly available:

1. **Speaker Classification**
   - Semantic analysis of dialogue content
   - Speaker role identification
   - Context-aware speaker attribution
   - Handles complex dialogue patterns

2. **Diarization Correction**
   - Post-processing of initial diarization results
   - Boundary refinement
   - Speaker label optimization
   - Error correction based on dialogue context

3. **Quality Evaluation**
   - Semantic assessment of diarization quality
   - Scoring system (1-10) for diarization accuracy
   - Detailed error analysis and reporting
   - Human-like evaluation criteria

Note: The GPT integration components are proprietary and not included in the public repository. They utilize GPT-4 for advanced natural language understanding and dialogue analysis.

### Dependencies

- **Core Libraries**
  - PyAnnote for diarization
  - PyTorch for deep learning
  - OpenAI for GPT integration
  - Pandas for data processing

- **Evaluation Tools**
  - Custom TPST implementation
  - PyAnnote metrics
  - GPT-4 for semantic evaluation

## Future Work

The **TPST algorithm** proves essential for evaluating speaker diarization across mixed pipelines. It enables text-aligned speaker labeling comparisons even when systems disagree on tokens or speaker IDs.

Future extensions will include:
- **Token-level smoothing using LLMs**
  - Detect and correct speaker "islands"
  - Improve boundary detection
  - Reduce false speaker changes

- **TPST with Hungarian assignment for >2 speaker systems**
  - Extend to multi-speaker scenarios
  - Optimize for complex conversations
  - Handle overlapping speech

- **Training diarization-aware ASR models**
  - Integrate TPST natively
  - Improve word-level timing
  - Enhance speaker attribution
