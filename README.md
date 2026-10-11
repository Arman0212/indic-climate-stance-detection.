# Climate Change Stance Detection in Hindi and Bengali YouTube Comments

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end NLP course project pipeline for 3-class stance detection on Hindi and Bengali YouTube comments toward the claim:
> **"Climate change and global warming is a serious concern."**

---

## 1. Project Overview

Social media discussions around climate change in South Asia feature high linguistic diversity and informal noise:
- **Native Indic Scripts**: Devanagari (Hindi) and Bengali (Bangla)
- **Romanized Text**: Hinglish and Banglish phonetic spellings
- **Code-Mixing**: Mixed English and Indic vocabulary
- **Noisy Inputs**: Non-standard orthography, emojis, colloquial expressions, and video-specific noise

### Stance Classes
The pipeline classifies comments into three discrete stance categories:
1. **`Favour`**: The author agrees that climate change/global warming is an urgent, severe, or real threat requiring action.
2. **`Against`**: The author disputes, minimizes, or mocks the claim (e.g., claiming it is a hoax, exaggeration, natural cycle, or foreign propaganda).
3. **`None`**: Neutral, off-topic, spam, or ambiguous comments (e.g., questions about video background music, greetings, unrelated commentary).

### Primary Evaluation Metric
- **Macro F1-Score**: Primary metric to ensure balanced sensitivity across minority and majority classes.
- **Secondary Metrics**: Overall Accuracy, Per-Class Precision / Recall / F1, and Confusion Matrix.

---

## 2. Repository Layout

```text
.
├── config.yaml                     # Central configuration for paths, columns, queries & seed
├── requirements.txt                # Pinned dependencies (tested on Python 3.10)
├── .env.example                    # Template for YouTube Data API v3 key
├── .gitignore                      # Excludes raw/processed data, secrets, venv, and outputs
├── Makefile                        # Shortcuts for running pipeline stages
├── run_all.sh                      # Shell script executing end-to-end verification pipeline
├── README.md                       # Comprehensive project documentation
├── data/
│   ├── raw/                        # Raw scraped comments (git-ignored)
│   ├── processed/                  # Preprocessed data, folds, and labeling sheets (git-ignored)
│   └── sample/                     # Synthetic 30-comment sample dataset (committed for testing)
│       └── sample_comments.csv
├── notebooks/                      # Exploratory data analysis and experimental notebooks
├── outputs/                        # Saved models, out-of-fold probabilities, logs, and plots
│   ├── baseline_model.joblib       # Serialized TF-IDF + Logistic Regression pipeline
│   ├── confusion_matrix.png        # High-resolution heatmap of stance classification
│   ├── oof_predictions.csv         # Detailed out-of-fold predictions with per-class probabilities
│   ├── oof_probabilities.npy       # Out-of-fold predicted probability matrix
│   ├── predictions.csv             # Test set predictions (id, predicted_label)
│   └── results_log.csv             # Benchmark log tracking model versions, settings, and F1
├── report/                         # Project reports and LaTeX assets
└── src/
    ├── __init__.py                 # Package initializer
    ├── utils.py                    # Seeding (seed=42), YAML config loader, and script guessing
    ├── scrape_youtube.py           # YouTube Data API v3 scraper with pagination & quota handling
    ├── make_labeling_sheet.py      # Stratified sampling and Cohen's kappa agreement calculation
    ├── preprocess.py               # Unicode NFC, URL/@mention removal, whitespace collapse + hooks
    ├── train_baseline.py           # TF-IDF char (1-5) ngrams + Logistic Regression (5-Fold CV)
    ├── evaluate.py                 # Macro F1, confusion matrix visualization, benchmark logging
    ├── predict.py                  # Standalone inference module producing outputs/predictions.csv
    ├── train_transformer.py        # [Stub] Multilingual transformer fine-tuning (IndicBERT/XLM-R)
    ├── llm_prompting.py            # [Stub] Zero-shot and few-shot LLM in-context prompting
    └── ensemble.py                 # [Stub] Soft-voting, hard-voting, and stacking meta-classifier
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+
- `pip` or `uv`

### Step 1: Clone and Create Virtual Environment
```bash
# Clone the repository (once remote is set)
cd NLP

# Option A: Using uv (fastest)
uv venv --python 3.10 .venv
source .venv/bin/activate

# Option B: Using standard python3 venv
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. YouTube Data API Key Configuration

To collect comments using `src/scrape_youtube.py`, you need a Google Cloud YouTube Data API v3 key:

1. Navigate to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `nlp-stance-detection`).
3. In the sidebar, select **APIs & Services** > **Library**.
4. Search for **YouTube Data API v3** and click **Enable**.
5. Select **APIs & Services** > **Credentials** > **Create Credentials** > **API key**.
6. Copy the generated key.
7. In your local repository root, copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
8. Edit `.env` and paste your key:
   ```env
   YOUTUBE_API_KEY=AIzaSy...YourActualKeyHere
   ```
> **Security Note**: `.env` is explicitly listed in `.gitignore` and must never be committed.

---

## 5. Running the Pipeline

### Quick Start: Run Everything on the Sample Dataset
Execute the full verification workflow (preprocess -> train baseline -> evaluate -> predict):

```bash
# Using Make:
make run-all

# Or directly using the shell script:
./run_all.sh
```

### Running Individual Pipeline Stages

#### 1. Data Collection (`src/scrape_youtube.py`)
Scrapes YouTube for search queries defined in `config.yaml` (default queries: `"जलवायु परिवर्तन"`, `"ग्लोबल वार्मिंग"`, `"জলবায়ু পরিবর্তন"`, `"বৈশ্বিক উষ্ণতা"`). Target is ~500 comments per language. Handles pagination, disabled comments, and quota exhaustion gracefully.
```bash
python src/scrape_youtube.py --config config.yaml
```
Output: `data/raw/standin_comments.csv`

#### 2. Labeling Sheet & Inter-Annotator Agreement (`src/make_labeling_sheet.py`)
- **Mode A: Sample comments for human labeling**:
  Samples 100 comments (50 Hindi, 50 Bengali) with blank annotator columns:
  ```bash
  python src/make_labeling_sheet.py --mode sample --n-per-lang 50
  ```
  Output: `data/processed/to_label.csv`

- **Mode B: Compute agreement and Cohen's Kappa**:
  After annotators fill in `label_annotator1` and `label_annotator2`:
  ```bash
  python src/make_labeling_sheet.py --mode agreement
  ```
  Prints raw percentage agreement, Cohen's kappa ($\kappa$), and details of conflicting rows.

#### 3. Text Preprocessing (`src/preprocess.py`)
Normalizes Unicode to NFC form, strips URLs and user `@mentions`, and collapses redundant whitespace. Contains clearly marked extension hooks for emoji mapping, character elongation collapse, and Indic transliteration.
```bash
python src/preprocess.py \
    --input data/sample/sample_comments.csv \
    --output data/processed/sample_preprocessed.csv
```

#### 4. Baseline Training (`src/train_baseline.py`)
Extracts sublinear character n-grams (1-5) and trains a `LogisticRegression(class_weight="balanced")` classifier using Stratified 5-Fold Cross-Validation.
```bash
python src/train_baseline.py \
    --input data/processed/sample_preprocessed.csv \
    --folds-output data/processed/folds.csv \
    --output-dir outputs
```
Outputs:
- `data/processed/folds.csv`: Fixed fold assignments for fair comparison with future models.
- `outputs/oof_probabilities.npy`: Out-of-fold probability distributions.
- `outputs/oof_predictions.csv`: Detailed out-of-fold predictions with per-class probabilities.
- `outputs/baseline_model.joblib`: Trained model pipeline for downstream inference.

#### 5. Evaluation & Logging (`src/evaluate.py`)
Calculates Macro F1, Accuracy, and per-class metrics. Generates a confusion matrix plot and appends a structured record to the results benchmark log.
```bash
python src/evaluate.py \
    --input outputs/oof_predictions.csv \
    --output-dir outputs
```
Outputs:
- `outputs/confusion_matrix.png`: Heatmap visualization.
- `outputs/results_log.csv`: Incremental benchmark log.

#### 6. Inference / Test Prediction (`src/predict.py`)
Applies the pre-trained pipeline to any input CSV and exports predictions:
```bash
python src/predict.py \
    --input data/sample/sample_comments.csv \
    --output outputs/predictions.csv
```
Output: `outputs/predictions.csv` with columns `id, predicted_label`.

---

## 6. Planned Model Extensions (Stubs)

- **`src/train_transformer.py`**: Fine-tuning pre-trained multilingual transformer encoders (`ai4bharat/indic-bert`, `xlm-roberta-base`, `bert-base-multilingual-cased`).
- **`src/llm_prompting.py`**: Zero-shot and few-shot in-context learning using Google Gemini or Claude with target claim alignment.
- **`src/ensemble.py`**: Ensembling baseline probabilities, transformer embeddings, and LLM outputs via soft-voting or a meta-classifier.

---

## 7. External Libraries and Pre-trained Models

| Library / Model | Pinned Version / Source | Primary Role in Project |
| :--- | :--- | :--- |
| **`scikit-learn`** | `1.7.2` | Feature extraction (`TfidfVectorizer`), classifier (`LogisticRegression`), Stratified CV, metrics |
| **`transformers`** | `5.19.0` | Tokenizers and sequence classification heads for multilingual transformer fine-tuning |
| **`datasets`** | `5.1.0` | Hugging Face dataset processing and streaming |
| **`torch`** | `2.14.1` | PyTorch backend for transformer training |
| **`pandas`** | `2.3.3` | Tabular data manipulation, annotation sheet creation, and CSV I/O |
| **`numpy`** | `2.2.6` | Numerical vector manipulation and OOF probability matrices |
| **`google-api-python-client`** | `2.201.0` | YouTube Data API v3 interactions (search and commentThreads) |
| **`python-dotenv`** | `1.2.4` | Secure `.env` environment variable resolution |
| **`pyyaml`** | `6.0.3` | Configuration parsing from `config.yaml` |
| **`indic-nlp-library`** | `0.92` | Indic text normalization and sentence/word tokenization |
| **`indic-transliteration`** | `2.3.82` | Transliteration between Romanized text (ITRANS/Harvard-Kyoto) and native Indic scripts |
| **`emoji`** | `2.16.0` | Emoji detection, parsing, and textual demojization |
| **`shap`** | `0.49.1` | Model interpretability and feature attribution for stance predictions |
| **`matplotlib`** / **`seaborn`** | `3.10.9` / `0.13.2` | High-quality confusion matrix heatmaps and diagnostic plotting |
| **`ai4bharat/indic-bert`** | Hugging Face Hub | Planned 12-language ALBERT model pre-trained on Indian languages |
| **`xlm-roberta-base`** | Hugging Face Hub | Planned multilingual masked language model for cross-lingual transfer |
