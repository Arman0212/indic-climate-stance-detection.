.PHONY: all run-all help preprocess train evaluate predict setup scrape sample-labels agreement clean

PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; elif [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

all: run-all

run-all:
	@bash run_all.sh

setup:
	@echo "Setting up Python virtual environment and dependencies..."
	uv venv --python 3.10 .venv || python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

scrape:
	$(PYTHON) src/scrape_youtube.py --config config.yaml

sample-labels:
	$(PYTHON) src/make_labeling_sheet.py --mode sample --config config.yaml

agreement:
	$(PYTHON) src/make_labeling_sheet.py --mode agreement --config config.yaml

preprocess:
	$(PYTHON) src/preprocess.py --input data/sample/sample_comments.csv --output data/processed/sample_preprocessed.csv --config config.yaml

train:
	$(PYTHON) src/train_baseline.py --input data/processed/sample_preprocessed.csv --folds-output data/processed/folds.csv --output-dir outputs --config config.yaml

evaluate:
	$(PYTHON) src/evaluate.py --input outputs/oof_predictions.csv --output-dir outputs --config config.yaml

predict:
	$(PYTHON) src/predict.py --input data/sample/sample_comments.csv --output outputs/predictions.csv --model-path outputs/baseline_model.joblib --config config.yaml

clean:
	rm -rf data/processed/* outputs/*
	touch data/processed/.gitkeep outputs/.gitkeep

help:
	@echo "Available Make targets:"
	@echo "  make run-all       : Run complete pipeline on sample data (preprocess -> train -> evaluate -> predict)"
	@echo "  make setup         : Create virtualenv and install pinned requirements"
	@echo "  make scrape        : Scrape YouTube comments using API v3"
	@echo "  make sample-labels : Sample comments into data/processed/to_label.csv"
	@echo "  make agreement     : Calculate Cohen's kappa and agreement on filled labeling sheet"
	@echo "  make preprocess    : Preprocess text data"
	@echo "  make train         : Train baseline model with 5-fold cross validation"
	@echo "  make evaluate      : Compute macro F1, confusion matrix, and log results"
	@echo "  make predict       : Generate stance predictions on sample test data"
	@echo "  make clean         : Clean intermediate outputs and processed files"
