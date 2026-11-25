# Human Activity Recognition Using Deep Learning (LSTM)

A Python mini-project that recognizes six human activities from smartphone accelerometer and gyroscope time-series signals using a Long Short-Term Memory (LSTM) neural network.

## Activities
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Dataset
Uses the **UCI Human Activity Recognition Using Smartphones (HAR)** dataset. The project downloads the dataset when first run (internet required).

Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Requirements
Python 3.10–3.12 recommended.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
Download and prepare dataset (automatically done by training):
```powershell
python src/train.py
```

Evaluate saved model:
```powershell
python src/evaluate.py
```

Predict an activity from one test sample:
```powershell
python src/predict.py
```

Training saves:
- `models/har_lstm.keras` – trained neural network
- `models/label_map.json` – activity labels
- `results/metrics.json` – test metrics
- `results/classification_report.txt` – per-class report
- `results/confusion_matrix.png` – confusion matrix

## Model overview
The input is a 128-time-step window containing 9 channels (3-axis body acceleration, 3-axis gravity acceleration, and 3-axis gyroscope). The network uses LSTM layers, dropout, and a dense softmax output for six classes.

## Project structure
```text
Human-Activity-Recognition-LSTM/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/             # downloaded dataset is kept here, not committed
├── models/           # trained model files (not committed)
└── results/          # generated evaluation outputs
```

## Notes
This is an academic mini-project starter implementation. Metrics depend on training settings and runtime environment; no accuracy value is claimed before training. The project is intended for educational demonstration, not medical or safety use.
