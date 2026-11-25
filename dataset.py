"""Download and load the UCI HAR inertial-signal dataset."""
from pathlib import Path
import zipfile
import requests
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"
ZIP_PATH = DATA_DIR / "uci_har.zip"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def download_dataset():
    """Download and extract dataset if it is not already available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATASET_DIR / "train").exists() and (DATASET_DIR / "test").exists():
        return
    print("Downloading UCI HAR dataset...")
    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)
    print("Dataset extracted:", DATASET_DIR)


def load_split(split):
    """Return X shaped (samples, 128, 9), and y as zero-based class IDs."""
    download_dataset()
    folder = DATASET_DIR / split / "Inertial Signals"
    arrays = []
    for signal in SIGNALS:
        path = folder / f"{signal}_{split}.txt"
        values = np.loadtxt(path, dtype=np.float32)
        arrays.append(values)
    X = np.stack(arrays, axis=-1)
    labels = np.loadtxt(DATASET_DIR / split / f"y_{split}.txt", dtype=np.int64)
    y = labels - 1
    return X, y


def load_data():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_test, y_test
