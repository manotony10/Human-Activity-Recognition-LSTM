"""Evaluate the saved HAR LSTM model on the official test split."""
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_split

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]


def main():
    model_path = ROOT / "models" / "har_lstm.keras"
    if not model_path.exists():
        raise SystemExit("Model not found. Run: python src/train.py")
    X_test, y_test = load_split("test")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=LABELS))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()
