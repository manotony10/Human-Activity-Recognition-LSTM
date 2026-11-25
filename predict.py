"""Predict the activity class for a sample window from the test dataset."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_split

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]


def main():
    model_path = ROOT / "models" / "har_lstm.keras"
    if not model_path.exists():
        raise SystemExit("Model not found. Train first with: python src/train.py")
    X_test, y_test = load_split("test")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    sample_index = 0
    probabilities = model.predict(X_test[sample_index:sample_index + 1], verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    actual = int(y_test[sample_index])
    print(f"Sample index: {sample_index}")
    print(f"Predicted activity: {LABELS[predicted]}")
    print(f"Actual activity:    {LABELS[actual]}")
    print(f"Confidence:         {probabilities[predicted] * 100:.2f}%")
    print("\nAll class probabilities:")
    for label, probability in zip(LABELS, probabilities):
        print(f"{label:25s} {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()
