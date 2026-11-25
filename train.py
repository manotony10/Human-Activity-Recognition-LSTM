"""Train, validate, and evaluate the HAR LSTM."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_data
from model import build_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    model = build_model()
    callbacks = [
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    history = model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=60, batch_size=64, callbacks=callbacks, verbose=1
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, target_names=LABELS, digits=4)
    matrix = confusion_matrix(y_test, predictions)

    model.save(MODEL_DIR / "har_lstm.keras")
    (MODEL_DIR / "label_map.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(LABELS)}, indent=2),
        encoding="utf-8"
    )
    metrics = {"test_accuracy": accuracy, "test_samples": int(len(y_test))}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix)
    ax.set_title("Confusion Matrix - HAR LSTM")
    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("True Activity")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save training history plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"], label="Training accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=150)
    plt.close(fig)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:\n", report)
    print(f"Saved model to {MODEL_DIR / 'har_lstm.keras'}")


if __name__ == "__main__":
    main()
