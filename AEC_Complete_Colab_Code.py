# ============================================================
# AEC: ARTIFICIAL INTELLIGENCE AND DEEP LEARNING
# Complete EDA + CNN experiments on MNIST, Fashion-MNIST, CIFAR-10
# Bonus: ANN vs CNN on MNIST, sample predictions, saved outputs
# Run this entire script in ONE Google Colab cell.
# ============================================================

# 1. IMPORT LIBRARIES
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# Create output folders
OUTPUT_DIR = "/content/AEC_AI_Deep_Learning_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "graphs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

# Settings: reduce epochs for a quick first run; increase for experiments.
EPOCHS_MNIST = 5
EPOCHS_FASHION = 5
EPOCHS_CIFAR10 = 10
EPOCHS_ANN = 5
BATCH_SIZE = 64

# Helper to save and display every plot
def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, "graphs", filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved graph:", path)

# ============================================================
# 2. DATASET LOADING
# ============================================================

print("\n========== LOADING DATASETS ==========")

(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = (
    tf.keras.datasets.mnist.load_data()
)

(x_fashion_train, y_fashion_train), (x_fashion_test, y_fashion_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

(x_cifar_train, y_cifar_train), (x_cifar_test, y_cifar_test) = (
    tf.keras.datasets.cifar10.load_data()
)

y_cifar_train = y_cifar_train.flatten()
y_cifar_test = y_cifar_test.flatten()

print("MNIST:", x_mnist_train.shape, x_mnist_test.shape)
print("Fashion-MNIST:", x_fashion_train.shape, x_fashion_test.shape)
print("CIFAR-10:", x_cifar_train.shape, x_cifar_test.shape)

# ============================================================
# 3. GENERAL EDA FUNCTION
# ============================================================

def perform_eda(dataset_name, x_train, y_train, class_names, grayscale=True):
    print(f"\n========== {dataset_name} EDA ==========")
    print("Training images:", x_train.shape)
    print("Labels shape:", y_train.shape)
    print("Number of classes:", len(class_names))
    print("Pixel minimum:", x_train.min())
    print("Pixel maximum:", x_train.max())
    print("Mean pixel value:", round(float(x_train.mean()), 3))
    print("Standard deviation:", round(float(x_train.std()), 3))

    # Display sample images
    plt.figure(figsize=(12, 6))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        if grayscale:
            plt.imshow(x_train[i], cmap="gray")
        else:
            plt.imshow(x_train[i])
        plt.title(class_names[int(y_train[i])], fontsize=9)
        plt.axis("off")
    plt.suptitle(f"{dataset_name}: Sample Images")
    save_plot(f"{dataset_name.lower().replace('-', '_')}_samples.png")

    # Class distribution
    counts = np.bincount(y_train, minlength=len(class_names))
    plt.figure(figsize=(11, 4))
    sns.barplot(x=class_names, y=counts)
    plt.title(f"{dataset_name}: Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of training images")
    plt.xticks(rotation=35, ha="right")
    save_plot(f"{dataset_name.lower().replace('-', '_')}_class_distribution.png")

    # Pixel intensity histogram
    plt.figure(figsize=(8, 4))
    plt.hist(x_train[0].flatten(), bins=40, edgecolor="black")
    plt.title(f"{dataset_name}: Pixel Intensity Distribution (first image)")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    save_plot(f"{dataset_name.lower().replace('-', '_')}_pixel_histogram.png")

    return counts

mnist_classes = [str(i) for i in range(10)]

fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

cifar_classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

mnist_counts = perform_eda(
    "MNIST", x_mnist_train, y_mnist_train, mnist_classes, True
)
fashion_counts = perform_eda(
    "Fashion-MNIST", x_fashion_train, y_fashion_train, fashion_classes, True
)
cifar_counts = perform_eda(
    "CIFAR-10", x_cifar_train, y_cifar_train, cifar_classes, False
)

# ============================================================
# 4. DATASET COMPARISON GRAPH
# ============================================================

comparison = pd.DataFrame({
    "Dataset": ["MNIST", "Fashion-MNIST", "CIFAR-10"],
    "Training images": [len(x_mnist_train), len(x_fashion_train), len(x_cifar_train)],
    "Testing images": [len(x_mnist_test), len(x_fashion_test), len(x_cifar_test)],
    "Image height": [28, 28, 32],
    "Image width": [28, 28, 32],
    "Channels": [1, 1, 3]
})

print("\n========== DATASET COMPARISON ==========")
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(OUTPUT_DIR, "reports", "dataset_comparison.csv"), index=False)

plt.figure(figsize=(8, 5))
comparison.set_index("Dataset")[["Training images", "Testing images"]].plot(
    kind="bar", figsize=(8, 5)
)
plt.title("Training and Testing Dataset Size Comparison")
plt.ylabel("Number of images")
plt.xticks(rotation=0)
save_plot("dataset_size_comparison.png")

# ============================================================
# 5. PREPROCESSING
# ============================================================

# Convert pixel values from 0-255 to 0-1.
x_mnist_train_p = x_mnist_train.astype("float32") / 255.0
x_mnist_test_p = x_mnist_test.astype("float32") / 255.0

x_fashion_train_p = x_fashion_train.astype("float32") / 255.0
x_fashion_test_p = x_fashion_test.astype("float32") / 255.0

x_cifar_train_p = x_cifar_train.astype("float32") / 255.0
x_cifar_test_p = x_cifar_test.astype("float32") / 255.0

# Add channel dimension to grayscale datasets.
x_mnist_train_cnn = x_mnist_train_p[..., np.newaxis]
x_mnist_test_cnn = x_mnist_test_p[..., np.newaxis]

x_fashion_train_cnn = x_fashion_train_p[..., np.newaxis]
x_fashion_test_cnn = x_fashion_test_p[..., np.newaxis]

print("\nPreprocessing completed.")

# ============================================================
# 6. CNN MODEL BUILDER
# ============================================================

def build_cnn(input_shape, number_of_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(number_of_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# 7. TRAINING, EVALUATION AND GRAPH FUNCTION
# ============================================================

results = []

def train_evaluate_cnn(
    dataset_name, x_train, y_train, x_test, y_test,
    input_shape, class_names, epochs
):
    print(f"\n========== TRAINING {dataset_name} CNN ==========")

    model = build_cnn(input_shape)
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"{dataset_name} test loss: {test_loss:.4f}")
    print(f"{dataset_name} test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Accuracy and loss graphs
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], marker="o", label="Training")
    plt.plot(history.history["val_accuracy"], marker="o", label="Validation")
    plt.title(f"{dataset_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], marker="o", label="Training")
    plt.plot(history.history["val_loss"], marker="o", label="Validation")
    plt.title(f"{dataset_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    save_plot(f"{dataset_name.lower().replace('-', '_')}_accuracy_loss.png")

    # Predictions
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_plot(f"{dataset_name.lower().replace('-', '_')}_confusion_matrix.png")

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(f"\n{dataset_name} Classification Report:\n")
    print(report)

    report_path = os.path.join(
        OUTPUT_DIR, "reports",
        f"{dataset_name.lower().replace('-', '_')}_classification_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(report)

    # Save model
    model_path = os.path.join(
        OUTPUT_DIR, "models",
        f"{dataset_name.lower().replace('-', '_')}_cnn.keras"
    )
    model.save(model_path)
    print("Saved model:", model_path)

    results.append({
        "Dataset": dataset_name,
        "Test Loss": float(test_loss),
        "Test Accuracy": float(test_accuracy),
        "Test Accuracy (%)": round(float(test_accuracy) * 100, 2),
        "Epochs": epochs
    })

    # Display sample predictions
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        if len(input_shape) == 3 and input_shape[-1] == 1:
            plt.imshow(x_test[i].squeeze(), cmap="gray")
        else:
            plt.imshow(x_test[i])

        actual = class_names[int(y_test[i])]
        predicted = class_names[int(y_pred[i])]
        plt.title(f"Actual: {actual}\nPred: {predicted}", fontsize=8)
        plt.axis("off")

    plt.suptitle(f"{dataset_name}: Sample Predictions")
    save_plot(f"{dataset_name.lower().replace('-', '_')}_predictions.png")

    return model, history, y_pred

# ============================================================
# 8. EXPERIMENT 1: MNIST CNN
# ============================================================

mnist_model, mnist_history, mnist_predictions = train_evaluate_cnn(
    "MNIST",
    x_mnist_train_cnn, y_mnist_train,
    x_mnist_test_cnn, y_mnist_test,
    (28, 28, 1), mnist_classes, EPOCHS_MNIST
)

# ============================================================
# 9. EXPERIMENT 2: FASHION-MNIST CNN
# ============================================================

fashion_model, fashion_history, fashion_predictions = train_evaluate_cnn(
    "Fashion-MNIST",
    x_fashion_train_cnn, y_fashion_train,
    x_fashion_test_cnn, y_fashion_test,
    (28, 28, 1), fashion_classes, EPOCHS_FASHION
)

# ============================================================
# 10. EXPERIMENT 3: CIFAR-10 CNN
# ============================================================

cifar_model, cifar_history, cifar_predictions = train_evaluate_cnn(
    "CIFAR-10",
    x_cifar_train_p, y_cifar_train,
    x_cifar_test_p, y_cifar_test,
    (32, 32, 3), cifar_classes, EPOCHS_CIFAR10
)

# ============================================================
# 11. BONUS: ANN VS CNN ON MNIST
# ============================================================

print("\n========== BONUS: ANN VS CNN ON MNIST ==========")

ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

ann_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ann_model.fit(
    x_mnist_train_p, y_mnist_train,
    epochs=EPOCHS_ANN,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

ann_loss, ann_accuracy = ann_model.evaluate(
    x_mnist_test_p, y_mnist_test, verbose=0
)

cnn_loss, cnn_accuracy = mnist_model.evaluate(
    x_mnist_test_cnn, y_mnist_test, verbose=0
)

print(f"ANN test accuracy: {ann_accuracy*100:.2f}%")
print(f"CNN test accuracy: {cnn_accuracy*100:.2f}%")

ann_cnn_comparison = pd.DataFrame({
    "Model": ["ANN", "CNN"],
    "Test Accuracy (%)": [ann_accuracy * 100, cnn_accuracy * 100]
})

print(ann_cnn_comparison.to_string(index=False))
ann_cnn_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "reports", "ann_vs_cnn_comparison.csv"),
    index=False
)

plt.figure(figsize=(6, 4))
sns.barplot(
    data=ann_cnn_comparison,
    x="Model", y="Test Accuracy (%)"
)
plt.ylim(0, 100)
plt.title("ANN vs CNN on MNIST")
plt.ylabel("Test Accuracy (%)")
save_plot("ann_vs_cnn_comparison.png")

ann_model.save(os.path.join(OUTPUT_DIR, "models", "mnist_ann.keras"))

# ============================================================
# 12. BONUS: SINGLE MNIST DIGIT PREDICTION
# ============================================================

print("\n========== BONUS: SINGLE DIGIT PREDICTION ==========")

sample_index = 0
sample_image = x_mnist_test_cnn[sample_index]
sample_prediction = mnist_model.predict(
    sample_image[np.newaxis, ...], verbose=0
)
predicted_digit = int(np.argmax(sample_prediction))
actual_digit = int(y_mnist_test[sample_index])
confidence = float(np.max(sample_prediction)) * 100

plt.figure(figsize=(4, 4))
plt.imshow(sample_image.squeeze(), cmap="gray")
plt.title(
    f"Actual: {actual_digit} | Predicted: {predicted_digit}\n"
    f"Confidence: {confidence:.2f}%"
)
plt.axis("off")
save_plot("single_digit_prediction.png")

print("Actual digit:", actual_digit)
print("Predicted digit:", predicted_digit)
print("Prediction confidence:", f"{confidence:.2f}%")

# ============================================================
# 13. FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== FINAL CNN RESULTS ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "reports", "final_results.csv"),
    index=False
)

with open(os.path.join(OUTPUT_DIR, "reports", "run_summary.json"), "w") as f:
    json.dump({
        "tensorflow_version": tf.__version__,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "results": results
    }, f, indent=4)

# ============================================================
# 14. ZIP ALL OUTPUTS FOR DOWNLOAD
# ============================================================

import shutil

zip_path = shutil.make_archive(
    "/content/AEC_AI_Deep_Learning_Results",
    "zip",
    OUTPUT_DIR
)

print("\n===================================================")
print("ALL EXPERIMENTS COMPLETED.")
print("Results folder:", OUTPUT_DIR)
print("ZIP file:", zip_path)
print("Download the ZIP from the Colab Files panel.")
print("===================================================")

# In Google Colab, uncomment the next two lines to download:
# from google.colab import files
# files.download(zip_path)
