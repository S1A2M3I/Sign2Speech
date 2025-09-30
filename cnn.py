import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy

def image_resize(image, height=45, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    r = height / float(h if w > h else w)
    dim = (int(w * r), height) if w > h else (height, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def data_read(data_path, category='Original', source='Dataset2'):
    labels, images = [], []
    full_path = os.path.join(data_path, source)
    for label in os.listdir(full_path):
        label_path = os.path.join(full_path, label, category)
        for image in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, image))
            if img is not None:
                img = image_resize(img)
                img = cv2.resize(img, (45, 45))  # uniform size
                images.append(img)
                labels.append(label)
    return labels, images

def evaluate_model(model, X_val, y_val, label_names, save_prefix="cnn"):
    print("\n🔍 Evaluating model...")
    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(f"🎯 F1 Score (Macro): {f1_macro:.2f}")
    print(f"🎯 F1 Score (Weighted): {f1_weighted:.2f}")

    if not os.path.exists("static"):
        os.makedirs("static")

    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[label_names[i] for i in np.unique(y_true_labels)],
                yticklabels=[label_names[i] for i in np.unique(y_true_labels)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"static/{save_prefix}_confusion_matrix.png")
    plt.close()

    # Bar Chart
    df_result = pd.DataFrame({
        "Actual": [label_names[i] for i in y_true_labels],
        "Predicted": [label_names[i] for i in y_pred_labels]
    })
    result_summary = df_result.groupby(["Actual", "Predicted"]).size().unstack(fill_value=0)
    result_summary.plot(kind='bar', figsize=(12, 6))
    plt.title("Actual vs Predicted Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"static/{save_prefix}_actual_vs_predicted.png")
    plt.close()

    # Classification Report
    report = classification_report(
        y_true_labels, y_pred_labels,
        labels=np.unique(y_true_labels),
        target_names=[label_names[i] for i in np.unique(y_true_labels)],
        zero_division=0
    )
    print("\n📄 Classification Report:\n", report)
    with open("static/CNN_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Classification Report\n\n")
        f.write(report)

    with open("static/metrics_summary.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"F1 Score (Macro): {f1_macro:.2f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.2f}\n")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def process():
    labels, images = data_read("D:/Projectcode/Signlanguage")
    label_cats = sorted(np.unique(labels))
    label_map = {label: idx for idx, label in enumerate(label_cats)}
    pd.DataFrame({'Label': label_cats, 'Encoded': list(label_map.values())}).to_csv('label_encoded.csv', index=False)

    y = np.array([label_map[l] for l in labels]).reshape(-1, 1)
    X = np.array(images).astype("float32") / 255.0

    onehot = OneHotEncoder(sparse=False)
    y_encoded = onehot.fit_transform(y)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, val_idx in split.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=False
    )
    datagen.fit(X_train)

    # Model
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(45, 45, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save training plot
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Custom_CNN Train vs Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/Custom_CNN_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Custom_CNN Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/Custom_CNN_loss.png")
    plt.close()

    # Save model
    model.save("CNN1.h5")
    print("✅ Model saved as 'trained1.h5'")

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, label_names=label_cats, save_prefix="cnn")
    print("📌 Final Metrics:", metrics)

# Optional: Run the training
# process()
