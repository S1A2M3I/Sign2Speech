import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def image_resize(image, height=96, inter=cv2.INTER_AREA):
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
                img = image_resize(img, height=96)
                img = cv2.resize(img, (96, 96))
                images.append(img)
                labels.append(label)
    return labels, images


def evaluate_model(model, X_val, y_val, label_names, save_prefix="densenet"):
    print("🔍 Evaluating model...")
    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)


    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)


    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(f"🎯 F1 Macro: {f1_macro:.4f}")
    print(f"🎯 F1 Weighted: {f1_weighted:.4f}")


    # Create folder
    if not os.path.exists("static"):
        os.makedirs("static")


    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/densenet_confusion_matrix.png")
    plt.close()


    # Classification Report
    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=label_names,
        zero_division=0
    )
    with open("static/densenet_classification_report.txt", "w") as f:
        f.write(report)


    with open("static/metrics_summary.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\nF1 Macro: {f1_macro:.4f}\nF1 Weighted: {f1_weighted:.4f}\n")


    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def process():
    labels, images = data_read('D:/Projectcode/Signlanguage')
    label_cats = sorted(np.unique(labels))
    label_map = {label: idx for idx, label in enumerate(label_cats)}
    pd.DataFrame({'Label': label_cats, 'Encoded': list(label_map.values())}).to_csv("label_encoded.csv", index=False)


    y = np.array([label_map[l] for l in labels]).reshape(-1, 1)
    X = np.array(images).astype("float32") / 255.0


    onehot = OneHotEncoder(sparse=False)
    y_encoded = onehot.fit_transform(y)


    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y)


    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(y_train.shape[1], activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])


    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=5, batch_size=32, verbose=1)


    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("DenseNet121 Train vs Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/densenet_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("DenseNet121 Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/densenet_loss.png")
    plt.close()


    # Save model
    model.save("trained1.h5")
    print("✅ Model saved as 'trained_densenet121.h5'")


    # Evaluate
    evaluate_model(model, X_val, y_val, label_names=label_cats)


# Uncomment to train
# process()