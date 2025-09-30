import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import Counter

# Resize image
def image_resize(image, height=96, inter=cv2.INTER_AREA):
    return cv2.resize(image, (height, height), interpolation=inter)

# Load dataset
def data_read(data_path, use_preprocessed=False):
    labels, images = [], []
    subfolder = "Gesture Image Pre-Processed Data" if use_preprocessed else "Gesture Image Data"
    full_path = os.path.join(data_path, subfolder)
    
    for label in sorted(os.listdir(full_path)):
        label_path = os.path.join(full_path, label)
        if not os.path.isdir(label_path): continue
        for image_file in os.listdir(label_path):
            img_path = os.path.join(label_path, image_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = image_resize(img, height=96)
                images.append(img)
                labels.append(label)
    return labels, images


# Evaluation
def evaluate_model(model, X_val, y_val, label_names, save_prefix="model"):
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

    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title("📊 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig(f"static/{save_prefix}_confusion_matrix.png")
    plt.close()
    
    # 📄 Save classification report and metrics summary
    if not os.path.exists("static"):
        os.makedirs("static")

    with open("static/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("📄 Classification Report\n")
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=label_names))

    with open("static/metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("✅ Accuracy: {:.2f}%\n".format(acc * 100))
        f.write("🎯 F1 Score (Macro): {:.2f}\n".format(f1_macro))
        f.write("🎯 F1 Score (Weighted): {:.2f}\n".format(f1_weighted))


    # Classification Report
    print("\n📄 Classification Report:\n")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_names, zero_division=0))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

# Training + Evaluation
def process():
    base_path = 'D:/Projectcode/Signlanguage/Dataset2'

    # ImageDataGenerator with split
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        base_path,
        target_size=(96, 96),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        base_path,
        target_size=(96, 96),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save label mapping
    label_map = {v: k for k, v in train_gen.class_indices.items()}
    label_df = pd.DataFrame({'Label': list(label_map.values()), 'Encoded': list(label_map.keys())})
    label_df.to_csv('label_encoded.csv', index=False)

    # Load model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('trained1.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Model saved as 'mobileNetV2.h5'")

    # Plotting accuracy/loss
    os.makedirs("static", exist_ok=True)
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("MobilenetV2 Train vs Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/MobileNetV2_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("MobilenetV2 Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/MobilenetV2_loss.png")
    plt.close()

    # Evaluate model on validation set
    y_true, y_pred = [], []
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        pred_batch = model.predict(X_batch)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred_batch, axis=1))
        if (i + 1) * val_gen.batch_size >= val_gen.samples:
            break

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(f"🎯 F1 Macro: {f1_macro:.2f}")
    print(f"🎯 F1 Weighted: {f1_weighted:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.title("📊 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/mobilenet_confusion_matrix.png")
    plt.close()

    # Save classification report
    with open("static/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=list(label_map.values()), zero_division=0))

    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(label_map.values()), zero_division=0))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }