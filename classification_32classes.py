import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 32

# Chemin vers le dossier contenant les 32 dossiers des classes 
DATA_DIR_32 = r'C:\Users\USER\Documents'

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR_32,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR_32,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Nombre de classes détectées : {len(class_names)}")
print("Classes :", class_names)


# Normalisation

AUTOTUNE = tf.data.AUTOTUNE

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Data Augmentation 
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")

#  Architecture CNN 
model_32 = keras.Sequential([
    data_augmentation,

    # Bloc 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloc 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloc 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloc 4 (couche supplémentaire pour 32 classes)
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classificateur
    layers.GlobalAveragePooling2D(),   # plus robuste que Flatten pour beaucoup de classes
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="CNN_32classes")

model_32.summary()

#  Compilation 
model_32.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#  Callbacks : early stopping + réduction du LR
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=8, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
)

#  Entraînement
history_32 = model_32.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

#  Evaluation
print("\n=== Résultats - 32 classes ===")
loss, acc = model_32.evaluate(val_ds, verbose=0)
print(f"Précision sur la validation : {acc*100:.2f}%")

#  Courbes d'apprentissage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CNN - 32 classes de feuilles", fontsize=13)

ax1.plot(history_32.history["accuracy"],    label="Entraînement")
ax1.plot(history_32.history["val_accuracy"],label="Validation")
ax1.set_title("Précision")
ax1.set_xlabel("Époque")
ax1.set_ylabel("Précision")
ax1.legend()
ax1.grid(True)

ax2.plot(history_32.history["loss"],    label="Entraînement")
ax2.plot(history_32.history["val_loss"],label="Validation")
ax2.set_title("Perte")
ax2.set_xlabel("Époque")
ax2.set_ylabel("Perte")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
