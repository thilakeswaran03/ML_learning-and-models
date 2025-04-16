import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt # type: ignore
import tensorflow_datasets as tfds # type: ignore

# Load the dataset with TensorFlow Datasets
(train_ds, val_ds), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],  # 80% training, 20% validation
    shuffle_files=True,
    as_supervised=True,  # returns (img, label)
    with_info=True
)

IMG_SIZE = (224, 224)

def format_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(format_image).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(format_image).batch(32).prefetch(tf.data.AUTOTUNE)

# Load base ResNet50 model without the top layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#base_model.trainable = False  # Freeze the base model
# Unfreeze last few layers of ResNet
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Add custom layers on top of base model
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Dense → BatchNorm → Activation (ReLU)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Dropout for regularization
x = Dropout(0.5)(x)

# Final sigmoid output for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.show()