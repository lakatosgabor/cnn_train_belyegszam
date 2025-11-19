import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2
import time

start_time = time.time()
#early_stop = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max')

dataset_size = 'SMALL_balanced_dataset_2400'
#path = r"E:\PythonProjects\ocr\adathalmaz_letrehozas\balanced_dataset_" + str(dataset_size)

path = r"E:\PythonProjects\ocr\adathalmaz_letrehozas\SMALL_balanced_dataset_2400"


img_size = 28
num_classes = 10

data = []
labels = []

# 2. Képek beolvasása, csak szűrkeárnyalat
for digit in range(num_classes):
    digit_path = os.path.join(path, str(digit))
    files = [f for f in os.listdir(digit_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imgs = []
    for fname in files:
        img = load_img(os.path.join(digit_path, fname), color_mode='grayscale', target_size=(img_size, img_size))
        img_array = img_to_array(img)
        imgs.append(img_array)

    # Adatok és címkék hozzáadása minden képhez
    data.extend(imgs)
    labels.extend([digit] * len(imgs))

print(f"Adathalmaz mérete: {len(data)} kép, {len(labels)} címke")

# 3. Normalizálás és átalakítás
data = np.array(data, dtype='float32') / 255.0
data = data.reshape(-1, img_size, img_size, 1)
labels = np.array(labels)

# 4. Adathalmaz szétválasztása
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# 5. One-hot kódolás
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 6. Példaképek megjelenítése
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i].reshape(img_size, img_size), cmap='gray')
    plt.title(np.argmax(y_train[i]))
plt.tight_layout()
plt.show()

# 7. CNN modell létrehozása
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# 8. Modell fordítása és tanítása
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    'best_model' + str(dataset_size) + '.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=130,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"A kód futási ideje: {elapsed_time:.2f} másodperc")

# 9. Tanítási eredmények ábrázolása
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

best_model = load_model('best_model' + str(dataset_size) + '.h5')
score = best_model.evaluate(x_test, y_test, verbose=0)
print('📉 Best Test loss:', score[0])
print('✅ Best Test accuracy:', score[1])
