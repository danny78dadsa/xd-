import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import itertools

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def preprocessing(img, label):
    img = tf.expand_dims(img, axis=-1)
    img = tf.image.resize(img, (28,28))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds = train_ds.map(preprocessing).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(preprocessing).batch(64).prefetch(tf.data.AUTOTUNE)

for imgs, labels in train_ds.take(1):
    plt.figure(figsize=(8,2.5))
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.imshow(tf.squeeze(imgs[i]), cmap='gray')
        plt.title(int(labels[i]))
        plt.axis('off')
    plt.show()
    break

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 clases
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

epochs = 5
history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('loss')
plt.legend()
plt.grid()

plt.show()

y_pred_prob = model.predict(test_ds)  
y_pred = np.argmax(y_pred_prob, axis=1)

print("Accuracy (CNN):", accuracy_score(y_test, y_pred))
print("\nReporte (CNN):\n", classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión'):
    plt.figure(figsize=(8,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False)

for imgs, labels in test_ds.take(1):
    preds = model.predict(imgs)
    plt.figure(figsize=(10,6))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(tf.squeeze(imgs[i]), cmap='gray')
        pred_label = np.argmax(preds[i])
        true_label = int(labels[i])
        plt.title(f"P:{pred_label} / T:{true_label}", color=('green' if pred_label==true_label else 'red'))
        plt.axis('off')
    plt.show()
    break

n_knn = 5000
x_train_flat = x_train.reshape(-1, 28*28)[:n_knn] / 255.0
y_train_flat = y_train[:n_knn]
x_test_flat  = x_test.reshape(-1, 28*28)[:2000] / 255.0
y_test_knn   = y_test[:2000]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat, y_train_flat)
y_knn = knn.predict(x_test_flat)

print("Accuracy (KNN subset):", accuracy_score(y_test_knn, y_knn))
print("\nReporte (KNN subset):\n", classification_report(y_test_knn, y_knn, digits=4))

n_k = 3000
x_k = x_train.reshape(-1, 28*28)[:n_k] / 255.0
y_k = y_train[:n_k]
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(x_k)

cluster_to_label = {}
for c in range(10):
    labels_here = y_k[clusters == c]
    if len(labels_here) == 0:
        cluster_to_label[c] = -1
    else:
        vals, counts = np.unique(labels_here, return_counts=True)
        cluster_to_label[c] = vals[np.argmax(counts)]

mapped = np.array([cluster_to_label[c] for c in clusters])
print("Accuracy (KMeans mapping) en subset:", accuracy_score(y_k, mapped))
