import os
import cv2
import shap
import numpy as np
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score


folder0 = 'C:/Users/jeeveth/Desktop/laryngeal dataset/Hbv'
folder1 = 'C:/Users/jeeveth/Desktop/laryngeal dataset/He'
folder2 = 'C:/Users/jeeveth/Desktop/laryngeal dataset/IPCL'
folder3 = 'C:/Users/jeeveth/Desktop/laryngeal dataset/Le'

images = []
labels = []
for folder_name, label in [(folder0, 0), (folder1, 1), (folder2, 2), (folder3, 3)]:
    for filename in os.listdir(folder_name):
        img_path = os.path.join(folder_name, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(label)

# Convert lists to arrays
images = np.array(images)
labels = np.array(labels)
#labels = to_categorical(labels)

# Perform train-test split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization

# Load the pre-trained VGG19 model
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in vgg19_model.layers:
    layer.trainable = False

# Create a new model with additional layers
model = Sequential()
model.add(vgg19_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_summary_string = get_model_summary(model)

print(model_summary_string)
history = model.fit(x_train, y_train_cnn, batch_size=64,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test_cnn))

test_loss, test_acc = model.evaluate(x_test, y_test_cnn, verbose=0)
train_loss, train_acc = history.history['loss'][-1], history.history['accuracy'][-1]

print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)
print('Train loss:', train_loss)
print('Test loss:', test_loss)

# Predict on test data
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels to categorical labels
y_test_labels = np.argmax(y_test_cnn, axis=1)

# Calculate evaluation metrics
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
roc_auc = roc_auc_score(y_test_cnn, y_pred, multi_class='ovr')

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Cohen\'s Kappa:', kappa)
print('ROC AUC:', roc_auc)

# Plot ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred[:, i])
    roc_auc[i] = roc_auc_score(y_test_cnn[:, i], y_pred[:, i])

plt.figure()
colors = ['blue', 'red', 'green', 'orange']
labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
for i in range(4):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=labels[i])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()