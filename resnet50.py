import os
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

# Define paths to the folders containing the images
folder_names = ['0', '1', '2', '3']
base_dir = 'C:/Users/jeeveth/Downloads/laryngeal dataset-20230623T072613Z-001/laryngeal dataset'

# Load images from all folders
images = []
labels = []
for label, folder_name in enumerate(folder_names):
    folder_path = os.path.join(base_dir, folder_name)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(label)

# Convert lists to arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

# Create data generators for image augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data iterators for training and testing
batch_size = 16
train_iterator = train_datagen.flow(x_train, y_train_cnn, batch_size=batch_size)
test_iterator = test_datagen.flow(x_test, y_test_cnn, batch_size=batch_size)

# Load the ResNet50 model and add a custom output layer
resnet_model = ResNet50(input_shape=(224, 224, 3), include_top=False)
for layer in resnet_model.layers:
    layer.trainable = False

x = Flatten()(resnet_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(4, activation='relu')(x)
predictions = Dense(len(folder_names), activation='softmax')(x)
model = Model(inputs=resnet_model.input, outputs=predictions)

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
# Train the model
model.fit(train_iterator, epochs=50, steps_per_epoch=len(x_train)//batch_size, validation_data=test_iterator, validation_steps=len(x_test)//batch_size)

# Evaluate the model on train and test sets
train_loss, train_accuracy = model.evaluate(train_iterator, steps=len(x_train)//batch_size)
test_loss, test_accuracy = model.evaluate(test_iterator, steps=len(x_test)//batch_size)

print('Train accuracy:', train_accuracy)
print('Train loss:', train_loss)
print('Test accuracy:', test_accuracy)
print('Test loss:', test_loss)
# Train accuracy: 0.5672348737716675
# Train loss: 0.951148509979248
# Test accuracy: 0.51953125
# Test loss: 1.0596365928649902
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