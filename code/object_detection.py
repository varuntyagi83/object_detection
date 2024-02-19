# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# Create a function to generate synthetic images of objects
def generate_synthetic_images(num_images, object_list, image_size):
    """
    Generates synthetic images of objects.

    Args:
        num_images: The number of images to generate.
        object_list: A list of objects to generate images of.
        image_size: The size of the images to generate.

    Returns:
        A list of synthetic images and their corresponding labels.
    """

    # Create a list to store the images and labels
    images = []
    labels = []

    # Iterate over the number of images to generate
    for i in range(num_images):

        # Choose a random object from the list
        object_index = np.random.randint(len(object_list))
        object_name = object_list[object_index]

        # Create a random bounding box for the object
        x1 = np.random.randint(0, image_size - 1)
        y1 = np.random.randint(0, image_size - 1)
        x2 = np.random.randint(x1 + 1, image_size)
        y2 = np.random.randint(y1 + 1, image_size)

        # Create a random color for the object
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # Create a synthetic image of the object
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[y1:y2, x1:x2] = color

        # Add the synthetic image and its label to the list
        images.append(image)
        labels.append(object_index)

    # Return the list of synthetic images and labels
    return images, labels

# Generate synthetic dataset for training
num_images = 1000
object_list = ['car', 'person', 'bicycle']
image_size = 224
synthetic_images, synthetic_labels = generate_synthetic_images(num_images, object_list, image_size)

# Create a synthetic dataset for training
train_data_generator = ImageDataGenerator(rescale=1./255)
train_dataset = train_data_generator.flow(np.array(synthetic_images), np.array(synthetic_labels), batch_size=32, shuffle=True)

# Create a MobileNetV2 model
base_model = MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False)

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(len(object_list), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Generate synthetic dataset for predictions
num_images = 100
synthetic_images_predictions, synthetic_labels_predictions = generate_synthetic_images(num_images, object_list, image_size)

# Reshape the input array
synthetic_images_predictions = np.array(synthetic_images_predictions).reshape((-1, image_size, image_size, 3))

# Make predictions on the synthetic dataset
predictions = model.predict(synthetic_images_predictions / 255.0)

# Print the predictions
for i in range(len(predictions)):
    print(f'Prediction for image {i}: {object_list[np.argmax(predictions[i])]}')

# Generate predictions for the entire test dataset
test_dataset = train_data_generator.flow(np.array(synthetic_images_predictions), np.array(synthetic_labels_predictions), batch_size=32, shuffle=False)
predictions = model.predict(test_dataset)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
acc = accuracy_score(synthetic_labels_predictions, predicted_labels)

# Create the confusion matrix
conf_matrix = confusion_matrix(synthetic_labels_predictions, predicted_labels, labels=np.arange(len(object_list)))

# Display the confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=object_list)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}')
plt.show()
