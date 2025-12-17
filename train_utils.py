from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Images are stored as integer pixel values from 0 to 255
    # Dividing by 255, scales them on the range from 0.0 to 1.0
    # Prevents gradient explosion and speeds up learning
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # This converts integer labels into a 10-element vector
    # (e.g., 3 for 'cat') => only the 3rd index is 1
    # This is required for the categorical cross-entropy loss function
    y_train_encoded = to_categorical(y_train, num_classes=10)
    y_test_encoded = to_categorical(y_test, num_classes=10)

    return (x_train, y_train_encoded), (x_test, y_test_encoded)

def create_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    # Sequential([]) defines the model as a sequential stack of layers
    # where output from one layer feeds directly into the next
    model = Sequential([

        # Conv2D() applies 32 filters (feature detectors) of size 3x3
        # relu (Rectified Linear Unit) is the activation function

        # These layers learn the edges and the corners from the input image
        # The input form is preferred to be passed down as a separate layer
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),

        # MaxPooling2D() reduces the resolution of the feature maps by taking the maximum value in a 2x2 window

        # This reduces the spatial size from 32x32 to 16x16 
        # lowering computation and helping the model become more robust to variations in image position.
        MaxPooling2D((2, 2)),
        # Disabling 25% of the neurons in the first block in a try to prevent overfitting
        Dropout(0.25),

        # A 2nd Conv2D layer learns more complex features like shapes and textures(abstract features) using 64 filters
        # Increases the complexity of learned features by combining with the output of the first layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # A 3rd layer intended to increase the model's capacity
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten() converts the 2D feature maps into a 1D feature vector

        # It is necessary to transition from the convolutional layers to 
        # the standard fully connected (Dense) layers for classification
        Flatten(),

        # Dense() is a fully connected layer with 100 neurons and relu activation

        # Performs high-level reasoning on the extracted 1D features before classification
        Dense(256, activation='relu'),
        Dropout(0.5),

        # The final Dense layer outputs a probability distribution over the 10 classes using softmax activation
        # It outputs a dense layer with 10 neurons as the final classification 
        # softmax ensures the output values are probabilities that sum up to 1
        Dense(num_classes, activation='softmax')
    ])

    # model.compile(...) configures the learning process before training begins
    # optimizer='adam' is the algorithm used to adjust the model's weights during backpropagation to minimize the loss
    # loss='categorical_crossentropy' is the metric used to calculate the error between the prediction and the true label
    # It's the standard choice for multi-class classification with one-hot encoded labels.
    # metrics=['accuracy'] is the value displayed during and after training to easily gauge model performance.
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    return model


def load_image_for_prediction(filename, target_size=(32, 32)):
    # besides loading, the image does also have to change its size
    img = load_img(filename, target_size=target_size)

    # the image converts into an array of pixels that also need normalizing
    img_array = img_to_array(img)
    # adds the batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    return img_array

def classify_single_image(model, image_path, class_names):
    processed_img = load_image_for_prediction(image_path)
    predictions = model.predict(processed_img, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return predicted_class, confidence