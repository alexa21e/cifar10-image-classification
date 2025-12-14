from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

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

        # A 2nd Conv2D layer learns more complex features like shapes and textures(abstract features) using 64 filters
        # Increases the complexity of learned features by combining with the output of the first layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten() converts the 2D feature maps into a 1D feature vector

        # It is necessary to transition from the convolutional layers to 
        # the standard fully connected (Dense) layers for classification
        Flatten(),

        # Dense() is a fully connected layer with 100 neurons and relu activation

        # Performs high-level reasoning on the extracted 1D features before classification
        Dense(100, activation='relu'),

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

if __name__ == '__main__':
    (features_train, labels_train), (features_test, labels_test) = load_and_preprocess_cifar10()
    assert features_train.shape == (50000, 32, 32, 3)
    assert labels_train.shape == (50000, 10)
    assert features_train.max() <= 1.0 and labels_train.min() >= 0.0
    print("Data shapes and normalization verified successfully!")

    create_baseline_cnn()
    model = create_baseline_cnn()
    model.summary()
