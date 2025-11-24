from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

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

if __name__ == '__main__':
    (features_train, labels_train), (features_test, labels_test) = load_and_preprocess_cifar10()
    assert features_train.shape == (50000, 32, 32, 3)
    assert labels_train.shape == (50000, 10)
    assert features_train.max() <= 1.0 and labels_train.min() >= 0.0
    print("Data shapes and normalization verified successfully!")
