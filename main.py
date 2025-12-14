from tensorflow.keras.models import load_model
import os
import sys

from train_utils import classify_single_image, CLASS_NAMES, create_baseline_cnn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = "baseline_cnn_weights.keras"
DEFAULT_TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "sample_image_2.jpeg")
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILE_NAME)

def load_cifar10_model(model_path: str):
    try:
        model = load_model(model_path)
        return model, "full_model"
    except Exception:
        model = create_baseline_cnn()
        model.load_weights(model_path)
        return model, "weights_only"

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_IMAGE_PATH
    print(f"Loading model")

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found")
        sys.exit(1)

    try:
        loaded_model, load_mode = load_cifar10_model(MODEL_PATH)
        print(f"Model loaded successfully ({load_mode} mode)")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model {e}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"FATAL ERROR: Test image not found")
        sys.exit(1)

    print(f"Predicting image")

    predicted_class, confidence = classify_single_image(loaded_model, image_path, CLASS_NAMES)

    print(f"Predicted class: {predicted_class}")
    print(f"Prediction confidence: {confidence:.2f}%")
