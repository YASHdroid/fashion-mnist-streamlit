import cv2
import numpy as np
from keras.models import load_model

# class names (Fashion MNIST)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# load trained model
model = load_model("fashion_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
  img = img.reshape(1, 28, 28, 1)


    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return class_names[class_index]
