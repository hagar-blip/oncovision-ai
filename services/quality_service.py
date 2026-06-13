import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/quality_gate_best.h5"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Low_Quality_Image",
    "Non_Medical_Image",
    "Valid_Medical_Image"
]

def predict_quality(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img, verbose=0)

    idx = int(np.argmax(pred[0]))
    conf = float(np.max(pred[0]))

    return {
        "class": CLASS_NAMES[idx],
        "confidence": round(conf * 100, 2)
    }