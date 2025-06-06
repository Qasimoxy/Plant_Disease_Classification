import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import Loss
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.saving import register_keras_serializable

# Register the function so it can be deserialized
@register_keras_serializable()
def preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


# Rebuild FocalLoss class
class FocalLoss(Loss):
    def __init__(self, gamma=2., alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if not self.from_logits:
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)



# ----- Configuration -----
IMG_SIZE = (224, 224)  
PLANET_NAMES = {
    "cashew_anthracnose": "Cashew",
    "cashew_gumosis": "Cashew",
    "cashew_healthy": "Cashew",
    "cashew_leaf_miner": "Cashew",
    "cashew_red_rust": "Cashew",
    "cassava_bacterial_blight": "cassava",
    "cassava_brown_spot": "cassava",
    "cassava_green_mite": "cassava",
    "cassava_healthy": "cassava",
    "cassava_mosaic": "cassava",
    "maize_fall_armyworm": "maize",
    "maize_grasshoper": "maize",
    "maize_healthy": "maize",
    "maize_leaf_beetle": "maize",
    "maize_leaf_blight": "maize",
    "maize_leaf_spot": "maize",
    "maize_streak_virus": "maize",
    "tomato_healthy": "tomate",
    "tomate_leaf_blight": "tomate",
    "tomate_leaf_curl": "tomate",
    "tomate_septoria_leaf_spot": "tomate",
    "tomate_verticulium_wilt": "tomate"
}

# Load Model 
MODEL_PATH ="Plant_disease_detection_model.h5" 
# Then load with the custom object
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss,
                                                                "preprocess_input": preprocess_input}, compile=False)

# Load Image from Argument 
if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]

#  Preprocess Image 
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)  # Make it batch size 1
img_array = preprocess_input(img_array) # Normalize if needed

# ----- Predict -----
predictions = model.predict(img_array)
predicted_class_name = np.argmax(predictions)
print(f"Predicted class name: {predicted_class_name}")
predicted_class = list(PLANET_NAMES.keys())[predicted_class_name]

predicted_planet = PLANET_NAMES[predicted_class]

print(f"\033[ Predicted class: {predicted_class} ({predicted_planet})\033[0m")
