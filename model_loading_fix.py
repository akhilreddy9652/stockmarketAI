
import tensorflow as tf
from tensorflow.keras import metrics

# Register custom metrics to fix loading issues
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

# Custom model loader that handles legacy issues
def load_model_safely(model_path):
    try:
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'mse': mse}
        )
        return model
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        # Rebuild model if needed
        return None
