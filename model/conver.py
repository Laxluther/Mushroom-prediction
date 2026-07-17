import tensorflow as tf

# Convert each model
models = ["InceptionV3", "DenseNet121", "Xception"]
for model_name in models:
    model = tf.keras.models.load_model(f"{model_name}_Model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f"{model_name}.tflite", "wb") as f:
        f.write(tflite_model)