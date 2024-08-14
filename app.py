from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from nlp_module import generate_description   

# Initialize Flask app
app = Flask(__name__)
app.static_folder = "static"

# Load the trained models (excluding ResNet50)
models = {
    "InceptionV3": load_model(r"C:\Mushroom\model\InceptionV3_Model.h5"),
    "DenseNet121": load_model(r"C:\Mushroom\model\DenseNet121_Model.h5"),
    "Xception": load_model(r"C:\Mushroom\model\Xception_Model.h5")
}

# Define image size (must match the size used during training)
img_size = (224, 224)

# Define class names
class_names = ["Agaricus bisporus", "Agrocybe aegerita", "Agaricus blazei Murill", 
               "Armillaria mellea", "Auricularia auricula", "Auricularia polytricha", "Boletus"]

# Define a route to render the upload page
@app.route("/")
def index():
    return render_template("upload.html")

# Define a route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Check if the image is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Save the uploaded file to the static folder
    file_path = os.path.join(app.static_folder, file.filename)
    file.save(file_path)
    
    # Load and preprocess the image
    img = load_img(file_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Get predictions from each model
    predictions = [model.predict(img_array) for model in models.values()]
    
    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)
    
    # Get the final predicted class
    predicted_class = np.argmax(avg_prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    
    # Get the accuracy of the prediction
    accuracy = np.max(avg_prediction) * 100
    
    # Generate detailed information about the predicted class
    detailed_info = generate_description(predicted_class_name)
    
    # Render the result page with the uploaded image, prediction, and detailed information
    return render_template('result.html', 
                           image=file.filename, 
                           predicted_class=predicted_class_name, 
                           accuracy=accuracy,
                           detailed_info=detailed_info)

if __name__ == '__main__':
    app.run(debug=True)
