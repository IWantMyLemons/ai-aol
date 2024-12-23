from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import base64
import pickle 

app = Flask(__name__)

model = tf.keras.models.load_model('aol-ocr.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_image(image_b64, min_width=20, min_height=10, aspect_ratio_range=(0.2, 5)):
    """
    Decode a base64 image, resize it to 224x224, and normalize pixel values.
    """
    # Decode the base64 image
    image_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Perform dilation to close gaps between edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter text-like regions
    cropped_texts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # Apply size and aspect ratio filters
        if w >= min_width and h >= min_height and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            cropped_text = img[y:y+h, x:x+w]
            cropped_texts.append(cropped_text)

    img_ready = []
    for cropped_text in cropped_texts:
        # Resize image to 224x224
        img_resized = cv2.resize(cropped_text, (224, 224))
        # Normalize pixel values to [0, 1] range
        img_normalized = img_resized / 255.0
        # Expand dimensions to match model input shape (1, 224, 224, 3)
        img_ready.append(img_normalized)

    return np.array(img_ready)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        uploaded_file = request.files['file']
        if not uploaded_file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read the file and encode it to base64
        image_data = uploaded_file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Preprocess the image
        input_images = preprocess_image(image_b64)

        # Make predictions using the OCR model
        predictions = model.predict(input_images)

        # Convert predictions to readable output (if applicable)
        predicted_text = decode_predictions(predictions)

        # Respond with the predicted text
        return jsonify({'predicted_text': predicted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


def decode_predictions(predictions):
    """
    Convert model predictions into readable text.
    This function depends on your OCR model's output format.
    """
    predicted_indices = np.argmax(predictions, axis=-1)    
    predicted_text = " ".join(tokenizer.sequences_to_texts(predicted_indices))
    return predicted_text


if __name__ == '__main__':
    app.run(debug=True)
