import json
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import tflite_runtime.interpreter as tflite

def download_image(url):
    """Downloads and preprocesses an image."""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    
    # Convert to RGB and resize
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((200, 200), Image.NEAREST)
    
    # Convert to numpy array and preprocess
    x = np.array(img, dtype=np.float32)
    x /= 255.0
    return np.expand_dims(x, axis=0)  # Add batch dimension

def lambda_handler(event, context):
    try:
        # Load model
        interpreter = tflite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        
        # Parse image URL from event
        image_url = event.get('image_url')
        if not image_url:
            raise ValueError("Missing 'image_url' in event data")
        
        # Download and preprocess image
        X = download_image(image_url)
        
        # Run inference
        interpreter.set_tensor(input_index, X)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_index)
        prediction = float(preds[0][0])
        
        # Return the prediction in JSON format
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": prediction})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
