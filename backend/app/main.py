echo "from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('models/gesture_recognition_model.h5')

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    return JSONResponse(content={"prediction": str(predicted_class)})
" > backend/app/main.py
