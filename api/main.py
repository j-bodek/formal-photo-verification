from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
# import uvicorn
import numpy as np 
from PIL import Image
#from keras.preprocessing import image
from numpy import asarray
from io import BytesIO
from typing import List
from fastapi.responses import HTMLResponse


app = FastAPI()


def img_to_arr(img):
        img = Image.open(BytesIO(img))

        #convert image to rgb
        img = img.convert('RGB')

        img = img.resize((224,224), Image.ANTIALIAS)

        img_arr = asarray(img) 

        # expend one dimension couse we want array of shape (1,224,224,3) instead of (224,224,3)
        img_arr = np.expand_dims(img_arr, axis = 0)
        #devide every value in array by 255 (so we get values between 0 and 1)
        img_arr = img_arr / 255.

        return img_arr


def predict(img_arr):
    
    # Load choosen model
    model = load_model('models/resnet50.h5')
          
    # make prediction with keras 
    predictions = model.predict(img_arr)
   
    return predictions


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <html>
        <head>
            <title>Formal Photo Verification API</title>
        </head>
        <body>
            <h1 style="text-align: center">Formal Photo Verification API</h1>

            <br /><br />

            <h2 style="text-align: center">How it works?</h2>
            <p style="text-align: center">
            API takes photo in bytes format and ResNet50 return array of predictions.
            </p>

            <br /><br />

            <h4 style="text-align: center">Request with one image to predict</h4>

            <pre style="margin-left: 40vh"><code class="python">
                import requests

                img = open('images/valid_1.jpg','rb')
                file = {"file": (fimg)}
                r = requests.post('https://formal-photo-verification-api.herokuapp.com/api/predict', files=file)

            </code></pre>

            <h4 style="text-align: center">Request with multiple image to predict</h4>

            <pre style="margin-left: 40vh"><code class="python">
                import requests

                img1 = open('img1_path','rb')
                img2 = open('img2_path','rb')

                files [
                ('files', (img1))
                ('files', (img2))
                ]

                r = requests.post('https://formal-photo-verification-api.herokuapp.com/api/predict', files=file)

            </code></pre>
        </body>
        </html>

    """


@app.post("/api/predict")
async def predict_image(files: List[bytes] = File(...)):

    imgs_arr = np.empty((0,224,224,3), int)

    for file in files:
        img_arr = img_to_arr(file)
        imgs_arr = np.append(imgs_arr, img_arr, axis = 0)

    pred = predict(imgs_arr)
    pred = pred.tolist()
    
    return {'items': pred}



