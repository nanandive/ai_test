from typing import Annotated,Optional
from fastapi import FastAPI, File, UploadFile, Form
from utils import visualize

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
model_path = 'mediapipe/model/efficientdet_lite0.tflite' # 모델 파일 경로 설정은 반드시 필요하다. 
base_options = python.BaseOptions(model_asset_path=model_path) #모든 task에 들어가는 기본 옵션 
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5) 
detector = vision.ObjectDetector.create_from_options(options) 
# fastapi가 만들어지기 전에 디텍터를 만들어준다. 
# -> 전역변수로 만들어 줌으로서 만들어진 디텍터를 계속 재활용하기 위함 
# 모델로드보다 서버가 먼저 뜬다. 따라서 먼저 모델 로드를 먼저 해야한다. 

app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

# 헤더에 curl의 정보들이 담겨진다. 

# 파일을 받았으면 

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read() # 동시에 실행되도 비동기로 파일을 읽을 수 있다. 
    return {"filename": file.filename, "file_size": len(contents)}

import io
from PIL import Image
import cv2

@app.post("/predict")
async def predict_api(image_file: UploadFile): # file 도 사용가능

    # 0. read bytes from http 
    contents = await image_file.read() 

    # 1. make buffer from bytes
    buffer = io.BytesIO(contents)

    # 2. decode image from buffer
    pil_img = Image.open(buffer) # 읽을 수 있도록 바이트로 전환해준다.

    # STEP 3: Load the input image. 이미지를 받을 때 파이썬 이미지를 mp로 변환해준다. 
    # image = mp.Image.create_from_file(IMAGE_FILE) # 추론이 가능한 형태로 가공해준다. 
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)) # 추론이 가능한 형태로 가공해준다. 

    # STEP 4: Detect objects in the input image. -> task는 디텍션한다. 추론한다.
    detection_result = detector.detect(image)
    return {'result':detection_result}

    # STEP 5: Process the detection result. In this case, visualize it.
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


from fastapi.responses import StreamingResponse
import base64

@app.post("/predict/img")
async def predict_api_img(image_file: UploadFile):

    # 0. read bytes from http
    contents = await image_file.read()

    # 1. make buffer from bytes
    buffer = io.BytesIO(contents)

    # 2. decode image from buffer
    pil_img = Image.open(buffer)

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(IMAGE_FILE)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # STEP 6: encode
    img_encode = cv2.imencode('.png', rgb_annotated_image)[1]
    image_stream = io.BytesIO(img_encode.tobytes())
    return StreamingResponse(image_stream, media_type="image/png")