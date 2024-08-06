import cv2
import json
import torch
import time
from ultralytics import YOLO 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

file_number = 1
weights = '/app/yolov8_epoch20_best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resize_image(img, img_size=640):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized_img = cv2.resize(img, (nw, nh))
    top_pad = (img_size - nh) // 2
    bottom_pad = img_size - nh - top_pad
    left_pad = (img_size - nw) // 2
    right_pad = img_size - nw - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return padded_img

def process_detection(results):
    processed_results = []
    for r in results:
        total_width = r.orig_shape[1]
        total_height = r.orig_shape[0]  # 이미지 높이

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()[:4]
            conf = r.boxes.conf[i].item()
            cls_id = int(r.boxes.cls[i].item())
            label = r.names[cls_id]
            
            result = {
                "name": label,
                "confidence": round(conf, 4),
                "left_x": round(x1 / total_width, 2),
                "right_x": round(x2 / total_width, 2),
                "down_y" : round(y2 / total_height, 2),  
                "up_y" : round(y1 / total_height, 2)
                                }

            processed_results.append(result)
    
    json_output = json.dumps(processed_results, indent=4)
    return json_output

def run_inference(image_data, img_size=640):
    img0 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img0, img_size)
    img = torch.from_numpy(resized_img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    model = YOLO(weights)
    results = model(img)
    results = process_detection(results)
    return results

@app.post("/infer/")
async def inference(file: UploadFile = File(...)):
    image_data = await file.read()
    results = run_inference(image_data)
    return JSONResponse(content=results)