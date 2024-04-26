import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import time
logger = logging.getLogger(__name__)

# Assume these global variables are needed as per your initial setup
# It's generally not best practice to rely on global variables like this, 
# but it will depend on the hosting environment specifics
model = None
tokenizer = None
device = None
generation_config = None

def model_fn(model_dir):
    """
    Load the model and tokenizer for inference.
    """
    global model, tokenizer, device, generation_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer from pre-trained or your fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logger.debug("Correction generation model from path {0} loaded successfully".format(model_dir))
    
    # Load generation configuration if exists
    generation_config_path = os.path.join(model_dir, "generation_config.json")
    if os.path.isfile(generation_config_path):
        with open(generation_config_path) as f:
            generation_config = json.load(f)
    else:
        generation_config = {}

    return {"model": model, "tokenizer": tokenizer, "device": device, "generation_config": generation_config}

def input_fn(request_body, request_content_type):
    """
    Parse the input data.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
    else:
        raise ValueError('Unsupported content type: {}'.format(request_content_type))

    # ':' 이후의 내용만 추출하기 위한 전처리
    processed_input_data = []
    for item in input_data:
        # "data" 키 또는 "body" 키를 기반으로 값을 가져옴
        data_field = item.get("data", "") if "data" in item else item.get("body", "")
        # ':'를 기준으로 문자열을 분리하고, 두 번째 부분을 추출. ':'가 없는 경우 전체 문자열 사용
        split_data = data_field.split(':', 1)
        if len(split_data) > 1:
            # ':' 이후의 내용만 추출
            content_after_colon = split_data[1].strip()
        else:
            # ':'가 없는 경우, 전체 데이터 사용
            content_after_colon = split_data[0].strip()
        processed_input_data.append({"data": content_after_colon})

    return processed_input_data


def predict_fn(input_data, model_env):
    """
    Generate predictions based on the input data.
    """
    start_time = time.time() # 전체 실행 시간 측정 시작

    try:
        text = [item["data"] if "data" in item else item["body"] for item in input_data]
    except KeyError as e:
        raise ValueError("Invalid input format") from e
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

    max_length = 1024 # 추가

    # Use the generation config if provided
    if generation_config:
        outputs = model.generate(**inputs, **generation_config, max_length=max_length)
    else:
        outputs = model.generate(**inputs, max_length=max_length)

    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    final_outputs = postprocessing(output)

    end_time = time.time() # 전체 실행 시간 측정 종료
    print('Total prediction function time: {:.6f} seconds'.format(end_time - start_time))

    return final_outputs


def output_fn(prediction_output, accept='application/json'):
    """
    Format the predictions into the desired format.
    """
    if accept == 'application/json':
        return json.dumps({"predictions": prediction_output})
    else:
        raise ValueError('Unsupported accept type: {}'.format(accept))

def postprocessing(pred):
    """
    Post-process the model predictions to extract the last final score from the predictions.
    """
    pattern = r"Final Score: (\d+)"
    final_score = 0  
    
    for prediction in pred:
        matches = re.findall(pattern, prediction)
        if matches:
            final_score = int(matches[-1])

    return {"final_score": final_score}
