import json
import torch
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Global variables to hold the model and tokenizer after initialization
model = None
tokenizer = None
device = None
generation_config = None

def postprocessing(dialogue: list, corrections: list):
    special_tokens = ["[COPY]", "[EDIT]", "[FILTER]"]
    outputs = []
    utterances = [d['data'].split("[Improve]Student:")[-1].strip() for d in dialogue]  # extract student's last utterance (after [Improve] token)
    print('utterances : ', utterances)

    for utter, corr in zip(utterances, corrections):
        # Clean up the correction output
        output = corr.replace("<pad>", '').replace("</s>", '').strip()

        # Identify the special token if present
        try:
            token = [st for st in special_tokens if st in output][0]
        except IndexError:
            token = ''
        
        only_correction = output.replace(token, '')

        # Remove(ignore) special character
        replace_dict = {s: '' for s in ['.', ',', '\'', '\"', '/']}
        utter = utter.lower().translate(str.maketrans(replace_dict)).strip()
        print('utter : ', utter)
        corr = only_correction.lower().translate(str.maketrans(replace_dict)).strip()
        print('corr : ', corr)


        # Maintain consistency of (special token) and (generated sentence)
        if corr == '':  # case of filtering
            output = special_tokens[2]  # return '[FILTER]'
        elif utter == corr:  # case of copying the sentence
            new_token = special_tokens[0]
            # output = utterance
            output = output.replace(only_correction, utter)

        else:  # case of editing the sentence
            if token == '[COPY]':  # follow the COPY token and copy the sentence
                output = output.replace(only_correction, utter)
            elif token == '[FILTER]':
                return special_tokens[2]  # return '[FILTER]'
            else:
                return output.strip()  # Keep the edited correction
            new_token = token
        output = output.replace(token, new_token + " ") if token else output
        outputs.append(output)

    return outputs


def model_fn(model_dir):
    global model, tokenizer, device, generation_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    generation_config_path = os.path.join(model_dir, "generation_config.json")

    if os.path.isfile(generation_config_path):
        with open(generation_config_path) as f:
            generation_config = json.load(f)
    else:
        generation_config = None

    return {"model": model, "tokenizer": tokenizer, "device": device, "generation_config": generation_config}

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError('Unsupported content type: {}'.format(request_content_type))

def predict_fn(input_data, model_handler):
    global model, tokenizer, device, generation_config
    
    if input_data is None:
        return None
    print('input_data', input_data)
    # Preprocess
    try:
        text = [item["data"] if "data" in item else item["body"] for item in input_data]
    except KeyError as e:
        raise ValueError("Invalid input format") from e
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Inference
    if generation_config:
        outputs = model.generate(**inputs, **generation_config)
    else:
        outputs = model.generate(**inputs)
    
    corrections = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # PostProcess
    final_outputs = postprocessing(input_data, corrections)
    
    return final_outputs

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps({"generated_text": prediction})
    else:
        raise ValueError('Unsupported content type: {}'.format(content_type))
