import starlette
import boto3    
import os
import random
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ray import serve
import logging

ray_serve_logger = logging.getLogger("ray.serve")
BUCKET = 'nonsensitive-data'
REGION = 'us-east-1'
S3_DIRECTORY = 'phi3_finetuned'
MODEL_LOCAL_DIR = '/tmp/phi3'
DEVICE = 'cpu'

def download_directory_from_s3(access_key, secret_key, region, bucket_name, s3_directory, local_directory):
    ray_serve_logger.warning("Start Model downloading ..")
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Ensure the local directory exists
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Iterate over objects in the S3 directory
    for obj in bucket.objects.filter(Prefix=s3_directory):
        target = os.path.join(local_directory, os.path.relpath(obj.key, s3_directory))

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if obj.key.endswith('/'):
            continue  # Skip directories, only download files

        bucket.download_file(obj.key, target)
        ray_serve_logger.warning(f"Model downloaded {obj.key} to {target}")


def load_model(model_path):
    ray_serve_logger.warning("Start Model loading ..")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    compute_dtype = torch.float32
    device = torch.device(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True
    )
    ray_serve_logger.warning(f"Model was loaded successfully.")
    return model, tokenizer


def get_next_word_probabilities(sentence, tokenizer, device, model, top_k=2):

    # Get the model predictions for the sentence.
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)  # .cuda()
    outputs = model(inputs)
    predictions = outputs[0]


    # Get the next token candidates.
    next_token_candidates_tensor = predictions[0, -1, :]

    # Get the top k next token candidates.
    topk_candidates_indexes = torch.topk(
        next_token_candidates_tensor, top_k).indices.tolist()

    # Get the token probabilities for all candidates.
    all_candidates_probabilities = torch.nn.functional.softmax(
        next_token_candidates_tensor, dim=-1)

    # Filter the token probabilities for the top k candidates.
    topk_candidates_probabilities = \
        all_candidates_probabilities[topk_candidates_indexes].tolist()

    # Decode the top k candidates back to words.
    topk_candidates_tokens = \
        [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

    # Return the top k candidates and their probabilities.
    return list(zip(topk_candidates_tokens, topk_candidates_probabilities))


@serve.deployment
class Translator:
    def __init__(self):
        self.device = DEVICE
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        download_directory_from_s3(aws_access_key_id, aws_secret_access_key, REGION, BUCKET, S3_DIRECTORY, MODEL_LOCAL_DIR)
        self.model, self.tokenizer = load_model(MODEL_LOCAL_DIR)

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        re = 'NO DATA - missing text field'
        if 'text' in req:
            sentence = req['text']
            re = get_next_word_probabilities(sentence, self.tokenizer, self.device, self.model, top_k=2)
        else:
            ray_serve_logger.warning(f"Missing text field in the json  request = {req}")
        return re



#app = Translator.options(route_prefix="/translate").bind()
app = Translator.bind()


