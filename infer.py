#! ./.venv/bin/python
import os
from datetime import datetime
from typing import Literal, List
import re
import json
from os import getenv
from dotenv import load_dotenv
from fireworks import LLM
from pydantic import BaseModel
from threading import Thread, Semaphore
from tqdm import tqdm

load_dotenv()
threads = []

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class Request(BaseModel):
    messages: List[Message]

class Response(BaseModel):
    code: str

def parse_response(response):
    response_content = response.choices[0].message.content
    reasoning_match = re.search(r"<think>(.*?)</think>", response_content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
    json_match = re.search(r"</think>\s*(\{.*})", response_content, re.DOTALL)
    code = json.loads(json_match.group(1).strip() if json_match else "{}")
    return code, reasoning

def main(names, inputs, models):
    sem = Semaphore(value=int(os.getenv('CONCURRENCY')))
    dt = datetime.now().isoformat()
    os.mkdir(f'outputs/{dt}')
    for model in tqdm(models):
        os.mkdir(f'outputs/{dt}/{model}')
        for i, question in enumerate(tqdm(inputs)):
            os.mkdir( f'outputs/{dt}/{model}/{names[i]}')
            for j in tqdm(range(int(os.getenv('REPEAT_RUN')))):
                def get(model, i, j, question):
                    with sem:
                        llm = LLM(model=model, deployment_type='auto',
                                  api_key=getenv('FIREWORKS_API_KEY')).with_temperature(0.5)
                        c, r = parse_response(llm.chat.completions.create(
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "Response",
                                    "schema": Response.model_json_schema()
                                }
                            },
                            messages = question['messages'],
                            max_tokens = 128000
                        ))
                        c = Response.model_validate(c)
                        with open(f'outputs/{dt}/{model}/{names[i]}/answer-{j}.txt','w') as f:
                            f.write(c.code)
                        with open(f'outputs/{dt}/{model}/{names[i]}/reason-{j}.txt','w') as f:
                            f.write(r)
                threads.append(Thread(target=get, args=(model, i, j, question)))
                threads[-1].start()

if __name__ == '__main__':
    names = []
    inputs = []
    for file in os.listdir('inputs'):
        names.append(file)
        with open('inputs/'+file) as f:
            inputs.append({"messages": [
                {"role": "system", "content": "Strictly outputs the response as a JSON object with a single key \"code\""
                                              " containing the complete solution code. Example: "
                                              "{\"code\": \"[complete solution code]\"}"},
                {"role": "user", "content": f.read()}
            ]})
    if len(inputs) == 0:
        raise Exception('No questions at the input.')
    if len(getenv('MODELS')) == 0:
        raise Exception('No MODELS provided as environment variable.')
    models = getenv('MODELS').split(',')
    if not getenv('FIREWORKS_API_KEY'):
        raise Exception('No FIREWORKS_API_KEY provided as environment variable.')
    main(names, inputs, models)
