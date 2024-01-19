import os
import json
import wandb

import sys

if len(sys.argv) < 6:
    print(
        """
Please provide a platform, backend, gpu name and model type as an argument
e.g. "python wandb-export.py dataset browser webgl nvidia-a100 gpt-nano", 
    
    allowed datasets are: "wikitext-103", "tiny-shakespeare"
    allowed platforms are: "browser", "node"
    allowed backends are: "cpu", "webgl", "wasm", "webgpu", "tensorflow"
    allowed model types are: "gpt-nano", "gpt-micro", "gpt-mini", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    """
    )
    exit()

dataset = sys.argv[1]
platform = sys.argv[2]
backend = sys.argv[3]
gpu = sys.argv[4]
model = sys.argv[5]

file_name = f"disco_{dataset}_{platform}_{backend}_{gpu}_{model}.json"
path = os.path.join(os.path.dirname(__file__), file_name)
print("Loading file:", path)
with open(path, "r") as f:
    save = json.load(f)

init = save["init"]
init["platform"] = platform
init["backend"] = backend
init["gpu"] = gpu
init["model"] = model
wandb.init(config=init, project="disco-benchmarks-2", name=file_name)
for log in save["logs"]:
    wandb.log(log)

wandb.finish()
