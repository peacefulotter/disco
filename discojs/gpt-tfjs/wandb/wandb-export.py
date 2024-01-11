import os
import json
import wandb

import sys

if len(sys.argv) < 4:
    print(
        """
Please provide a platform and gpu name as an argument
e.g. "python wandb-export.py browser nvidia-a100 gpt-nano", 
          
    allowed platforms are: "browser", "node"
    """
    )
    exit()

platform = sys.argv[1]
gpu = sys.argv[2]
model = sys.argv[3]

file_name = f"disco_{platform}_{gpu}_{model}.json"
path = os.path.join(os.path.dirname(__file__), file_name)
print("Loading file:", path)
with open(path, "r") as f:
    save = json.load(f)

init = save["init"]
config = init["config"]
wandb.init(config=config, project="disco-benchmarks", name=file_name)
for log in save["logs"]:
    wandb.log(log)

wandb.finish()
