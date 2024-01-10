import os
import json
import wandb

import sys

sys.path.append(os.path.abspath("../../core/"))

if len(sys.argv) < 3:
    print(
        """
Please provide a platform and gpu name as an argument
e.g. "python wandb-export.py browser nvidia-a100", 
          
    allowed platforms are: "browser", "node"
    """
    )
    exit()

platform = sys.argv[1]
gpu = sys.argv[2]

file_name = f"disco_{platform}_{gpu}.json"
path = os.path.join(os.path.dirname(__file__), "wandb", file_name)
print("Loading file:", path)
with open(path, "r") as f:
    save = json.load(f)

init = save["init"]
config = init["config"]
wandb.init(config=config, project=config["wandbProject"], name=file_name)
for log in save["logs"]:
    wandb.log(log)

wandb.finish()
