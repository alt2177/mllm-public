import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

from huggingface_hub import ModelCard, ModelCardData, create_repo
from jinja2 import Template
from huggingface_hub import HfApi


# Set the parameters for merge
OUTPUT_PATH = "./tam_test_merge_out_drug_data_dare_linear_test"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "merge_dare_linear.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

# Open the configuration file
with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

# Run the merge file
run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
        allow_crimes=True
    ),
)
print("Done!")

access_token = HF_ACCESS_TOKEN 
username = HF_USERNAME

# Defined in the secrets tab in Google Colab
api = HfApi(token=access_token)

MODEL_NAME = "gpt2_m_experiment_drug_data_dare_linear_test"

try:
    create_repo(f"{username}/{MODEL_NAME}", repo_type="model")
except:
    print('error creating repo for model. it probably already exists')

api.upload_folder(
    repo_id=f"{username}/{MODEL_NAME}",
    folder_path=OUTPUT_PATH,
    repo_type="model"
)
