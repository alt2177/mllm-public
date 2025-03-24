# Author: Austin
# Shared constants used

# Put your HF credentials here!
HF_ACCESS_TOKEN = None
HF_USERNAME = None

# CONFIG_DIR_PATH = Path(__file__).resolve() / "experiment_configs" / 'configs'
DRUG_REVIEW_DATASET_PATH: str = "lewtun/drug-reviews"
YELP_REVIEW_DATASET_PATH: str = "yelp_review_full"
PROB_CSV_FILE_NAME: str = "{}/{}_{}_probabilities.csv"
ACC_FILE_NAME: str = "{}/{}_accuracy.txt"
TEST_LOSS_EXP_DIR: str = "test_loss_exp"
RESULTS_FILE_NAME: str = "model_results.txt"
EMPTY_STRING: str = ""
SEED: int = 467
TOKENIZE_MAX_LENGTH: int = 1024
APPEND_MODE: str = "a" 
WRITE_MODE: str = "w"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_FILE_PREFIX_EXP: str = r"\/(.*)"
TEXT_KEY: str = "text"
LABEL_KEY: str = "label"
TRUE_LABEL_COL: str = "true_label"


MODEL_PATHS: list[str] = [
    f"{HF_USERNAME}/gpt2_f_experiment_0",
    f"{HF_USERNAME}/gpt2_f_experiment_1",
    f"{HF_USERNAME}/gpt2_f_experiment_2",
    f"{HF_USERNAME}/gpt2_f_experiment_3",
    f"{HF_USERNAME}/gpt2_f_experiment_4",
    f"{HF_USERNAME}/gpt2_f_experiment_5",
    f"{HF_USERNAME}/gpt2_f_experiment_6",
    f"{HF_USERNAME}/gpt2_f_experiment_7",
    f"{HF_USERNAME}/gpt2_f_experiment_8",
    f"{HF_USERNAME}/gpt2_f_experiment_9",
    f"{HF_USERNAME}/gpt_f_experiment_large",
    f"{HF_USERNAME}/gpt2_m_experiment_dare_linear",
    f"{HF_USERNAME}/gpt2_m_experiment_ties",
    f"{HF_USERNAME}/gpt2_m_experiment_linear",
    f"{HF_USERNAME}/gpt2_m_experiment"       # note that this is DARE TIES
] 

# get the paths for drug dataset models
DRUG_MODEL_PATHS: list[str] = [
    f"{HF_USERNAME}/gpt2_f_experiment_0_drug_data",
    f"{HF_USERNAME}/gpt2_f_experiment_1_drug_data",
    f"{HF_USERNAME}/gpt2_f_experiment_2_drug_data",
    f"{HF_USERNAME}/gpt2_f_experiment_3_drug_data",
    f"{HF_USERNAME}/gpt2_f_experiment_4_drug_data",
    f"{HF_USERNAME}/gpt2_f_experiment_drug_data_large",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_linear",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_ties",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_dare_linear",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_dare_ties",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_dare_linear_test",
    f"{HF_USERNAME}/gpt2_m_experiment_drug_data_ties_test"
]

# EOF