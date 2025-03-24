config = {
    'dataset': {
        'name': "<DATASET_NAME>",               # name of the dataset. Should match a corresponding variable name found in datasets/config.py
    },
    'model': {
        'name': "openai-community/gpt2",                 # name of the model
        'num_labels': 2,
        'use_auth_token': HF_ACCESS_TOKEN,  # TODO: DELETEE!
        'id2label': {0: "NEGATIVE", 1: "POSITIVE"}, 
        'label2id': {"NEGATIVE": 0, "POSITIVE": 1},
        'dir': '<MODEL_CKPT_DIR>',              # checkpoint directory where models are stored
        'bases': []                             # list of optional model paths. Empty by default
    },
    'training_args': {
        'output_dir': 'imdb_finetune_gpt2_test_epoch_1_with_prob_1_merge_output',
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 1,
        'weight_decay': 0.01,
        'evaluation_strategy': "epoch",
        'save_strategy': "epoch",
        'load_best_model_at_end': True,
        'push_to_hub': True,
        'hub_model_id': "imdb_finetune_gpt2_test_epoch_1_with_prob_1_merge_output",
        'hub_token ': HF_ACCESS_TOKEN
    }
    'merging_fn': 'match_tensors_zipit',        # matching function desired. Please see "matching_functions.py" for a complete list of supported functions.
    'eval_type': 'clip',                        # Evaluation type, whether to use clip or standard cross entropy loss
    'merging_metrics': ['covariance', 'mean'],  # Alignment Metric types desired upon which to compute merging. Please see metric_calculators.py for more details
    'upload_to_hf': True,
    'hf_output_dir': "gpt2_finetune_2_with_prob_epoch_1",
    'hf_username': "mllm-dev",
}
