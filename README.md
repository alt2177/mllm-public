# Merged Large Language Models (MLLM): Blending Ensemble Learning and Deep Learning

Merging models has come about as a popular new paradigm for improving pretrained model
performance at lower paramaters and without the need to retrain. There are several ways
to merge models and a significant amount of work right now going into testing
different combinations of models to find out what can bring about large improvements.

MLLM is a framework for evaluating different merged, pre-trained LLMs.
We aim to benchmark and evaluate a Merge of Large Language Models (MLLM) that leverages several medium sized LLMs (~100 million parameters each)
to attempt to match the performance of larger, state-of-the-art LLMs. We focus on classification tasks and classification performance on a particular domain, in our case on the
[Drug Review Dataset (Drugs.com) via UCI ML Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com).

**Authors:**

- Austin Tao ([austin.tao@berkeley.edu](mailto:austin.tao@berkeley.edu))
- Robert Thompson ([robert_thompson@berkeley.edu](mailto:robert_thompson@berkeley.edu))
- Phudish (Tam) Prateepamornkul ([phudish_p@berkeley.edu](mailto:phudish_p@berkeley.edu))
- Sean McAvoy ([sean_mcavoy@berkeley.edu](mailto:sean_mcavoy@berkeley.edu))

## Technologies

We use [MergeKit](https://github.com/arcee-ai/mergekit) to run our LLM merging operations. Specifically,
we evalaute three merge techniques:

- Linear Merge [Wortsman et al.](https://doi.org/10.48550/arXiv.2203.05482)
- TIES Merge [Yadav et al.](https://doi.org/10.48550/arXiv.2306.01708)
- DARE + TIES/Linear Merge [Yu et al.](https://doi.org/10.48550/arXiv.2311.03099)

For the base LLM, we use GPT-2 [Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) for
its relatively small size and moderate base performance.

## Datasets

So far we are evaluating [FLAN](https://huggingface.co/docs/transformers/model_doc/flan-t5) pretrained models on Huggingface, and are considering merging with smaller iterations of [LLama](https://huggingface.co/meta-llama) and some other more specific models
We specifically will be assessing summarization performance on a particular domain, in our case on the
[Drug Review Dataset (Drugs.com) via UCI ML Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com). We also use two other data sets, [summarized in this table](#descriptive-statistics)

### <a name="descriptive-statistics"></a>Table: Descriptive Statistics Datasets

| **Name**    | **Average Word Count** | **Median Word Count** | **Word Count Std Dev** |
| ----------- | ---------------------- | --------------------- | ---------------------- |
| **Yelp**    | 134.098089             | 99.0                  | 121.396115             |
| **Drug**    | 84.699802              | 84.0                  | 45.044833              |
| **Android** | 11.775620              | 5.0                   | 17.058663              |

<!--
# [**IN PROGRESS**] Installation Instructions

All of our dependencies are specified in the `requirements.txt` file. To run any of
our experiments, you will want to create a virtual enviromnent and install dependencies.
If you are on a remote machine that uses `slurm` to schedule and run jobs, you can simply
run

```
sbatch bin/run_experiment.sh
```

which will take care of all dependency installations to execute an example script.
Otherwise, run the following to set up a local venv:

```{bash}
python -m venv mllm_venv
source mllm_venv/bin/activate
pip install -r ./requirements.txt
```

## Usage

If you wish to run your job using `slurm`, run the following command from the top-level
directory:

```{bash}
sbatch bin/{NAME_OF_YOUR_SBATCH_FILE}.sh
```

Note that if you try to run this command while in the `bin` directory, you will have to
make sure your `.sh` script takes that into account and adjusts your `pwd` as necessary.

As an example, `sbatch bin/run_experiment.sh` will run the following command and output
its results to `bin/outputs`, as defined in the header:

```{bash}
## Resource Requests
#SBATCH --output=./bin/outputs/output-%j.log

...

python -m src.eval_scripts.eval_gpt_linear_merge
```

Simply change the module to be used in the `.sh` script for your experiment.

## Creating New Experiments

Experiments are set up through config files in `src/configs`. To define your own, create a new
`config.py` file in `src/configs`.

```{yaml}
# An example config file

config:
  dataset:
    name: "imdb"
  model:
    name: "openai-community/gpt2"
    num_labels: 2
  training_args:
    output_dir: "imdb_finetune_gpt2_test_epoch_1_with_prob_1_merge_output"
    learning_rate: 2e-5
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
    num_train_epochs: 1
    weight_decay: 0.01
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    push_to_hub: true
    hub_model_id: "imdb_finetune_gpt2_test_epoch_1_with_prob_1_merge_output"
  mergekit_config:
    models:
      - model: gpt2_f_experiment_0_drug_data
        parameters:
  # any other specifications you want

```

Once you have defined your new config, create a new module in `src/` and specify your new
config path:

```{python}
# Example: for eval_gpt2_linear_merge.py in src/eval_scripts/

CONFIG_YML = CONFIG_DIR_PATH / 'example_config.yml'     # change this to be your config file

# load your config
config = load_yaml_config(CONFIG_YML, 'r')


do_stuff():
    stuff(config["training_args"])

if __name__ == "__main__":
    do_stuff()

```

Afterwards, you can run your experiment with the `.sh` script as explained above.
-->
