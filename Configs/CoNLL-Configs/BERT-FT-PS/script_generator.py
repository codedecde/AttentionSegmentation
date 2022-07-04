# This is a script generator file.
# Very hacky, will be removed from the final branch
import os
import re

BASE_DIR = os.path.join("Configs/CoNLL-Configs/BERT-FT-PS")
CONF_DIR = os.path.join(BASE_DIR, "Configs")
SCRIPT_DIR = os.path.join(BASE_DIR, "Scripts")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

NUM_SCRIPTS = 1
METHODS = [
    "bert-base-multilingual-cased"
]
# drop_list = [0.2, 0.4, 0.5, 0.6, 0.8]

INP_DIMS = {
    "bert-base-multilingual-cased": 768
}

FINETUNE_DICT = {
    "bert-base-multilingual-cased": ["[8, 9, 10, 11]"]
}
OPTIMIZER_LIST = [
    f"""{{
                "type": "adam",
                "parameter_groups": [
                    [[".*bert.*"], {{"lr": 2e-5}}],
                    [[".*encoder_word.*", ".*attn.*", ".*logit.*"], {{"lr": 1e-3}}]
                ]
            }}""",
    f"""{{
                "type": "adam",
                "parameter_groups": [
                    [[".*bert.*"], {{"lr": 2e-6}}],
                    [[".*encoder_word.*", ".*attn.*", ".*logit.*"], {{"lr": 1e-3}}]
                ]
            }}""",
    f"""{{
                "type": "adam",
                "parameter_groups": [
                    [[".*bert.*"], {{"lr": 6e-7}}],
                    [[".*encoder_word.*", ".*attn.*", ".*logit.*"], {{"lr": 1e-3}}]
                ]
            }}""",
    f"""{{
                "type": "adam",
                "parameter_groups": [
                    [[".*bert.*"], {{"lr": 2e-7}}],
                    [[".*encoder_word.*", ".*attn.*", ".*logit.*"], {{"lr": 1e-3}}]
                ]
            }}""",

]
OPT_ix_to_string = {
    0: "2em5", 1: "2em6", 2: "6em7", 3: "2em7"
}
TOTAL = (len(OPTIMIZER_LIST) * sum([len(x) - 1 for x in FINETUNE_DICT.values()]) ) + 1
num_per_script = -(-TOTAL // NUM_SCRIPTS)
SCRIPT_HEADER = f"""


"""
SCRATCH = "/home/ubuntu/mount_dir/AttentionSegmentation"
scripts = [[SCRIPT_HEADER] for _ in range(NUM_SCRIPTS)]
config_count = 0
TEMP = 1.

METHOD = METHODS[0]
lowercase = "true" if "uncased" in METHOD else "false"
for FINETUNE in FINETUNE_DICT[METHOD]:
    if FINETUNE == "[]":
        OPTIMIZER_OPTS = ["adam"]
    else:
        OPTIMIZER_OPTS = OPTIMIZER_LIST
    for ix, OPT in enumerate(OPTIMIZER_OPTS):
        DIM = INP_DIMS[METHOD]
        script_no = config_count % NUM_SCRIPTS
        raw = f"""
        {{
          "base_output_dir": "{SCRATCH}/Experiments/CoNLL/BERT-Finetune-2/Dir-{script_no}",
          "dataset_reader": {{
            "type": "WeakConll2003DatasetReader",
            "tag_label": "ner",
            "convert_numbers": true,
            "label_indexer": {{
                "label_namespace": "labels",
                "tags": ["LOC", "ORG", "MISC", "PER"]
            }},
            "token_indexers": {{
                "bert": {{
                      "type": "bert-pretrained",
                      "pretrained_model": "./Data/embeddings/BERTEmbeddings/{METHOD}/vocab.txt",
                      "do_lowercase": {lowercase},
                      "use_starting_offsets": true
                }}
             }},
          }},
          "train_data_path": "./Data/CoNLLData/train.txt",
          "validation_data_path": "./Data/CoNLLData/valid.txt",
          "test_data_path": "./Data/CoNLLData/test.txt",
          "evaluate_on_test": true,
          "model": {{
            "type": "MultiClassifier",
            "method": "binary",
            "text_field_embedder": {{
              "type": "basic",
              "token_embedders": {{
                  "bert": {{
                      "type": "bert-pretrained",
                      "pretrained_model": "./Data/embeddings/BERTEmbeddings/{METHOD}/{METHOD}.tar.gz",
                      "top_layer_only": true,
                      "requires_grad": {FINETUNE}
                  }}
              }},
              "embedder_to_indexer_map": {{
                "bert": ["bert", "bert-offsets"],
              }},
              "allow_unmatched_keys": true
            }},
            "encoder_word": {{
              "type": "pass_through",
              "input_dim": {DIM}
            }},
            "attention_word":{{
              "type": "KeyedAttention",
              "key_emb_size": 300,
              "ctxt_emb_size": {DIM},
              "attn_type": "sum",
              "temperature": {TEMP},
            }},
            "threshold": 0.5
          }},
          "iterator": {{
            "type": "bucket",
            "sorting_keys": [["tokens", "bert"]],
            "padding_noise": 0.1,
            "batch_size": 32
          }},
          "segmentation": {{
            "type": "SymbolStopwordFilteredMultiPredictions",
            "tol": 0.01,
            "use_probs": true,
            "visualize": true
          }},
          "trainer": {{
            "optimizer": {OPT},
            "num_epochs": 50,
            "patience": 10,
            "cuda_device": 0,
            "validation_metric": "+accuracy",
            "num_serialized_models_to_keep": 1,
            "learning_rate_scheduler": {{
              "type": "reduce_on_plateau",
              "factor" : 0.5,
              "patience" : 5
              }}
          }}
        }}
        """
        if FINETUNE == "[]":
            ft_layers = 0
        elif FINETUNE == "all":
            ft_layers = "all"
        else:
            ft_layers = len(FINETUNE.split(","))
        optstr = OPT_ix_to_string[ix]
        config_file = os.path.join(CONF_DIR, f"config_{METHOD}_TLO_true_FTL_{ft_layers}_OPT_{optstr}.json")
        with open(config_file, "w") as f:
            f.write(raw)
        run_cmd = f"python -m AttentionSegmentation.main --config_file {config_file}"
        scripts[script_no].append(run_cmd)
        config_count += 1
# Now write the scripts
for ix in range(len(scripts)):
    script_file = os.path.join(SCRIPT_DIR, f"script_{ix}.sh")
    if len(scripts[ix]) > 1:
        with open(script_file, "w") as f:
            f.write("\n".join(scripts[ix]))
