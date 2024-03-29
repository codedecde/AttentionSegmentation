# This is a script generator file.
# Very hacky, will be removed from the final branch
import os
import re

BASE_DIR = os.path.join("Configs/CoNLL-Configs/BERT-mlingual-case")
CONF_DIR = os.path.join(BASE_DIR, "Configs")
SCRIPT_DIR = os.path.join(BASE_DIR, "Scripts")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

NUM_SCRIPTS = 1
METHODS = [
    "bert-base-multilingual-cased",
]
# drop_list = [0.2, 0.4, 0.5, 0.6, 0.8]

INP_DIMS = {
    "bert-base-cased": 768,
    "bert-base-multilingual-cased": 768,
    "bert-base-multilingual-uncased": 768,
    "bert-large-cased": 1024
}

TOP_LAYER_ONLY_VALUES = ["false"]

TOTAL = len(METHODS) * len(TOP_LAYER_ONLY_VALUES)
num_per_script = -(-TOTAL // NUM_SCRIPTS)
SCRIPT_HEADER = f"""

"""
SCRATCH = "~/mountdir/AttentionSegmentation/output/"
scripts = [[SCRIPT_HEADER] for _ in range(NUM_SCRIPTS)]
config_count = 0
DROPOUT = 0.
TEMP = 1.

for METHOD in METHODS:
    lowercase = "true" if "uncased" in METHOD else "false"
    for TLO in TOP_LAYER_ONLY_VALUES:
        DIM = INP_DIMS[METHOD]
        raw = f"""
        {{
          "base_output_dir": "{SCRATCH}/Experiments/CoNLL/BERT-mlingual-case/{METHOD}",
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
                      "use_starting_offsets": "true"
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
                      "top_layer_only": {TLO}
                  }}
              }},
              "embedder_to_indexer_map": {{
                "bert": ["bert", "bert-offsets"],
              }},
              "allow_unmatched_keys": true
            }},
            "encoder_word": {{
              "type": "gru",
              "input_size": {DIM},
              "hidden_size": 150,
              "num_layers": 1,
              "dropout": 0.5,
              "bidirectional": true
            }},
            "attention_word":{{
              "type": "KeyedAttention",
              "key_emb_size": 300,
              "ctxt_emb_size": 300,
              "attn_type": "sum",
              "dropout": {DROPOUT},
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
            "type": "BasicMultiPredictions",
            "tol": 0.01,
            "visualize": true
          }},
          "trainer": {{
            "optimizer": "adam",
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
        # Write the config
        temp_txt = re.sub("\.", "_", str(TEMP))
        config_file = os.path.join(CONF_DIR, f"config_{METHOD}_TLO_{TLO}.json")
        with open(config_file, "w") as f:
            f.write(raw)
        script_no = config_count // num_per_script
        run_cmd = f"${{PYTHON}} -m AttentionSegmentation.main --config_file {config_file}"
        scripts[script_no].append(run_cmd)
        config_count += 1
# Now write the scripts
for ix in range(len(scripts)):
    script_file = os.path.join(SCRIPT_DIR, f"script_{ix}.sh")
    if len(scripts[ix]) > 1:
        with open(script_file, "w") as f:
            f.write("\n".join(scripts[ix]))
