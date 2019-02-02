# This is a script generator file.
# Very hacky, will be removed from the final branch
import os
import re

BASE_DIR = os.path.join("Configs/CoNLL-Configs/Dropout")
CONF_DIR = os.path.join(BASE_DIR, "Configs")
SCRIPT_DIR = os.path.join(BASE_DIR, "Scripts")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

NUM_SCRIPTS = 4
tag_list = ["LOC", "PER", "ORG", "MISC"]
drop_list = [0.2, 0.4, 0.5, 0.6, 0.8]
TOTAL = len(tag_list) * len(drop_list)

num_per_script = -(-TOTAL // NUM_SCRIPTS)

SCRIPT_HEADER = f"""
PYTHON=/zfsauton3/home/bpatra/miniconda3/bin/python3.6

"""
scripts = [[SCRIPT_HEADER] for _ in range(NUM_SCRIPTS)]
config_count = 0

for TAG in tag_list:
    for DROPOUT in drop_list:
        raw = f"""
        {{
          "base_output_dir": "./Experiments/CoNLL/DROPOUT/{TAG}",
          "dataset_reader": {{
            "type": "WeakConll2003DatasetReader",
            "tag_label": "ner",
            "convert_numbers": true,
            "label_indexer": {{
                "label_namespace": "labels",
                "tags": ["{TAG}"]
            }},
            "token_indexers": {{
                "tokens": {{
                    "type": "single_id",
                    "lowercase_tokens": true,
                    "namespace": "token_ids"
                }},
                "chars": {{
                    "type": "characters",
                    "namespace": "token_chars"
                }},
                "elmo": {{
                     "type": "elmo_characters"
                }}
             }}
          }},
          "train_data_path": "./Data/CoNLLData/train.txt",
          "validation_data_path": "./Data/CoNLLData/valid.txt",
          "test_data_path": "./Data/CoNLLData/test.txt",
          "evaluate_on_test": true,
          "model": {{
            "type": "Classifier",
            "text_field_embedder": {{
              "tokens": {{
                "type": "embedding",
                "vocab_namespace": "token_ids",
                "embedding_dim": 300,
                "pretrained_file": "./Data/embeddings/GloveEmbeddings/glove.6B.300d.txt.gz",
                "trainable": true
              }},
              "chars": {{
                "type": "character_encoding",
                "embedding": {{
                  "vocab_namespace": "token_chars",
                  "embedding_dim": 50 
                }},
                "encoder": {{
                  "type": "gru",
                  "input_size": 50,
                  "hidden_size": 80,
                  "num_layers": 2,
                  "dropout": 0.25,
                  "bidirectional": true
                }}
              }},
              "elmo": {{
                  "type": "elmo_token_embedder",
                  "options_file": "./Data/embeddings/ELMOEmbeddings/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                  "weight_file": "./Data/embeddings/ELMOEmbeddings/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                  "do_layer_norm": false,
                  "dropout": 0.0
              }}
            }},
            "encoder_word": {{
              "type": "gru",
              "input_size": 1484,
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
              "dropout": {DROPOUT}
            }},
            "threshold": 0.5
          }},
          "iterator": {{
            "type": "bucket",
            "sorting_keys": [["tokens", "num_tokens"]],
            "padding_noise": 0.1,
            "batch_size": 32
          }},
          "visualize": true,
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
        dropout_txt = re.sub("\.", "_", str(DROPOUT))
        config_file = os.path.join(CONF_DIR, f"config_tag_{TAG}_dropout_{dropout_txt}.json")
        with open(config_file, "w") as f:
            f.write(raw)
        script_no = config_count // num_per_script
        run_cmd = f"${{PYTHON}} -m AttentionSegmentation.main --config_file {config_file}"
        scripts[script_no].append(run_cmd)
        config_count += 1
# Now write the scripts
for ix in range(len(scripts)):
    script_file = os.path.join(SCRIPT_DIR, f"script_{ix}.sh")
    with open(script_file, "w") as f:
        f.write("\n".join(scripts[ix]))
