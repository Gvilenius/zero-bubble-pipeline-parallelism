import os
import json

rank = 0
# batch_sizes = [72, 144, 216, 288]
# acc_steps=[6, 12, 18, 24]



world_size = 8

def set_env(k, v):
	os.environ[k] = str(v)

def init(config):
    set_env("PIPELINE_SIZE", world_size)
    set_env("LAYERS", config["n_layers"])
    set_env("MICRO_BATCH_SIZE", config["batch_size"] // config["gradient_accumulation_steps"])
    set_env("GLOBAL_BATCH_SIZE", world_size * config["batch_size"])
    set_env("HIDDEN_SIZE", config["dim"])
    set_env("ATTENTION_HEADS", config["n_heads"])
    set_env("SEQ_LEN", config["max_seq_len"])
    set_env("ZERO_BUBBLE_MEM_LIMIT", 2 * world_size)
    set_env("EXIT_INTERVAL", config["iters_num"])
    set_env("INTERLEAVED_1F1B", 1)
    # set_env("ENABLE_ZERO_BUBBLE", 1)


with open("../weipipe/config.json", "r") as f:
    config = json.load(f)
		
init(config)

os.system(
    f'bash examples/pretrain_zero_bubble.sh'
)