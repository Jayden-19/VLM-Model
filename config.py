
def llm_get_config():
    NUM_LAYERS = 12
    D_MODEL = 768

    return {
        "num_layers":NUM_LAYERS,
        "d_model" : D_MODEL,
        "d_ff" : D_MODEL * 4,
        "num_heads" : 12,
        "dropout" : 0.2,

        "batch_size" : 20,
        "num_epochs" : 20,
        "lr" : 2e-5,
        "warmup_steps":500,
        "seq_len" : 512,
    }

def vit_get_config():
    D_MODEL = 768
    NUM_LAYERS = 12
    IMAGE_SIZE = 1024
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    return {
        "num_layers": NUM_LAYERS,
        "d_model": D_MODEL,
        "num_heads" : 12,
        "dropout": 0.1,
        "img_size" : IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "in_channels": IN_CHANNELS
    }

def get_training_config():
    return {
        "datasource" : "",

        "model_folder" : "weights",
        "model_basename" : "llmmodel",
        "tokenizer_file" : "tokenizer{0}.json",
        "experiment_name" : "runs/llmmodel"
    }