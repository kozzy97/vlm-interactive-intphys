from unsloth import FastVisionModel, is_bf16_supported
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from fastcore.script import call_parse
import re
import os


### WANDB CONFIG
os.environ["WANDB_ENTITY"] = "lucaschulzebuschoff"
os.environ["WANDB_PROJECT"] = "interact_grpo"


### DATASET CONFIG
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start  = "<answer>"
solution_end    = "</answer>"

def format_fn(sample, reasoning=False):
    if reasoning:
        prompt = f"In the image you see a block tower and a single red block next to it. You can move the red block to the left or right and up, by responding with two values (x = left or right, y = up). Provide your reasoning between {reasoning_start} and {reasoning_end}. You can think about the problem for as long as you'd like. While thinking, you should robustly verify your solution. Once you are done thinking, return the final answer like this, with two floats between {solution_start} 1.2, 0.6 {solution_end}."
    else:
        prompt = f"In the image you see a block tower and a single red block next to it. Your task is to build a bigger tower by placing the red block on top of the existing tower. To accomplish this, you can move the red block to the left or right and up, by responding with two values (x = left or right, y = up). Return your final answer like this, with two integers between {solution_start} {solution_end}."

    conversation = [
        { "role": "user",
          "content" : [
            {"type": "text", "text": prompt},
            {"type" : "image", "image" : sample["image"]}]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : f"{solution_start} {int(sample["x_offset"] * 100)}, {int(sample["y_offset"] * 100)} {solution_end}"}]
        },
    ]

    return { "messages" : conversation }

class FormattedDataset():
    def __init__(self, dataset, format_fn, reasoning):
        self.dataset = dataset
        self.format_fn = format_fn
        self.reasoning = reasoning

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.format_fn(item, self.reasoning)

    def __len__(self):
        return len(self.dataset)


@call_parse
def main(model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct", reasoning: bool = False):

    ### MODEL CONFIG
    model_name_no_unsloth = model_name.split("/")[-1]
    print(f"Running SFT on {model_name_no_unsloth} with reasoning = {reasoning}.")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    print('-'*20, "Model loaded", flush=True)

    ### DATASET AND OUTPUT CONFIG
    output_base = "outputs/sft/" 
    output_dir = f"{output_base}/{model_name_no_unsloth}_reasoning={reasoning}"
    ds = load_dataset("lsbuschoff/interact_single_r1",split="train")
    train_dataset = FormattedDataset(dataset=ds, format_fn=format_fn, reasoning=reasoning)
    print('-'*20, "Dataset converted", flush=True)

    ### TRAIN
    FastVisionModel.for_training(model) # Enable for training!

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = train_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 64,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # max_steps = 5000,
            num_train_epochs = 10, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "wandb",    

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )

    print('-'*20, "Beginning training", flush=True)
    trainer.train()