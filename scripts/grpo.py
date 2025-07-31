from vlmgrpo import VLMGRPOTrainer
from trl import GRPOConfig
from unsloth import FastVisionModel, is_bf16_supported
from datasets import load_dataset, Dataset
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
     
def format_fn(sample, reasoning=False, triplet=False):
    if reasoning:
        prompt = f"In the image you see a block tower and a single red block next to it. You can move the red block to the left or right and up, by responding with two values (x = left or right, y = up). Provide your reasoning between {reasoning_start} and {reasoning_end}. You can think about the problem for as long as you'd like. While thinking, you should robustly verify your solution. Once you are done thinking, return the final answer like this, with two floats between {solution_start} 1.2, 0.6 {solution_end}."
    else:
        prompt = f"In the image you see a block tower and a single red block next to it. Your task is to build a bigger tower by placing the red block on top of the existing tower. To accomplish this, you can move the red block to the left or right and up, by responding with two values (x = left or right, y = up). Return your final answer like this, with two integers between {solution_start} {solution_end}."
        
    if triplet:
        format = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # First image
                        {"type": "text", "text": prompt + f" You have three attempts left. You answered: {[sample['red_x_move_1'], sample['red_y_move_1']]}."},  # Text after first image
                        {"type": "image"},  # Second image
                        {"type": "text", "text": f"You have two attempts left. You answered: {[sample['red_x_move_2'], sample['red_y_move_2']]}."},  # Text after second image
                        {"type": "image"},  # Third image
                        {"type": "text", "text": f"You have one attempt left. Return your final answer like this, with two integers between {solution_start} {solution_end}."}  # Text after third image
                    ]
                }
            ],
            "image": [sample['first'], sample['second'], sample['third']], # sample['images'] should be a list of 3 images
            "answer": [
                [sample['final_x_move'], sample['final_y_move']],  #  Answer to text after image 3
            ]
        }
    else:  
        format = {"prompt": [
                    {
                    "role": "user",
                    "content": [
                        {"type": "image"}, # because we have only 1 image per sample
                        {"type": "text", "text": prompt}]
                    }],
                    "image": sample['image'],
                    "answer": [sample['scale'], sample['x_offset'], sample['y_offset']]
                }
    return format

class FormattedDataset():
    def __init__(self, dataset, format_fn, reasoning, triplet):
        self.dataset = dataset
        self.format_fn = format_fn
        self.reasoning = reasoning
        self.triplet = triplet

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.format_fn(item, self.reasoning, self.triplet)

    def __len__(self):
        return len(self.dataset)
    

### REWARD FUNCTIONS
def formatting_reward_func(completions,**kwargs):
    thinking_pattern = f'{reasoning_start}(.*?){reasoning_end}'
    answer_pattern = f'{solution_start}(.*?){solution_end}'

    scores=[]
    for completion in completions:
      score=0
      thinking_matches = re.findall(thinking_pattern, completion[0]['content'], re.DOTALL)
      answer_matches = re.findall(answer_pattern, completion[0]['content'], re.DOTALL)
      if len(thinking_matches) == 1 :
        score +=1.0
      if len(answer_matches) == 1 :
        score +=1.0
      scores.append(score)
    return scores

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    top_reward = 10
    exponential = False
    c = 1
    answer_pattern = f'{solution_start}(.*?){solution_end}'
    responses = [re.findall(answer_pattern, completion[0]['content'], re.DOTALL) for completion in completions]
    q = prompts[0][-1]['content']

    print('-'*20, f"Question:\n{q}", flush=True)
    print('-'*20, f"First completion:\n{completions[0]}", flush=True)

    # Loop through responses
    rewards = []
    for r, a in zip(responses, answer):

        # Parse answer
        reward = 0
        scale = a[0]
        x_offset = a[1]
        y_offset = a[2]

        # Check if the model returned clean parse-able integers
        if r and isinstance(r[0], str):
            try:
                # Remove square brackets and strip and extract integer
                cleaned = r[0].strip().strip('[]')
                x_move, y_move = [int(item.strip()) for item in cleaned.split(',')]
                
                # Transform answer back from integer to float
                x_move /= 100
                y_move /= 100

            # If the answer is note cleany parse-able, also return -5     
            except ValueError:
                reward = -5
        else:
            reward = -5

        # Only continue if outputs were parsed correctly
        if reward == 0 and y_offset > 0:

            # Check X position
            # Initial X position + movement on X axis -> center of block above last block?
            final_x = x_offset + x_move
            x_good = True if (final_x > (-scale/2) and final_x < (scale/2)) else False
            
            # Check Y position
            # Last block height - red block displacement -> red block lifted higher than tower?
            final_y = 0 + y_move
            y_good = True if final_y > y_offset else False

            # Define optimal position
            optimal_x = 0
            optimal_y = y_offset

            # Compute distance to optimal position
            # square root of summed square of x distance + square of y distance 
            distance = ((final_x - optimal_x) ** 2 + (final_y - optimal_y) ** 2) ** 0.5

            # If both positions are good, give reward
            # x good and y good -> Bigger tower: 10 - distance to optimal position
            # x not good -> Red block outside tower: 5 - distance to optimal position
            # x good and y not good -> Red block inside tower: -5
            # y negative -> Red block below ground: -5
            
            # If block is on top of tower, give reward based on distance to optimal position
            if x_good and y_good:
                if exponential:
                    reward = top_reward + (-(1/(c * -abs(distance))))
                else:
                    reward = top_reward - distance
            # If position is not on top of tower but above ground, it is legal
            elif not x_good:
                reward = 5 - distance
            # If position is on top of tower but below top block, it is inside tower, and therefore illegal
            elif x_good and not y_good:
                reward = -5 
            # If position is below ground, give negative reward
            elif final_y < 0:
                reward = -5

        # Add reward to rewards list
        print('-'*10, f"Parsed response:\n{r}, reward: {reward}, top reward: {top_reward}", flush=True)
        rewards.append(reward)
    return rewards


@call_parse
def main(model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct", reasoning: bool = False, hot_start: bool = False, top_reward: int = 10, exponential: bool = False, triplet: bool = False):
    
    ### MODEL CONFIG
    model_name_no_unsloth = model_name.split("/")[-1]
    print(f"Running {model_name_no_unsloth} with reasoning = {reasoning} and hot start = {hot_start} and top reward = {top_reward}, with triplet = {triplet}")
    model, tokenizer = FastVisionModel.from_pretrained(
        f"./outputs/sft/{model_name_no_unsloth}/checkpoint-5000/" if hot_start else model_name,
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    if not hot_start:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers
            r = 16,           # The larger, the higher the accuracy, but might overfit
            lora_alpha = 16,  # Recommended alpha == r at least
            lora_dropout = 0.1,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

    # Configure output directory
    output_base = "outputs/grpo/hot" if hot_start else "outputs/grpo/cold"
    output_dir = f"{output_base}/{model_name_no_unsloth}_reasoning={reasoning}_topreward={top_reward}_exponential={exponential}_triplet={triplet}"
    if triplet:
        ds = load_dataset("lsbuschoff/interact_triple_static",split="train")
    else:
        ds = load_dataset("lsbuschoff/interact_single_r1",split="train")
    train_dataset = FormattedDataset(dataset=ds, format_fn=format_fn, reasoning=reasoning, triplet=triplet)

    ### GRPO CONFIG
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bf16_supported(),
        fp16 = not is_bf16_supported(),
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 16, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 200,
        num_train_epochs = 10, # Set to 1 for a full training run
        # max_steps = 1000,
        save_steps = 1000,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_dir,
    )

    trainer = VLMGRPOTrainer(
        model = model,
        processing_class=tokenizer, # MUST put unsloth processor here !
        reward_processing_classes = tokenizer, #Here also
        reward_funcs = [
            # formatting_reward_func,  # Removing this for now as it's not needed
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = train_dataset,
        grad_verbose=True
    )

    trainer.train()