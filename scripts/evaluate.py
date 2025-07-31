import csv
import re
import os

from unsloth import FastVisionModel
from datasets import load_dataset, Dataset
from fastcore.script import call_parse

solution_start  = "<answer>"
solution_end    = "</answer>"

building_instruction = f"In the image you see a block tower and a single red block next to it. Your task is to build a bigger tower by placing the red block on top of the existing tower. To accomplish this, you can move the red block to the left or right and up, by responding with two values (x = left or right, y = up). Return your final answer like this, with two integers between {solution_start} {solution_end}."

PREPROMPT_INTPHYS = """You are now viewing a pyramid of blocks. Will any of the blocks fall?
Answer Yes if you think that at least one block will fall.
Answer No if you think that no blocks will fall."""

def _convert_stable_to_yes_no(sample, reverse: bool = False) -> str:
    if reverse:
        return "No" if sample == "unstable" else "Yes"
    else:
        return "Yes" if sample == "unstable" else "No"

def produce_multiple_in_context_examples(initial_prompt, in_context_examples) -> tuple[dict, list]:
    images = []
    if in_context_examples is not None:
        user_content = []
        for key, value in in_context_examples.items():
            prompt = initial_prompt if key == 0 else _convert_stable_to_yes_no(value[1])
            images.append(value[0])
            user_content.extend(
                    [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                )
        user_content.append({"type": "image"})
    else: # no in-context examples, just use the test image
        user_content = [
            {"type": "text", "text": initial_prompt},
            {"type" : "image"}]
    messages = [
        { "role": "user",
          "content" : user_content,
        },
    ]
    return messages, images
    
    
def evaluate_model_output(model_output: str, x_offset: float, y_offset: float, scale: float) -> tuple[int, int, str, float, float, float]:
    answer_pattern = f'{solution_start}(.*?){solution_end}'
    response = re.findall(answer_pattern, model_output, re.DOTALL)
    if response and isinstance(response[0], str):
        try:
            # Remove square brackets and strip and extract integer
            cleaned = response[0].strip().strip('[]')
            x_move, y_move = [int(item.strip()) for item in cleaned.split(',')]
            # Transform answer back from integer to float
            x_move /= 100
            y_move /= 100
        except ValueError:
            return 0, -1, "Not-parseable response", -999, -999, -999
    else:
        return 0, -1, "Not-parseable response", -999, -999, -999
    # Only continue if outputs were parsed correctly
    if y_offset > 0: # just to be sure, y_offset should always be > 0
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
        # If block is on top of tower, give reward based on distance to optimal position
        if x_good and y_good:
            return 1, 1, "Legal, Taller", distance, x_move, y_move
        # If position is not on top of tower but above ground, it is legal
        else:
            return 0, 0, "Legal, Not Taller", distance, x_move, y_move
    else:
        raise ValueError(f"y_offset must be greater than 0, got {y_offset}")

def eval_single_static_eval(model_name: str, # Path to the model to evaluate, can be a local path or a HuggingFace model name
         save_directory: str = "results", # Directory to save the results
         ) -> None:
    
    ### MODEL CONFIG
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    FastVisionModel.for_inference(model)

    dataset_name = "lsbuschoff/interact_single_static_eval" # Name of the dataset to evaluate on
    # Create csv path making some assumptions about the model name
    if "VLM_GRPO/" in model_name:
        csv_path = os.path.join(save_directory, "interact_single_static_eval", f"results_{model_name.split('VLM_GRPO/')[-1].replace("/", "_")}.csv")
    else:
        csv_path = os.path.join(save_directory, "interact_single_static_eval", f"results_{model_name.replace("/", "_")}.csv")
    
    # Create save directory if it does not exist
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = sum(1 for line in f) - 1 # Count the number of lines to resume from the last line, assuming the first line is the header and the order of the dataset does not change
    else:
        with open(csv_path, "w", newline='') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(["dataset_item", "x_offset", "y_offset", "scale", "correct_binary", "correct_ternary", "text_reason", "distance", "llm_x_move", "llm_y_move", "llm_text_response"])
        lines = 0

    print('-'*20, "Loading dataset", '-'*20, flush=True)
    dataset = load_dataset(dataset_name, split="train")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": building_instruction}
            ]}
        ]
    
    print('\n'*10, '-'*20, "Running evaluation" ,'-'*17, flush=True)
    for i in range(lines, len(dataset)):
        if i % 100 == 0:
            print(f"Evaluating item {i+1}/{len(dataset)}", flush=True)
        
        image, x_offset, y_offset, scale = dataset[i]['image'], dataset[i]['x_offset'], dataset[i]['y_offset'], dataset[i]['scale']
        input_text = tokenizer.apply_chat_template([messages], add_generation_prompt = True)

        inputs = tokenizer(
                image,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
                ).to("cuda")
        
        model_outputs = model.generate(**inputs, max_new_tokens=100, temperature=1.0, min_p=0.1, return_dict_in_generate=False, output_hidden_states=False, output_logits=False)
        out_string = tokenizer.batch_decode(model_outputs[:, inputs.input_ids.shape[1]:])[0]
        correct_binary, correct_ternary, text_reason, distance, llm_x_move, llm_y_move = evaluate_model_output(out_string, x_offset, y_offset, scale)
        
        with open(csv_path, "a", newline='') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow([i+1, x_offset, y_offset, scale, correct_binary, correct_ternary, text_reason, distance, llm_x_move, llm_y_move, out_string.strip().replace("\n", "<lb>")]) # replace newlines with <lb> for better readability in csv
    print('-'*20, "Complete", '-'*20, flush=True)


def eval_stability_judgments(model_name: str, # Path to the model to evaluate, can be a local path or a HuggingFace model name
                             dataset_name: str = "lsbuschoff/redremove-eval", # Name of the dataset to evaluate on
                             save_directory: str = "results", # Directory to save the results
                             num_in_context_examples: int = 0, # Number of in-context examples to use
                            ) -> None:

    ### MODEL CONFIG
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    FastVisionModel.for_inference(model)

    dataset_type = "redremove" if "redremove" in dataset_name else "towerblocks"
    # Create csv path making some assumptions about the model name
    if "VLM_GRPO/" in model_name:
        csv_path = os.path.join(save_directory, dataset_type, f"results_{model_name.split('VLM_GRPO/')[-1].replace("/", "_")}_nicl_{num_in_context_examples}.csv")
    else:
        csv_path = os.path.join(save_directory, dataset_type, f"results_{model_name.replace("/", "_")}_nicl_{num_in_context_examples}.csv")
    
    # Create save directory if it does not exist
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = sum(1 for line in f) - 1 # Count the number of lines to resume from the last line, assuming the first line is the header and the order of the dataset does not change
    else:
        with open(csv_path, "w", newline='') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(["dataset_item", "label", "correct", "llm_text_response"])
        lines = 0

    # Create dataset
    print('-'*20, "Loading dataset", '-'*20, flush=True)
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(lambda example, idx: {"index": idx}, with_indices=True) # add index to the dataset for easier tracking (will be in order of HF dataset)

    if "redremove" in dataset_name:
        if num_in_context_examples > 0: # If we want to use in-context examples
            train_split_prop = num_in_context_examples/len(ds)
            ds_split = ds.train_test_split(train_size=train_split_prop, seed=1997) # gets shuffled - good thing we added indices
            in_context_examples = {i : [item['second'], item['label']] for i, item in enumerate(ds_split['train'])}
            dataset = ds_split['test']
            messages, icl_images = produce_multiple_in_context_examples(PREPROMPT_INTPHYS, in_context_examples)
        else: # If we do not want to use in-context examples
            dataset = ds
            messages, _ = produce_multiple_in_context_examples(PREPROMPT_INTPHYS, None)
        print('\n'*10, '-'*20, "Dataset converted" ,'-'*18, flush=True)
    elif "finetune_stability" in dataset_name:
        if num_in_context_examples > 0: # If we want to use in-context examples
            train_split_prop = num_in_context_examples/len(ds)
            ds_split = ds.train_test_split(train_size=train_split_prop, seed=1997) # gets shuffled - good thing we added indices
            in_context_examples = {i : [item['image'], item['answer']] for i, item in enumerate(ds_split['train'])}
            dataset = ds_split['test']
            messages, icl_images = produce_multiple_in_context_examples(PREPROMPT_INTPHYS, in_context_examples)
        else: # If we do not want to use in-context examples
            dataset = ds
            messages, _ = produce_multiple_in_context_examples(PREPROMPT_INTPHYS, None)
        print('\n'*10, '-'*20, "Dataset converted" ,'-'*18, flush=True)
    
    print('\n'*10, '-'*20, "Running evaluation" ,'-'*17, flush=True)

    for i in range(lines, len(dataset)):
        if i % 100 == 0:
            print(f"Evaluating item {i+1}/{len(dataset)}", flush=True)
        
        if "redremove" in dataset_name:
            d_image, d_label, d_index = dataset[i]["second"], dataset[i]["label"], dataset[i]["index"]
        elif "finetune_stability" in dataset_name:
            d_image, d_label, d_index = dataset[i]["image"], dataset[i]["answer"], dataset[i]["index"]
        label_yes_no = _convert_stable_to_yes_no(d_label)
        opposite_label = _convert_stable_to_yes_no(d_label, reverse=True)
        
        input_text = tokenizer.apply_chat_template([messages], add_generation_prompt = True)

        image_list = d_image if num_in_context_examples == 0 else icl_images + [d_image] # add in-context examples if we have any

        inputs = tokenizer(
                image_list,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
                ).to("cuda")
        
        model_outputs = model.generate(**inputs, max_new_tokens=10, temperature=1.0, min_p=0.1, return_dict_in_generate=False, output_hidden_states=False, output_logits=False)
        out_string = tokenizer.batch_decode(model_outputs[:, inputs.input_ids.shape[1]:])[0]

        correct = 1 if f"{label_yes_no}" in out_string and f"{opposite_label}" not in out_string else 0
        
        with open(csv_path, "a", newline='') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow([d_index, d_label, correct, out_string.strip().replace("\n", "<lb>")]) # replace newlines with <lb> for better readability in csv
    
    print('-'*20, "Complete", '-'*20, flush=True)


@call_parse
def main(model_name: str, # Path to the model to evaluate, can be a local path or a HuggingFace model name
        dataset_name: str = "lsbuschoff/redremove-eval", # Name of the dataset to evaluate on
        save_directory: str = "results", # Directory to save the results
        num_in_context_examples: int = 0 # Number of in-context examples to use for stability judgments
         ):
    assert dataset_name in ["lsbuschoff/interact_single_static_eval", "lsbuschoff/redremove-eval", "lsbuschoff/finetune_stability"], "Dataset must be one of ['lsbuschoff/interact_single_static_eval', 'lsbuschoff/redremove-eval', 'lsbuschoff/finetune_stability']"
    if dataset_name == "lsbuschoff/interact_single_static_eval":
        if num_in_context_examples > 0:
            print("Warning: num_in_context_examples is not used for interact_single_static_eval, it will be ignored.")
        eval_single_static_eval(model_name=model_name, save_directory=save_directory)
    elif dataset_name == "lsbuschoff/redremove-eval" or dataset_name == "lsbuschoff/finetune_stability":
        eval_stability_judgments(model_name=model_name, dataset_name=dataset_name, save_directory=save_directory, num_in_context_examples=num_in_context_examples)