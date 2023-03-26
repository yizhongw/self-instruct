import os
import json
import argparse
import glob
import re
import random
import tqdm
import pandas as pd


random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_files",
        nargs="+",
        default=["data/batch_221203/machine_generated_instances.jsonl"],
        type=str,
        help="The input files that contain the machine generated instances."
    )
    parser.add_argument(
        "--classification_type_files",
        nargs="+",
        default=["data/batch_221203/is_clf_or_not_davinci_template_1.jsonl"],
    )
    parser.add_argument(
        "--output_dir",
        default="data/gpt3_generations/batch_221203/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for GPT3 finetuning."
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="The number of instructions to load."
    )
    parser.add_argument(
        "--include_seed_tasks",
        action="store_true",
        help="Whether to include the seed human-written instances in the finetuning data."
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the seed data.",
    )
    return parser.parse_args()


def encode_instance(instruction, input, output, random_template=True):
    encoding_templates_w_input = [
        ("{instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nInput: {input}\n\nOutput:", " {output}<|endoftext|>"),
        ("Task: {instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
        ("{instruction}\n{input}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
    ]
    encoding_templates_wo_input = [
        ("{instruction} Output:", " {output}<|endoftext|>"),
        ("{instruction}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n", "{output}<|endoftext|>"),
        ("{instruction}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n", "{output}<|endoftext|>"),
    ]
    if random_template:
        if input.strip() != "":
            prompt_template, completion_template = random.choice(encoding_templates_w_input)
            prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
            completion = completion_template.format(output=output.strip())
        else:
            prompt_template, completion_template = random.choice(encoding_templates_wo_input)
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
        completion = output.strip() + "<|endoftext|>"

    data = {
        "prompt": prompt,
        "completion": completion,
        "instruction": instruction.strip(),
        "input": input.strip(),
        "output": output.strip(),
    }
    return data


def parse_input_output(response_text):
    if re.findall(r"Output\s*\d*\s*:", response_text):
        inst_input = re.split(r"Output\s*\d*\s*:", response_text)[0].strip()
        inst_output = re.split(r"Output\s*\d*\s*:", response_text)[1].strip()
    else:
        inst_input = ""
        inst_output = response_text.strip()
    # to avoid the case multiple input/output pairs are generated
    if re.findall(r"Input\s*\d*\s*:", inst_output):
        inst_output = re.split(r"Input\s*\d*\s*:", inst_output)[0].strip()
    # remove the prefix "Input:" from the string
    inst_input = re.sub(r"^Input\s*\d*\s*:", "", inst_input).strip()
    return inst_input, inst_output


def filter_duplicate_instances(instances):
    # if the instances have same non-empty input, but different output, we will not use such instances
    same_input_diff_output = False
    for i in range(1, len(instances)):
        for j in range(0, i):
            if instances[i][1] == "":
                continue
            if instances[i][1] == instances[j][1] and instances[i][2] != instances[j][2]:
                same_input_diff_output = True
                break
    if same_input_diff_output:
        return []

    # remove duplicate instances
    instances = list(set(instances))
    return instances

def filter_invalid_instances(instances):
    filtered_instances = []
    for instance in instances:
        # if input and output are the same, we will not use such instances
        if instance[1] == instance[2]:
            continue
        # if output is empty, we will not use such instances
        if instance[2] == "":
            continue
        # if input or output ends with a colon, these are usually imcomplete generation. We will not use such instances
        if instance[1].strip().endswith(":") or instance[2].strip().endswith(":"):
            continue
        filtered_instances.append(instance)
    return filtered_instances

def parse_instances_for_generation_task(raw_text, instruction, response_metadata):
    instances = []
    raw_text = raw_text.strip()
    if re.findall("Example\s?\d*\.?", raw_text):
        instance_texts = re.split(r"Example\s?\d*\.?", raw_text)
        instance_texts = [it.strip() for it in instance_texts if it.strip() != ""]
        for instance_text in instance_texts:
            inst_input, inst_output = parse_input_output(instance_text)
            instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    elif re.findall(r"Output\s*\d*\s*:", raw_text):
        # we assume only one input/output pair in this case
        inst_input, inst_output = parse_input_output(raw_text)
        instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    else:
        return []
    # if the generation stops because of length, we remove the last instance
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]
    
    instances = filter_invalid_instances(instances)
    instances = filter_duplicate_instances(instances)
    return instances

def parse_instances_for_classification_task(raw_text, instruction, response_metadata):
    instances = []
    if not "Class label:" in raw_text:
        return []
    instance_texts = raw_text.split("Class label:")[1:]
    for instance_text in instance_texts:
        instance_text = instance_text.strip()
        fields = instance_text.split("\n", 1)
        if len(fields) == 2:
            # the first field split by \n is the class label
            class_label = fields[0].strip()
            # the rest is the input
            input_text = fields[1].strip()
        elif len(fields) == 1:
            # the first field split by \n is the input
            class_label = fields[0].strip()
            input_text = ""
        else:
            raise ValueError("Invalid instance text: {}".format(instance_text))
        instances.append((instruction.strip(), input_text.strip(), class_label.strip()))

    # if the generation stops because of length, we remove the last instance
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]
    instances = filter_invalid_instances(instances)
    instances = filter_duplicate_instances(instances)
    return instances


if __name__ == "__main__":
    args = parse_args()

    training_instances = []
    
    generated_tasks = []
    for instance_file in args.instance_files:
        with open(instance_file) as fin:
            for line in fin:
                generated_tasks.append(json.loads(line))
    print(f"Loaded {len(generated_tasks)} raw generated tasks")

    task_clf_types = {}
    for file in args.classification_type_files:
        with open(file) as fin:
            for line in fin:
                data = json.loads(line)
                task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    for task in tqdm.tqdm(generated_tasks):
        # get instruction
        instruction = task["instruction"]
        task["is_classification"] = task_clf_types[instruction]

        # get the instances
        if task["is_classification"]:
            task_instances = parse_instances_for_classification_task(task["raw_instances"], instruction, task["instance_metadata"])
        else:
            task_instances = parse_instances_for_generation_task(task["raw_instances"], instruction, task["instance_metadata"])

        # we only allow max 5 instances per task
        task_instances = random.sample(task_instances, min(len(task_instances), 5))
        
        if not task_instances:
            continue

        training_instances += task_instances


    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "all_generated_instances.jsonl"), "w") as fout:
        for instance in training_instances:
            fout.write(json.dumps({
                "instruction": instance[0],
                "input": instance[1],
                "output": instance[2],
            }) + "\n")
    print(f"Saved {len(training_instances)} instances")
    unique_instructions = set([it[0] for it in training_instances])
    print(f"Unique instructions: {len(unique_instructions)}")
    clf_instructions = [instruction for instruction in unique_instructions if task_clf_types[instruction]]
    print(f"Classification instructions: {len(clf_instructions)}")
    non_clf_instructions = [instruction for instruction in unique_instructions if not task_clf_types[instruction]]
    print(f"Non-classification instructions: {len(non_clf_instructions)}")

    if args.num_instructions is not None:
        print(f"Sampling {args.num_instructions} instructions")
        sampled_instructions = random.sample(unique_instructions, args.num_instructions)
        training_instances = [it for it in training_instances if it[0] in sampled_instructions]
        print(f"Only using {len(training_instances)} instances for these sampled instructions.")
        with open(os.path.join(args.output_dir, f"sampled_generated_instances_{args.num_instructions}.jsonl"), "w") as fout:
            for instance in training_instances:
                fout.write(json.dumps({
                    "instruction": instance[0],
                    "input": instance[1],
                    "output": instance[2],
                }) + "\n")

    if args.include_seed_tasks:
        seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
        for task in seed_tasks:
            for instance in task["instances"]:
                training_instances.append((task["instruction"], instance["input"], instance["output"]))
        print(f"Included {len(seed_tasks)} seed tasks")

    # get the prompt and completion for training gpt3
    gpt3_instances = []
    for instance in training_instances:
        # get input and do preprocessing
        inst_input = instance[1]
        # for some tasks, we check whether the input contains colon, and if so, we remove the part before the colon
        if random.random() < 0.5:
            colon_words = re.findall(r"(\w+):", inst_input)
            # if only one colon is found, we assume the instance only have one input and we remove the field name before the colon
            if len(set(colon_words)) == 1:
                inst_input = inst_input.split(":", 1)[1].strip()
            else:
                inst_input = inst_input.strip()
            # we also replace two consecutive new lines with one new line half of the time
            inst_input = inst_input.replace("\n\n", "\n")
        
        gpt3_instances.append(encode_instance(instance[0], inst_input, instance[2]))

    # remove duplicates
    filtered_instances = []
    prompt_completion_set = set()
    for instance in gpt3_instances:
        instance_pair = (instance["prompt"], instance["completion"])
        if instance_pair not in prompt_completion_set:
            prompt_completion_set.add((instance["prompt"], instance["completion"]))
            filtered_instances.append(instance)
    gpt3_instances = filtered_instances

    # shuffle
    random.shuffle(gpt3_instances)
    with open(os.path.join(args.output_dir, f"gpt3_finetuning_data_{len(gpt3_instances)}.jsonl"), "w") as fout:
        for instance in gpt3_instances:
            fout.write(json.dumps({
                "prompt": instance["prompt"],
                "completion": instance["completion"],
            }) + "\n")