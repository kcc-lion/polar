import torch
from lm_eval.tasks import TaskManager, get_task_dict


def generate_lm_eval_prompts(task_name: str, verbose: bool = True):
    task_manager = TaskManager()
    tasks = get_task_dict(task_name, task_manager)
    task = tasks[task_name]
    train_dataset = task.training_docs()
    target_delimiter = task.config.target_delimiter
    # General logic: doc_to_text(doc) + target_delimiter + doc_to_target(doc)
    count = 0
    prompts = []
    for doc in train_dataset:
        if task_name == "winogrande":
            # To the best of my knowledge, the preprocessing functions in winogrande, might be mixed up
            # doc_to_text mapping to the target idx and doc_to_target giving the remaining part
            # TODO: check lm_eval/tasks/winogrande/preprocess_winogrande.py
            text = task.doc_to_choice(doc)[task.doc_to_text(doc)]
            correct_choice = task.doc_to_target(doc)
        else:
            text = task.doc_to_text(doc)
            correct_choice = get_gold_answer(task, doc)
        prompt = text + target_delimiter + correct_choice
        count += 1
        prompts.append({"prompt": prompt})
        if verbose and count < 3:
            count += 1
            print(prompt)
            print("\n")
    torch.cuda.empty_cache()
    return prompts


def get_gold_answer(task, doc):
    gold_target = task.doc_to_target(doc)
    if task.config.doc_to_choice is not None:
        choices = task.doc_to_choice(doc)
        if isinstance(gold_target, int):
            return choices[gold_target]
        elif isinstance(gold_target, list):
            return ", ".join(choices[idx] for idx in gold_target)
        else:
            return str(gold_target)
    else:
        if isinstance(gold_target, list):
            return ", ".join(str(t) for t in gold_target)
        else:
            return str(gold_target)
