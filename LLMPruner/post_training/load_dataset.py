from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import torch
import datasets
from functools import partial
from itertools import chain

from LLMPruner.utils.template import custom_template


def get_train_val_data(data_path, tokenizer, cutoff_len, val_set_size, num_proc, data_point_number=None, fraction=1):
    data_paths = [path.strip() for path in data_path.split(',')]
    print(data_paths)

    def process_and_count(data_point, process_func):
        processed = process_func(data_point)

        token_count = len(processed['input_ids'])
        return {
            **processed,
            'token_count': token_count
        }

    def count(data_point):
        token_count = len(data_point['input_ids'])
        return {
            **data_point,
            'token_count': token_count
        }
    
    def tokenize(prompt, truncation=True, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=truncation,
            max_length=cutoff_len if truncation else None,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def process_openhermes_slimorca_wizardlmevolinstruct_guanaco(data_point):
        conversations = data_point['conversations']
        messages = []
        segment_lengths = []
        if conversations[0]['from'] == 'system':
            system_message = {"role": "system", "content": conversations[0]['value']}
        else:
            system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages.append(system_message)
        system_tokens = tokenize(tokenizer.apply_chat_template([system_message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
        segment_lengths.append(len(system_tokens["input_ids"]))
        
        for i in range(len(conversations)):
            if conversations[i]['from'] == 'human':
                message = {"role": "user", "content": conversations[i]['value']}
                messages.append(message)
                user_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(user_tokens["input_ids"]))
            elif conversations[i]['from'] == 'gpt':
                message = {"role": "assistant", "content": conversations[i]['value']}
                messages.append(message)
                assistant_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(assistant_tokens["input_ids"]))
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        is_assistant = []
        for i, length in enumerate(segment_lengths):
            is_assistant.extend([messages[i]["role"] == "assistant"] * length)
        is_assistant.append(True)   # caused by add_eos_token in tokenized_full_prompt["labels"]

        tokenized_full_prompt["labels"] = [
            label if is_assistant[i] else -100 
            for i, label in enumerate(tokenized_full_prompt["labels"])
        ]

        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }

    def process_magpie(data_point):
        conversation = data_point['conversation']
        messages = []
        segment_lengths = []
        if conversation[0]['role'] == 'system':
            system_message = {"role": "system", "content": conversation[0]['value']}
        else:
            system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages.append(system_message)
        system_tokens = tokenize(tokenizer.apply_chat_template([system_message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
        segment_lengths.append(len(system_tokens["input_ids"]))
        
        for i in range(len(conversation)):
            if conversation[i]['role'] == 'human':
                message = {"role": "user", "content": conversation[i]['value']}
                messages.append(message)
                user_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(user_tokens["input_ids"]))
            elif conversation[i]['role'] == 'gpt':
                message = {"role": "assistant", "content": conversation[i]['value']}
                messages.append(message)
                assistant_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(assistant_tokens["input_ids"]))
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        is_assistant = []
        for i, length in enumerate(segment_lengths):
            is_assistant.extend([messages[i]["role"] == "assistant"] * length)
        is_assistant.append(True)   # caused by add_eos_token in tokenized_full_prompt["labels"]

        tokenized_full_prompt["labels"] = [
            label if is_assistant[i] else -100 
            for i, label in enumerate(tokenized_full_prompt["labels"])
        ]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_oasst2_ultrachat(data_point):
        conversation = data_point['messages']
        messages = []
        segment_lengths = []
        if conversation[0]['role'] == 'system':
            system_message = {"role": "system", "content": conversation[0]['content']}
        else:
            system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages.append(system_message)
        system_tokens = tokenize(tokenizer.apply_chat_template([system_message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
        segment_lengths.append(len(system_tokens["input_ids"]))
        
        for i in range(len(conversation)):
            if conversation[i]['role'] == 'user':
                message = {"role": "user", "content": conversation[i]['content']}
                messages.append(message)
                user_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(user_tokens["input_ids"]))
            elif conversation[i]['role'] == 'assistant':
                message = {"role": "assistant", "content": conversation[i]['content']}
                messages.append(message)
                assistant_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(assistant_tokens["input_ids"]))
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        is_assistant = []
        for i, length in enumerate(segment_lengths):
            is_assistant.extend([messages[i]["role"] == "assistant"] * length)
        is_assistant.append(True)   # caused by add_eos_token in tokenized_full_prompt["labels"]

        tokenized_full_prompt["labels"] = [
            label if is_assistant[i] else -100 
            for i, label in enumerate(tokenized_full_prompt["labels"])
        ]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_daringanteater(data_point):
        conversation = data_point['conversations']
        messages = []
        segment_lengths = []
        if data_point['system'] != '':
            system_message = {"role": "system", "content": data_point['system']}
        else:
            system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages.append(system_message)
        system_tokens = tokenize(tokenizer.apply_chat_template([system_message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
        segment_lengths.append(len(system_tokens["input_ids"]))
        
        for i in range(len(conversation)):
            if conversation[i]['from'] == 'User':
                message = {"role": "user", "content": conversation[i]['value']}
                messages.append(message)
                user_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(user_tokens["input_ids"]))
            elif conversation[i]['from'] == 'Assistant':
                message = {"role": "assistant", "content": conversation[i]['value']}
                messages.append(message)
                assistant_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(assistant_tokens["input_ids"]))
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        is_assistant = []
        for i, length in enumerate(segment_lengths):
            is_assistant.extend([messages[i]["role"] == "assistant"] * length)
        is_assistant.append(True)   # caused by add_eos_token in tokenized_full_prompt["labels"]

        tokenized_full_prompt["labels"] = [
            label if is_assistant[i] else -100 
            for i, label in enumerate(tokenized_full_prompt["labels"])
        ]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_wildchat_lmsyschat(data_point):
        conversation = data_point['conversation']
        messages = []
        segment_lengths = []
        if conversation[0]['role'] == 'system':
            system_message = {"role": "system", "content": conversation[0]['content']}
        else:
            system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages.append(system_message)
        system_tokens = tokenize(tokenizer.apply_chat_template([system_message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
        segment_lengths.append(len(system_tokens["input_ids"]))
        
        for i in range(len(conversation)):
            if conversation[i]['role'] == 'user':
                message = {"role": "user", "content": conversation[i]['content']}
                messages.append(message)
                user_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(user_tokens["input_ids"]))
            elif conversation[i]['role'] == 'assistant':
                message = {"role": "assistant", "content": conversation[i]['content']}
                messages.append(message)
                assistant_tokens = tokenize(tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False), add_eos_token=False)
                segment_lengths.append(len(assistant_tokens["input_ids"]))
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        is_assistant = []
        for i, length in enumerate(segment_lengths):
            is_assistant.extend([messages[i]["role"] == "assistant"] * length)
        is_assistant.append(True)   # caused by add_eos_token in tokenized_full_prompt["labels"]

        tokenized_full_prompt["labels"] = [
            label if is_assistant[i] else -100 
            for i, label in enumerate(tokenized_full_prompt["labels"])
        ]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_1milliongpt4_openorca(data_point):
        messages = [
            {"role": "system", "content": data_point["system_prompt"]}, 
            {"role": "user", "content": data_point["question"]}, 
            {"role": "assistant", "content": data_point["response"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": data_point["system_prompt"]}, 
            {"role": "user", "content": data_point["question"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_databricksdolly15k(data_point):
        messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["context"]}, 
            {"role": "assistant", "content": data_point["response"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["context"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_gpt4llmcleaned_alpacacleaned_alpaca(data_point):
        messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["input"]}, 
            {"role": "assistant", "content": data_point["output"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["input"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_gpteachergeneralinstruct(data_point):
        messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["input"]}, 
            {"role": "assistant", "content": data_point["response"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": data_point["instruction"]}, 
            {"role": "user", "content": data_point["input"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_ultrainteract(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"]}, 
            {"role": "assistant", "content": data_point["response"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_webinstructsub_gsm8k(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["question"]}, 
            {"role": "assistant", "content": data_point["answer"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["question"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_p3(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["inputs_pretokenized"]}, 
            {"role": "assistant", "content": data_point["targets_pretokenized"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["inputs_pretokenized"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_magicoder(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["problem"]}, 
            {"role": "assistant", "content": data_point["solution"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["problem"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_metamathqa(data_point):
        messages = [
            {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"},
            {"role": "user", "content": data_point["query"]}, 
            {"role": "assistant", "content": data_point["response"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"},
            {"role": "user", "content": data_point["query"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_openo1sft(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"]}, 
            {"role": "assistant", "content": data_point["output"]}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenize(full_prompt)

        user_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"]}
        ]
        user_full_prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        tokenized_user_full_prompt = tokenize(user_full_prompt)

        tokenized_full_prompt["labels"] = [-100] * len(tokenized_user_full_prompt["input_ids"]) + \
            tokenized_full_prompt["labels"][len(tokenized_user_full_prompt["input_ids"]):]
        
        assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])

        return {
            "input_ids": tokenized_full_prompt["input_ids"],
            "attention_mask": tokenized_full_prompt["attention_mask"],
            "labels": tokenized_full_prompt["labels"]
        }
    
    def process_pretrain_text(data_point):
        tokenized = tokenize(data_point['text'], truncation=False)
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"]
        }
    
    def group_texts(tokenized):
        concatenated_examples = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
        total_length = len(concatenated_examples[list(tokenized.keys())[0]])
        
        result = {
            k: [t[i : i + cutoff_len] for i in range(0, total_length, cutoff_len)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    tokenizer.chat_template = custom_template

    all_train_data = []
    all_val_data = {}
    total_tokens = 0

    for data_path in data_paths:
        print(f"Processing dataset: {data_path}")
        # Load dataset
        if "gsm8k" in data_path.lower():
            data = load_dataset(data_path, 'main')
        elif "fineweb" in data_path.lower():
            data = load_dataset(data_path, 'sample-10BT')
        elif "zyda-2" in data_path.lower():
            data = load_dataset(data_path, 'sample-100BT')
        elif "books" in data_path.lower():
            data = load_from_disk(data_path)
        else:
            data = load_dataset(data_path)
        
        # Limit data points if specified
        if data_point_number:
            data["train"] = data["train"].select(
                range(min(data_point_number, len(data["train"])))
            )

        # load dataset processor
        if any(name in data_path.lower() for name in ["openhermes-2.5", "slimorca", "wizardlm_evol_instruct", "guanaco"]):
            process_func = process_openhermes_slimorca_wizardlmevolinstruct_guanaco
        elif "magpie-ultra-v1.0" in data_path.lower():
            process_func = process_magpie
        elif any(name in data_path.lower() for name in ["oasst2_curated", "ultrachat"]):
            process_func = process_oasst2_ultrachat
        elif any(name in data_path.lower() for name in ["wildchat-1m", "lmsys-chat-1m"]):
            process_func = process_wildchat_lmsyschat
        elif any(name in data_path.lower() for name in ["1million-gpt-4", "openorca"]):
            process_func = process_1milliongpt4_openorca
        elif "databricks-dolly-15k" in data_path.lower():
            process_func = process_databricksdolly15k
        elif any(name in data_path.lower() for name in ["gpt4-llm-cleaned", "alpaca-cleaned", "alpaca", "python_code_instructions_18k_alpaca"]):
            process_func = process_gpt4llmcleaned_alpacacleaned_alpaca
        elif "gpteacher-general-instruct" in data_path.lower():
            process_func = process_gpteachergeneralinstruct
        elif "ultrainteract_sft" in data_path.lower():
            process_func = process_ultrainteract
        elif any(name in data_path.lower() for name in ["webinstructsub", "gsm8k"]):
            process_func = process_webinstructsub_gsm8k
        elif "daring-anteater" in data_path.lower():
            process_func = process_daringanteater
        elif "p3" in data_path.lower():
            process_func = process_p3
        elif "magicoder-oss-instruct-75k" in data_path.lower():
            process_func = process_magicoder
        elif "metamathqa" in data_path.lower():
            process_func = process_metamathqa
        elif "openo1-sft" in data_path.lower():
            process_func = process_openo1sft
        elif any(name in data_path.lower() for name in ["books", "slimpajama", "dolma", "fineweb", "zyda"]):
            process_func = process_pretrain_text
        else:
            raise NotImplementedError

        if "ultrachat" in data_path.lower():
            train_key = "train_sft"
            val_key = "test_sft"
        else:
            train_key = "train"
            val_key = "test"
        
        if any(name in data_path.lower() for name in ["books", "slimpajama", "dolma", "fineweb", "zyda"]):
            # pretrain datasets
            train_data = data[train_key].shuffle(seed=42)
    
            # If it's dolma dataset, only take 1/50 of the data
            if "dolma" in data_path.lower():
                total_size = len(train_data)
                subset_size = total_size // 30
                train_data = train_data.select(range(subset_size))
            elif "zyda" in data_path.lower():
                total_size = len(train_data)
                subset_size = total_size // 10
                train_data = train_data.select(range(subset_size))
            
            tokenized_train = train_data.map(
                process_pretrain_text,
                remove_columns=data[train_key].column_names,
                num_proc=num_proc
            )

            processed_train = tokenized_train.map(
                group_texts,
                batched=True, 
                batch_size=1024,
                num_proc=num_proc
            )

            processed_train = processed_train.map(
                count,
                num_proc=num_proc
            )

        else:
            # Instruction following datasets
            if val_set_size > 0:
                # Split into train and validation
                train_val = data[train_key].train_test_split(
                    test_size=val_set_size, shuffle=True, seed=42
                )
                
                # Process train split
                processed_train = train_val[train_key].shuffle(seed=42).map(
                    partial(process_and_count, process_func=process_func),
                    remove_columns=data[train_key].column_names,
                    num_proc=num_proc
                )
                
                # Process validation split
                processed_val = train_val[val_key].shuffle(seed=42).map(
                    process_func,
                    remove_columns=data[train_key].column_names,
                    num_proc=num_proc
                )
                all_val_data[data_path] = processed_val
            else:
                # Process entire dataset as train
                processed_train = data[train_key].shuffle(seed=42).map(
                    partial(process_and_count, process_func=process_func),
                    remove_columns=data[train_key].column_names,
                    num_proc=num_proc
                )
        
        '''if any(name in data_path.lower() for name in ["books", "slimpajama", "dolma", "fineweb", "zyda"]):
            train_size = 6000000000 // cutoff_len
            processed_train = processed_train.select(range(train_size))
            print(f"Using {train_size:.2%} of processed dataset: {train_size} samples")'''
        if fraction < 1:
            train_size = int(len(processed_train) * fraction)
            processed_train = processed_train.select(range(train_size))
            print(f"Using {fraction:.2%} of processed dataset: {train_size} samples")
            
        # count tokens of each dataset
        cur_tokens = sum(processed_train['token_count'])
        print(f"Dataset: {data_path} contains {cur_tokens:,} tokens")
        print(f"Dataset: {data_path} contains {len(processed_train):,} samples")
        total_tokens += cur_tokens
        processed_train = processed_train.remove_columns(['token_count'])
        
        all_train_data.append(processed_train)

        print(f"Dataset: {data_path} successfully processed")

    print(f"All dataset contains {total_tokens:,} tokens in total")

    # Combine all training data
    combined_train = concatenate_datasets(all_train_data)
    combined_train = combined_train.shuffle(seed=42)
    
    return combined_train, all_val_data if val_set_size > 0 else None

def load_tokenized_dataset_for_training(data_path):
    print(f"Loading dataset from {data_path}")
    return load_from_disk(data_path), None
