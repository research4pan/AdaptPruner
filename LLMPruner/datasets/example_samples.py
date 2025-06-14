import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

from LLMPruner.utils.prompter import Prompter, ZeroPrompter

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train', trust_remote_code=True
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_slimpajama(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'DKYoon/SlimPajama-6B', split='train', trust_remote_code=True
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_openhermes(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'teknium/OpenHermes-2.5', split='train', trust_remote_code=True
    )
    prompter = Prompter("openhermes")
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            data_point = traindata[i]
            if data_point['conversations'][0]['from'] == 'system':
                assert data_point['conversations'][1]['from'] == 'human'
                assert data_point['conversations'][2]['from'] == 'gpt'
                full_prompt = prompter.generate_prompt(
                    instruction=data_point['conversations'][0]['value'],
                    input=data_point['conversations'][1]['value'],
                    label=data_point['conversations'][2]['value']
                )
            elif data_point['conversations'][0]['from'] == 'human':
                assert data_point['conversations'][1]['from'] == 'gpt'
                full_prompt = prompter.generate_prompt(
                    instruction=data_point['conversations'][0]['value'],
                    input=None,
                    label=data_point['conversations'][1]['value']
                )
            tokenized_sample = tokenizer(full_prompt, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_examples(dataset, tokenizer, n_samples, seq_len):
    if dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'openhermes':
        return get_openhermes(tokenizer, n_samples, seq_len)
    elif dataset == 'slimpajama':
        return get_slimpajama(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
