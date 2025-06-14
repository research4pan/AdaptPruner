"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

alpaca_template = {
    "description": "Template used by Alpaca",
    "prompt_more_context": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_single_context": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

openhermes_template = {
    "description": "Template used by Openhermes",
    "prompt_more_context": "Below is an instruction that describes how to approach the question, paired with an specific question. Write a response that answers the question based on the provided instruction.\n\n### Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Response:\n",
    "prompt_single_context": "Below is a question. Write a response that answers the question appropriately.\n\n### Question:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}
magpie_template = {
    "description": "Template used by magpie",
    "prompt_single_context": "Below is a question. Write a response that answers the question appropriately.\n\n### Question:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"  
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name or template_name == 'alpaca':
            self.template = alpaca_template
        elif template_name == 'openhermes':
            self.template = openhermes_template
        elif template_name == 'magpie':
            self.template = magpie_template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input is not None and instruction is not None:
            res = self.template["prompt_more_context"].format(
                instruction=instruction, input=input
            )
        elif instruction is not None:
            res = self.template["prompt_single_context"].format(
                instruction=instruction
            )
        else:
            raise ValueError(f"Invalid instructoin {instruction}, input {input}, label {label}")
        
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class ZeroPrompter(object):
    __slots__ = ("_verbose")

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        
        if self._verbose:
            print(
                f"Without using prompt template!"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if instruction[-1] == '.':
            instruction = instruction[:-1] + ':'
        if instruction[-1] not in ['.', ':', '?', '!']:
            instruction = instruction + ':'
        instruction += ' '

        if input:
            if input[-1] not in ['.', ':', '?', '!']:
                input = input + '.'
            res = instruction + input
        else:
            res = instruction
        if label:
            res = f"{res} {label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.strip()
