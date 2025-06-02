import json
import os
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(
        self,
        template_name: str = "",
        verbose: bool = False,
    ):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join(os.getcwd(), "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    @staticmethod
    def get_input(query):
        if query.find("\n") == -1:
            return None
        return "\n".join(query.split("\n")[1:])

    def generate_prompt(self, data_point, data_path, include_label: bool = True) -> str:
        if data_path == "metamathqa":
            instruction = data_point["query"].split("\n")[0]
            input = self.get_input(data_point["query"])
            label = data_point["response"]
        else:
            instruction: str = data_point["instruction"]
            input = data_point["input"]
            label = data_point["output"]
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label and include_label:
            res = f"{res} {label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
