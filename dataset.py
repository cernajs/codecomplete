
import torch
import numpy as np
import os

import ast

from typing import Tuple, List

import transformers

class CodeCompletionDataset(torch.utils.data.Dataset):

    class Element:
        preffix: str
        suffix: str
        middle: str

        PREFIX_TOKEN = "<fim_prefix>"
        SUFFIX_TOKEN = "<fim_suffix>"
        MIDDLE_TOKEN = "<fim_middle>"

        def __init__(self, preffix: str, suffix: str, middle: str, device = "mps") -> None:
            self.preffix = preffix
            self.suffix = suffix
            self.middle = middle

            self.device = device


        def elements(
            self, tokenizer: transformers.PreTrainedTokenizer
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

            """
            returns input ids, attention mask of input
            and target value
            """

            input_text = f"{self.PREFIX_TOKEN}{self.preffix}\n    {self.SUFFIX_TOKEN}\n    {self.suffix}{self.MIDDLE_TOKEN}"
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
            target_value = tokenizer.encode(self.middle, return_tensors="pt").to(self.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            return input_ids, attention_mask, target_value

        def __repr__(self) -> str:
            return f"Element({self.preffix}, {self.suffix}, {self.middle})"

    class BatchElement:
        def __init__(self, elements: List["CodeCompletionDataset.Element"]) -> None:
            self.batch_elements = elements

        def elements(
            self, tokenizer: transformers.PreTrainedTokenizer
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            returns batched CodeCompletionDataset.Element
            """

            if len(self.batch_elements) == 0:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])

            input_ids = []
            attention_mask = []
            target_values = []
            for element in self.batch_elements:
                input_id, attention, target_value = element.elements(tokenizer)

                input_id, attention = input_id.squeeze(0), attention.squeeze(0)
                target_value = target_value.squeeze(0)

                input_ids.append(input_id)
                attention_mask.append(attention)
                target_values.append(target_value)

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            target_values = torch.nn.utils.rnn.pad_sequence(target_values, batch_first=True, padding_value=tokenizer.pad_token_id)

            return input_ids, attention_mask, target_values

        def __repr__(self) -> str:
            return f"BatchElement len = {len(self.batch_elements)}"

    def __init__(
        self,
        path = "~/dev/npfl138/labs/",
        device = "mps"
    ) -> None:
            self.path = os.path.expanduser(path)
            self.device = device
            self.files = self.load_files()

    def load_files(self) -> List[str]:
        files = []
        for file in os.listdir(self.path):

            if file.endswith(".py"):
                files.append(os.path.join(self.path, file))
                continue

            if not os.path.isdir(os.path.join(self.path, file)):
                continue

            subfolders = os.listdir(os.path.join(self.path, file))
            for subfile in subfolders:
                if subfile.endswith(".py"):
                    files.append(os.path.join(self.path, file, subfile))

        return files

    def __len__(self) -> int:
        return len(self.files)

    def _get_functions(self, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            source = file.read()

        tree = ast.parse(source)

        function_strings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno

                function_code = "\n".join(source.splitlines()[start_line:end_line])
                function_strings.append(function_code)

        return function_strings


    def _split_file(self, file_path: str) -> List["CodeCompletionDataset.Element"]:
        functions = self._get_functions(file_path)

        if not functions:
            return []

        elements = []

        for function in functions:
            lines = function.split("\n")
            lines_without_comments = [line for line in lines if not line.strip().startswith("#")]
            lines = [line for line in lines_without_comments if line.strip()]

            if len(lines) <= 1:
                continue

            drop_line_index = np.random.randint(1, len(lines))

            preffix = "\n".join(lines[:drop_line_index])
            suffix = "\n".join(lines[drop_line_index+1:])

            target = lines[drop_line_index]

            elements.append(CodeCompletionDataset.Element(preffix, suffix, target, self.device))

        return elements

    def __getitem__(self, index: int) -> "CodeCompletionDataset.BatchElement":
        file = self.files[index]
        elements = self._split_file(file)
        return CodeCompletionDataset.BatchElement(elements)
