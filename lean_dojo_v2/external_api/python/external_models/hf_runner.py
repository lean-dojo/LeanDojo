import os
from typing import List, Tuple

import requests
from loguru import logger

from .external_parser import *


class HFTacticGenerator(Generator, Transformer):
    def __init__(
        self, model_name: str = "deepseek-ai/DeepSeek-Prover-V2-671B:novita", **args
    ) -> None:
        self.model_name = model_name
        self.used_tactics = set()

    def generate(self, input: str, target_prefix: str = "") -> List[Tuple[str, float]]:
        prompt = input + target_prefix
        '''prompt= 'Here is a theorem you need to prove in Lean:\n'+prompt+'\nNow you should suggest one line tactic in lean code:'
        prompt = f"""<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
        '''
        prompt = pre_process_input(self.model_name, prompt)

        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "deepseek-ai/DeepSeek-Prover-V2-671B:novita",
            "temperature": 0.6,
            "max_tokens": 256,
            "top_p": 0.9,
            "n": 4,
        }

        response_data = requests.post(API_URL, headers=headers, json=payload).json()
        response = [
            choice["message"]["content"] for choice in response_data.get("choices", [])
        ]

        if response_data.get("error"):
            logger.error(f"Error generating tactic: {response_data['error']}")
            return "sorry"

        result = []
        for out in response:
            out = post_process_output(self.model_name, out)
            result.append(out)

        for r in result:
            if (input, r) not in self.used_tactics:
                self.used_tactics.add((input, r))
                logger.info(f"tactic: {r}")
                return r

        return "sorry"

    def generate_whole_proof(self, theorem: str) -> str:
        prompt = (
            "Here is a theorem you need to prove in Lean:```lean\n"
            + theorem
            + "```\nNow you should provide the complete proof of the theorem in lean code. Only output the proof, no explanation, no comments, no theorem, nothing else:"
        )

        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "deepseek-ai/DeepSeek-Prover-V2-671B:novita",
            "temperature": 0.6,
            "max_tokens": 256,
            "top_p": 0.9,
            "n": 1,
        }

        response_data = requests.post(API_URL, headers=headers, json=payload).json()
        response = [
            choice["message"]["content"] for choice in response_data.get("choices", [])
        ]

        if response_data.get("error"):
            logger.error(f"Error generating tactic: {response_data['error']}")
            return "sorry"

        if len(response) == 0:
            return "sorry"

        out = post_process_output(self.model_name, response[0])

        return out


if __name__ == "__main__":
    model = HFTacticGenerator()
    print(model.generate("n : ℕ\n⊢ gcd n n = n"))
