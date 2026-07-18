import subprocess
import sys


def test_faithful_generation_falls_back_without_gpt2_cache():
    code = r'''
from transformers import GPT2Tokenizer

def unavailable(*args, **kwargs):
    assert kwargs.get("local_files_only") is True
    raise OSError("GPT-2 cache unavailable")

GPT2Tokenizer.from_pretrained = unavailable
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset, faithful_token_edit_vocab,
)
dataset = FaithfulTokenEditDataset(
    faithful_token_edit_vocab(), size=1, seed=1,
)
assert dataset[0]["actions"]
'''
    subprocess.run([sys.executable, "-c", code], check=True)
