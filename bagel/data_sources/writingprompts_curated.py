import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import as_conversation, has_refusal

PRIORITY = 2

INSTRUCTION = """Write a short story based on this writing prompt: %s"""

def load_data(known_uids=set([]), **_):
    """WritingPrompts-Curated"""
    logger.info("Loading WritingPrompts-Curated dataset...")
    dataset = load_dataset("euclaise/WritingPrompts_curated")
    data = []
    for item in tqdm(dataset["train"]):
        as_conv = as_conversation(INSTRUCTION % item["prompt"], item["body"])
        if as_conv["id"] in known_uids:
            continue
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
