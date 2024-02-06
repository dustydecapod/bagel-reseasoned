import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import as_conversation, has_refusal

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Dolphin-Coder"""
    logger.info("Loading Dolphin-Coder dataset...")
    dataset = load_dataset("cognitivecomputations/dolphin-coder")
    data = []
    for item in tqdm(dataset["train"]):
        if has_refusal(item["response"]):
            continue
        as_conv = as_conversation(item["question"], item["response"])
        if as_conv["id"] in known_uids:
            continue
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
