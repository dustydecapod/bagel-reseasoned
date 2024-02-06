import json
import random
import requests
from loguru import logger
from datasets import Dataset
from .util import has_refusal

PRIORITY = 1


def load_data(known_uids=set([]), **_):
    """OpenHermes-2.5"""
    logger.info("Loading OpenHermes-2.5 dataset...")
    raw_data = json.loads(requests.get(
            "https://huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/main/openhermes2_5.json?download=true"
        ).text)
    keep = []
    i = 0
    for item in raw_data:
        if len(item["conversations"]) < 3:
            continue
        if has_refusal("\n".join([conv["value"] for conv in item["conversations"]])):
            continue

        # If the conversation starts with GPT, we'll add the first message to system prompt, otherwise it messes up llama-2 chat.
        if (
            item["conversations"][0]["from"] == "system"
            and item["conversations"][1]["from"] == "gpt"
        ):
            item["conversations"][0]["value"] = (
                "First message: " + item["conversations"][1]["value"]
            )
            del item["conversations"][1]
        elif item["conversations"][0]["from"] == "gpt":
            continue
        keep.append({"id": i, "conversations": item["conversations"]})
        i += 1
    return Dataset.from_list(keep)


if __name__ == "__main__":
    print(load_data())
