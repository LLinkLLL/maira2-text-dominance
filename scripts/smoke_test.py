#!/usr/bin/env python3
"""Run the official MAIRA-2 phrase-grounding example plus negative controls."""

import argparse
import io
import json
from urllib.request import Request, urlopen

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from image_utils import normalize_cxr_image


DEFAULT_IMAGE = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
DEFAULT_PHRASES = [
    "Pleural effusion.",
    "Right pleural effusion.",
    "Left pleural effusion.",
    "Pneumothorax.",
]


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        request = Request(path_or_url, headers={"User-Agent": "MAIRA-2"})
        return normalize_cxr_image(Image.open(io.BytesIO(urlopen(request).read())))
    with Image.open(path_or_url) as image:
        return normalize_cxr_image(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--phrase", action="append", dest="phrases")
    parser.add_argument("--model-id", default="microsoft/maira-2")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This smoke test expects an NVIDIA GPU.")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    image = load_image(args.image)

    for phrase in args.phrases or DEFAULT_PHRASES:
        inputs = processor.format_and_preprocess_phrase_grounding_input(
            frontal_image=image, phrase=phrase, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        decoded = processor.decode(tokens[0][prompt_length:], skip_special_tokens=True)
        parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
        print(json.dumps({"phrase": phrase, "decoded": decoded, "parsed": parsed}))


if __name__ == "__main__":
    main()
