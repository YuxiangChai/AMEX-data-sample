import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import pandas as pd
import torch
import torch.distributed as dist
from accessory.data.conversation import default_conversation
from accessory.data.transform import get_transform
from accessory.model.meta import MetaModel
from accessory.util import misc
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from accessory.util.tensor_type import default_tensor_type
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_size: int, list_path: str) -> None:
        with open(list_path, "r") as f:
            test_qa = json.load(f)

        self.urls = [qa["image"] for qa in test_qa]

        self.prompts = [qa["conversations"][0]["value"] for qa in test_qa]

        self.gts = [qa["conversations"][1]["value"] for qa in test_qa]

        self.transform = get_transform("padded_resize", image_size)
        self.tokenizer = None

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        url = self.urls[idx]
        img = Image.open(url).convert("RGB")
        img = self.transform(img)
        prompt = self.prompts[idx]
        gt = self.gts[idx]
        return img, url, prompt, gt


def get_local_indices(rank: int, world_size: int, dataset_len: int) -> List[int]:
    indices = list(range(dataset_len))
    while len(indices) % world_size != 0:
        indices.extend(indices[: world_size - len(indices) % world_size])
    indices = indices[rank::world_size]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()

    # data configuration
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="file path")
    parser.add_argument("--prompt", type=str, default="")

    # SPHINX model configuration
    parser.add_argument(
        "--sphinx_type", type=str, choices=["SPHINX", "SPHINX-1k", "internlm"]
    )
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--model_parallel_size", type=int, choices=[1, 2])
    parser.add_argument("--batch-size", type=int, default=10)

    # generation configuration
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.75)

    args = parser.parse_args()

    if args.sphinx_type == "SPHINX-1k":
        args.llama_type = "llama_ens5"  # SPHINX-1k
    elif args.sphinx_type == "SPHINX":
        args.llama_type = "llama_ens"
    elif args.sphinx_type == "internlm":
        args.llama_type = "internlm_ems5_light"

    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)

    with default_tensor_type(dtype=torch.bfloat16, device="cuda"):
        model = MetaModel(
            args.llama_type,
            llama_config=[],
            tokenizer_path=args.tokenizer_path,
            with_visual=True,
            max_seq_len=4096,
        )
    print("Loading pretrained weights ...")
    load_result = load_tensor_parallel_model_list(model, [args.pretrained_path])
    print("load result:\n", load_result)
    assert load_result == {
        "missing_keys": [],
        "unexpected_keys": [],
    }, "checkpoint and model mismatch"
    model.eval()

    dataset = Dataset(
        getattr(model.llma, "image_size", 448), args.input_path
    )  # 448 for SPHINX-1k, 224 for SPHINX
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=get_local_indices(
            fs_init.get_data_parallel_rank(),
            fs_init.get_data_parallel_world_size(),
            len(dataset),
        ),
    )

    conv = default_conversation()
    conv.load_qas([[args.prompt, None]])
    prompt = conv.get_prompt()
    conv_sep = conv.response_end_signal

    if dist.get_rank() == 0:
        print("Formatted prompt:", repr(prompt))
        from tqdm import tqdm
    else:
        tqdm = lambda x: x

    urls, captions = [], []
    prompts, gts = [], []
    for image, batch_urls, batch_prompts, batch_gts in tqdm(dataloader):
        image = image.cuda(non_blocking=True)
        model_dtype = next(model.parameters()).dtype
        image = image.to(dtype=model_dtype)
        if fs_init.get_model_parallel_world_size() > 1:
            dist.broadcast(
                image,
                src=fs_init.get_model_parallel_src_rank(),
                group=fs_init.get_model_parallel_group(),
            )

        used_prompt = []
        for prompt_ in batch_prompts:
            conv = default_conversation()
            conv.load_qas([[prompt_, None]])
            prompt = conv.get_prompt()

            used_prompt.append(prompt)

        generated = model.generate(
            used_prompt,
            image,
            args.max_gen_len,
            args.temperature,
            args.top_p,
        )

        truncated = []
        for cap in generated:
            end_pos = cap.find(conv_sep)
            if end_pos != -1:
                cap = cap[:end_pos].rstrip()
            truncated.append(cap)

        if fs_init.get_data_parallel_world_size() > 1:
            truncated_allgather = [
                None for _ in range(fs_init.get_data_parallel_world_size())
            ]
            batch_urls_allgather = [
                None for _ in range(fs_init.get_data_parallel_world_size())
            ]
            batch_prompts_allgather = [
                None for _ in range(fs_init.get_data_parallel_world_size())
            ]
            batch_gts_allgather = [
                None for _ in range(fs_init.get_data_parallel_world_size())
            ]
            dist.all_gather_object(
                truncated_allgather, truncated, fs_init.get_data_parallel_group()
            )
            dist.all_gather_object(
                batch_urls_allgather, batch_urls, fs_init.get_data_parallel_group()
            )
            dist.all_gather_object(
                batch_prompts_allgather,
                batch_prompts,
                fs_init.get_data_parallel_group(),
            )
            dist.all_gather_object(
                batch_gts_allgather,
                batch_gts,
                fs_init.get_data_parallel_group(),
            )
            batch_urls = list(sum(zip(*batch_urls_allgather), ()))
            truncated = list(sum(zip(*truncated_allgather), ()))
            batch_prompts = list(sum(zip(*batch_prompts_allgather), ()))
            batch_gts = list(sum(zip(*batch_gts_allgather), ()))

        if dist.get_rank() == 0:
            urls.extend(batch_urls)
            captions.extend(truncated)
            prompts.extend(batch_prompts)
            gts.extend(batch_gts)

            # for url, caption in zip(batch_urls, truncated):
            #     print(url)
            #     print(caption)
            #     print("==========")

    if dist.get_rank() == 0:
        urls = urls[: len(dataset)]
        captions = captions[: len(dataset)]
        prompts = prompts[: len(dataset)]
        gts = gts[: len(dataset)]
        with open(Path(args.output_path), "w") as f:
            json.dump(
                [
                    {
                        "image": url,
                        "caption": caption,
                        "prompt": prompt,
                        "gt": gt,
                    }
                    for url, caption, prompt, gt in zip(urls, captions, prompts, gts)
                ],
                f,
                indent=4,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
