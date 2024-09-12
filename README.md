# AMEX-codebase

The repo in based on [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory). Follow the [instructions](https://llama2-accessory.readthedocs.io/en/latest/install.html) to install. Then replace the `accessory` folder with the one in this repo. Model checkpoint is at [Huggingface](https://huggingface.co/SiyuanH/GUIAgent-InternLM7B/tree/main).

## Data Preparation

Run the following command in `data_utils`:

```bash
python amex_to_qa.py --level all --root-dir /path/to/AMEX
```

You can change the `level` to 'l1', 'l2', 'l3', 'all' to process different levels of data.

## Finetune

Follow the [instructions](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX#finetune-sphinx) to finetune the model.

## Inference

Follow the [instructions](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX#inference-1) to inference. Or run the following command in `accessory`:

```bash
bash infer.sh
```
