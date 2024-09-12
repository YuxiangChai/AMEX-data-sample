pretrained_path='path/to/checkpoint/'
pretrained_type=consolidated

tokenizer_path="path/to/checkpoint/tokenizer.model"

data_parallel=fsdp
model_parallel=1

RANK=${RANK:-0}

torchrun  --nproc_per_node=2 --nnodes=1 infer.py \
--input_path path/to/l3_test_qa.json \
--output_path path/to/save/infer_results.json \
--sphinx_type internlm \
--tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" \
--model_parallel_size "$model_parallel" \