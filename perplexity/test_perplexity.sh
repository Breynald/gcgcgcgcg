#!/bin/bash

export CUDA_VISIBLE_DEVICES=7


python /work/table-fp/nanoGCG-main/perplexity/perplexity_calculator.py --file /work/table-fp/nanoGCG-main/perplexity/test_text.txt --model /work/models/Qwen/Qwen2.5-1.5B-Instruct