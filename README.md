# Tiny GPT

Build your own language model on your own data.
BPE tokenizer, only supports pretraining and SFI at the moment, no RL.

The model arcitecture is a multi-head attention transformer with optional
Q/K normalization as well as MoE. 

This project has been built with AI assistance.

## How 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
``` 

Check out the CLI flags:

```bash
$ python main.py --help
usage: main.py [-h] [--tokenizer TOKENIZER] [--vocab_size VOCAB_SIZE] [--verbose VERBOSE] [--lower-case LOWER_CASE] [--bpe-part BPE_PART] [--epochs EPOCHS] [--context-length CONTEXT_LENGTH] [--d-model D_MODEL] [--n-heads N_HEADS]
               [--num-layers NUM_LAYERS] [--lr LR] [--qk-norm | --no-qk-norm] [--moe | --no-moe] [--num-experts NUM_EXPERTS] [--moe-top-k MOE_TOP_K] [--moe-aux-loss-weight MOE_AUX_LOSS_WEIGHT] [--compile-qk-norm] [--compile-model] [--beta1 BETA1]
               [--beta2 BETA2] [--lr-schedule {constant,cosine}] [--warmup-steps WARMUP_STEPS] [--lr-decay-steps LR_DECAY_STEPS] [--min-lr-ratio MIN_LR_RATIO] [--batch-size BATCH_SIZE] [--model MODEL] [--data-config DATA_CONFIG]
               [--bpe-max-words BPE_MAX_WORDS] [--rawdata RAWDATA] [--oneliners ONELINERS] [--device DEVICE] [--precision {auto,float32,bfloat16}] [--bpe-alphabet-policy {limit,expand,error}] [--bpe-max-chars BPE_MAX_CHARS] [--prompt PROMPT]
               [--max-new-tokens MAX_NEW_TOKENS] [--log-progress LOG_PROGRESS] [--monitor-layer-norms CSV] [--monitor-layer-norms-every N] [--save-every-steps SAVE_EVERY_STEPS] [--validation-fraction VALIDATION_FRACTION]
               [--validation-every-steps VALIDATION_EVERY_STEPS] [--validation-batches VALIDATION_BATCHES] [--validation-seed VALIDATION_SEED] [--temperature TEMPERATURE] [--top-k TOP_K] [--seed SEED]

Train TinyGPT using a BPE tokenizer with streaming data.

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        Path to tokenizer file (will load or create). Default: tokenizer.json.
  --vocab_size VOCAB_SIZE
                        Requested BPE vocabulary size (default: 1920); oversized alphabets are limited by default.
  --verbose VERBOSE     Enable verbose output (default: true).
  --lower-case LOWER_CASE
                        Only handle lower-case (default: true).
  --bpe-part BPE_PART   Fraction of dataset used to train BPE.
  --epochs EPOCHS       Training epochs for TinyGPT.
  --context-length CONTEXT_LENGTH
                        Context length (sequence length). Default: {default_model_context_length}.
  --d-model D_MODEL     Embedding size (model width). Default 240
  --n-heads N_HEADS     Number of heads in tranformer multi head self attention block. Default 5
  --num-layers NUM_LAYERS
                        Number of transformer layers. Default 6.
  --lr LR               Learning rate (cosine peak): 0.0003 for new runs; omitted on resume preserves the saved rate/schedule, explicitly supplied overrides it.
  --qk-norm, --no-qk-norm
                        Enable per-head Q/K RMSNorm for fresh models. On resume/inference, the saved architecture takes precedence; old checkpoints disable it. (default: False)
  --moe, --no-moe       Use sparse MoE MLPs for fresh models; saved architecture wins on resume. (default: False)
  --num-experts NUM_EXPERTS
  --moe-top-k MOE_TOP_K
  --moe-aux-loss-weight MOE_AUX_LOSS_WEIGHT
  --compile-qk-norm     Compile Q/K RMSNorm forward/backward with Inductor (training only; requires QK normalization). Execution option, not saved in checkpoints.
  --compile-model       Compile the whole model with Inductor during training; takes precedence over --compile-qk-norm. Not saved in checkpoints.
  --beta1 BETA1         AdamW beta1: default 0.9 for new runs; omitted on resume preserves the saved value.
  --beta2 BETA2         AdamW beta2: default 0.999 for new runs; omitted on resume preserves the saved value.
  --lr-schedule {constant,cosine}
                        Default: constant for new runs, saved schedule on resume. Cosine enables linear warmup then cosine decay.
  --warmup-steps WARMUP_STEPS
                        Cosine warmup updates; 0 disables warmup. Default: min(1000, total schedule updates // 10).
  --lr-decay-steps LR_DECAY_STEPS
                        Total schedule updates INCLUDING warmup; default: planned remaining training updates. Saved on resume.
  --min-lr-ratio MIN_LR_RATIO
                        Cosine floor as a fraction of peak --lr (default: 0.1).
  --batch-size BATCH_SIZE
                        Batch size (approximate, streaming). Default 16.
  --model MODEL         Where to store the model data.
  --data-config DATA_CONFIG
                        JSON suffix/excluded-directory config (default: training_data.json beside main.py).
  --bpe-max-words BPE_MAX_WORDS
                        Maximum unique words retained for BPE merges (default: 100000).
  --rawdata RAWDATA     Folder where raw text corpus data resides.
  --oneliners ONELINERS
                        Files that are best read one at a time as text corpus
  --device DEVICE       Compute device.
  --precision {auto,float32,bfloat16}
                        Training precision: auto uses native CUDA BF16 when supported, otherwise FP32. Checkpoint weights stay FP32.
  --bpe-alphabet-policy {limit,expand,error}
                        New tokenizer behavior when characters exceed vocabulary capacity: limit rare characters to <unk> (default), expand vocabulary, or error.
  --bpe-max-chars BPE_MAX_CHARS
                        Optional base alphabet token cap, including reserved symbols and <unk>, for policy limit; 0 caps to half the vocabulary only when it overflows.
  --prompt PROMPT       Text prompt to generate from. If set together with --model, training is skipped.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate during inference.
  --log-progress LOG_PROGRESS
                        Path to CSV file to append training progress (iteration, loss, mean_loss).
  --monitor-layer-norms CSV
                        Append block norms and selected per-parameter norms to CSV files during training.
  --monitor-layer-norms-every N
                        Record layer norms every N optimizer steps (default: 1).
  --save-every-steps SAVE_EVERY_STEPS
                        Save the latest model every N optimizer steps (default: 1000; 0 disables interval saves). Epoch-end saves always run.
  --validation-fraction VALIDATION_FRACTION
                        Fraction of whole files held out from training (default: 0, disabled).
  --validation-every-steps VALIDATION_EVERY_STEPS
                        Evaluate every N optimizer steps; 0 evaluates only at start and epoch end.
  --validation-batches VALIDATION_BATCHES
                        Maximum batches per evaluation, using the same sampled windows (default: 100).
  --validation-seed VALIDATION_SEED
                        Seed for the file split and validation window sample (default: 42).
  --temperature TEMPERATURE
                        Sampling temperature; lower is less random, 0 is greedy (default: 1).
  --top-k TOP_K         Sample from the top K tokens; 0 uses the full vocabulary (default: 0).
  --seed SEED           Random seed for repeatable inference.
````

## More info

See details.md
