# Tiny GPT

Build your own language model on your own data.
BPE tokenizer, only supports pretraining, no RL.

## How 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
``` 

Check out the CLI flags:

```bash
$ python main.py -h
usage: main.py [-h] [--tokenizer TOKENIZER] [--vocab_size VOCAB_SIZE] [--verbose VERBOSE] [--lower-case LOWER_CASE] [--bpe-part BPE_PART]
               [--epochs EPOCHS] [--context-length CONTEXT_LENGTH] [--d-model D_MODEL] [--n-heads N_HEADS] [--num-layers NUM_LAYERS] [--lr LR]
               [--batch-size BATCH_SIZE] [--model MODEL] [--rawdata RAWDATA] [--oneliners ONELINERS] [--device DEVICE] [--prompt PROMPT]
               [--max-new-tokens MAX_NEW_TOKENS]

Train TinyGPT using a BPE tokenizer with streaming data.

options:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        Path to tokenizer file (will load or create). Default: tokenizer.json.
  --vocab_size VOCAB_SIZE
                        Vocabulary size for BPE training (default: 1920).
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
  --lr LR               Learning rate. Default 0.0003.
  --batch-size BATCH_SIZE
                        Batch size (approximate, streaming). Default 16.
  --model MODEL         Where to store the model data.
  --rawdata RAWDATA     Folder where raw text corpus data resides.
  --oneliners ONELINERS
                        Files that are best read one at a time as text corpus
  --device DEVICE       Compute device.
  --prompt PROMPT       Text prompt to generate from. If set together with --model, training is skipped.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate during inference.
```

Place your raw data files under the folder "rawdata" and the program will by default
train a BPE tokenizer on the data. With --bpe-part you can set a percentage of all
the data included in the tokenization effort. It typically takes a while.

Once the tokenizer is trained, it will try to train the model on your data.

If you provide the program with --model it will read its hyperparameters from that
saved pytorch-file. 

# Other

Not sure about what data to train on? Under other/ there is a script that downloads
the top 10 books in text format from Project Gutenberg. If you use the flag --top50 it
instead downloads the top 50 books. That can serve as a dataset if you don't know 
what to train on. 
