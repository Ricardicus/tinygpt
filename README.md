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

## Training on a cloned repository

Use Python 3.10 or newer. Folder scanning is recursive, for both tokenizer
training and model pretraining. Edit `training_data.json` to choose suffixes
(case insensitive) and directory names to exclude:

```json
{
  "suffixes": [".txt", ".c", ".cpp", ".md", ".h", ".hpp"],
  "exclude_dirs": [".git", "venv", ".venv", "node_modules", "build", "dist"]
}
```

For example, after cloning a repository into `rawdata/my-project`:

```bash
python main.py --rawdata rawdata/my-project --data-config training_data.json \
  --tokenizer tokenizers/my-project.json --model models/my-project.pt \
  --lower-case false --bpe-part 1 --bpe-max-words 100000
```

Use a fresh tokenizer and model path for a new corpus. Existing tokenizers are
loaded rather than retrained; they may lack characters in new data. For source
code, `--lower-case false` preserves case. Newly trained tokenizers preserve
spaces, tabs, indentation, blank lines, trailing whitespace, and line endings
(including CRLF). Whitespace uses explicit character tokens, with no synthetic
end-of-word markers, so decoding reproduces the original formatting.

Symbolic links are skipped. Files are decoded as UTF-8 with replacement for
invalid bytes; NUL-containing data raises an error identifying the file. Check
that the suffix list selects text files. Words longer than 4096 characters
(such as encoded blobs) are omitted in both stages.

The raw corpus now streams from disk. BPE retains at most `--bpe-max-words`
unique words for merge learning; later unseen words are skipped for merge
statistics, while their characters remain encodable. `--bpe-part` selects the
fraction of chunks used for merges; all accepted text is scanned for character
coverage, and all accepted text is used for model pretraining. The word cap
reduces memory at the cost of less representative merge statistics; lower it
if memory is tight. It is a sampling limit, not a fixed RAM budget.

Disk streaming is used for corpus preparation. During model pretraining, all
context windows across all accepted files share one globally shuffled index.
Each epoch reshuffles that index and visits every complete window once, so
batches sample across the dataset rather than walking through one file at a
time. This is sampling without replacement within each epoch.

Token IDs use four bytes each, and training windows are sliced as needed rather
than duplicated. Token IDs and shuffle indices still reside in memory, so very
large corpora can require substantial RAM. Token caches are rebuilt when source
paths, sizes, modification times, or tokenizer settings change. Old caches
without metadata are rebuilt once.

Previously saved tokenizers keep their original encoding behavior for checkpoint
compatibility. To learn formatting, train a new tokenizer and model using fresh
paths; existing weights cannot be reused with a changed token-to-ID mapping.
Token caches rebuild automatically for the updated text processing.

Run the corpus/tokenizer and transformer regression tests with:

```bash
python -m unittest discover -s tests -v
```
