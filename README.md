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

Training saves the latest model and optimizer to `--model` every 1,000 optimizer
steps and at the end of each epoch. Set `--save-every-steps 500` to change the
interval, or `0` for epoch-end saves only. Each save atomically replaces the
previous checkpoint. Restarting with the same model path resumes saved weights
and optimizer state; a checkpoint saved during an epoch restarts that epoch with
a new shuffle rather than restoring the exact data position. Changes to the
save interval apply when launching a new training process.

Enable validation with `--validation-fraction 0.05` (the prepared dataset's
`train.sh` uses this by default after refreshing helpers). A deterministic split
holds out 5% of whole files from model training. Validation reports token-weighted
cross-entropy loss and perplexity before training, every 1,000 optimizer steps,
and at epoch end. Lower is better. Use `--validation-every-steps 500` to change
the interval and `--validation-batches 100` to cap evaluation cost. Every check
uses the same sampled windows across the held-out corpus. With `--log-progress`,
validation records go into a separate `<log path>.validation.csv` file.

The split is by file count rather than bytes and repeats with the same corpus
and `--validation-seed` (default 42). Training and validation use separate token
caches; enabling the split requires tokenization once. The tokenizer vocabulary
is shared; a new tokenizer learns merges from training files and includes
held-out characters for encoding. A prepared tokenizer may have learned merges
from the full corpus. Models trained before validation was enabled have already
seen the corpus, so use a fresh model path for an unseen-data evaluation.
Disable validation with `--validation-fraction 0` (the default for `main.py`).

Generate text from a trained checkpoint:

```bash
python other/inference.py --model raw-data/artifacts/model.pt \
  --prompt "Once upon a time" --max-new-tokens 100
```

The script loads the model architecture from the checkpoint and defaults to
`tokenizer.json` beside the model. Use `--tokenizer PATH` if it is elsewhere;
it must be the tokenizer used during training. Omit `--prompt` to enter it
interactively. Output includes the prompt followed by the generated text.
Use `--seed 42` for repeatable sampling and `--device cpu` to select the CPU.

# Other

Not sure about what data to train on? Under other/ there is a script that downloads
the top 10 books in text format from Project Gutenberg. If you use the flag --top50 it
instead downloads the top 50 books. That can serve as a dataset if you don't know 
what to train on. 

## English Wikipedia dataset

English Wikipedia is a useful starting point for factual English prose. It teaches
an encyclopedic style; pretraining alone does not teach chat or instruction following.
The new preparation script selects the cleaned `20231101.en` snapshot from
[Wikimedia's Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia).
Start with the default 256 MiB subset before scaling up:

```bash
python -m pip install datasets
python other/prepare_wikipedia_dataset.py
bash wikipedia-downloaded-data/train.sh
```

The script downloads from the internet and never reads `fullEnglish/` or other
local source folders. Output defaults to `wikipedia-downloaded-data`; use
`--output PATH` to choose another destination.
The download streams with a seeded, bounded shuffle. Increase `--max-mib 1024`
for a larger corpus, or use `--max-files 1000` to limit accepted articles.
Preparation never starts training and refuses to overwrite a nonempty destination.
The generated launcher uses separate tokenizer/model paths and holds out 5% of
files for validation. The corpus config accepts only `.txt`; metadata and training
artifacts are stored outside the corpus.

The script filters recognizable code blocks, code-like paragraphs, markup and
short fragments, and removes exact cleaned-file duplicates. English Wikipedia can
still contain foreign quotations and snippets that these heuristics miss. No source
repositories or books are mixed in. Article URLs and file hashes are recorded in
`metadata/files.jsonl`. See the
dataset card for its CC BY-SA/GFDL licensing information.

## Books and source-code dataset (previous example)

To prepare a complete book and code corpus (20 Rust repositories, five C/C++
repositories, and Linux), run:

```bash
python3 other/prepare_books_and_code_dataset.py
```

The script prompts for a destination folder, downloads or reuses sources, filters
and deduplicates the text, and records source revisions and licenses. Use
`--fresh-downloads` to download every source instead of automatically reusing
local books and Linux. The resulting folder contains its own README and a
`train.sh` launcher. Preparation requires Python 3.10+ and Git; model training
requires the dependencies above.

This is the original dataset builder, renamed from `prepare_example_dataset.py`.
Its books, repository list and preparation behavior are preserved.

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
