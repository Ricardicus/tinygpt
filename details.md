# TinyGPT

TinyGPT is a single-device PyTorch project for training a decoder-only language model on your own text. You choose the corpus, train a byte-pair encoding (BPE) tokenizer, and pretrain a transformer to predict the next token. You can then use the checkpoint to complete text, or fine-tune it on instruction/response pairs and talk to it through a local chat server.

Two files travel together. **`tokenizer.json`** maps text to token IDs. **`model.pt`** holds the weights that give those particular IDs meaning. Keep the exact tokenizer with every model derived from it, including fine-tuned and minified copies. A matching vocabulary size alone cannot establish compatibility.

[First run](#first-run) · [Corpus preparation](#corpus-preparation) · [Tokenizer](#tokenizer) · [Model architecture](#model-architecture) · [Pretraining](#pretraining) · [Monitoring and validation](#monitoring-and-validation) · [Checkpoints and resume](#checkpoints-and-resume) · [Supervised fine-tuning](#supervised-fine-tuning) · [Inference and chat](#inference-and-chat)

## First run

From the repository root, create a Python environment and install the project dependencies. The examples below use Bash.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

To try the whole workflow without a download, use two documents from this checkout as a tiny corpus. This example uses the CPU, a small model, and eager execution so you can check that everything works before spending time on a real run. Keep the same shell for the inference command.

```bash
RUN=$(mktemp -d)
mkdir "$RUN/corpus"
cp README.md details.md "$RUN/corpus/"

python main.py --rawdata "$RUN/corpus" \
  --tokenizer "$RUN/tokenizer.json" --model "$RUN/model.pt" \
  --vocab_size 256 --bpe-part 1 --lower-case false \
  --d-model 64 --n-heads 4 --num-layers 2 --context-length 64 \
  --epochs 1 --device cpu --no-compile-loss \
  --validation-fraction 0.5 --log-progress "$RUN/progress.csv"

python other/inference.py --model "$RUN/model.pt" \
  --prompt "TinyGPT" --max-new-tokens 20 --device cpu
```

The first command learns a tokenizer, holds out one document, trains on the other, and saves the checkpoint. The inference script finds `tokenizer.json` beside it. Expect this small run to exercise the pipeline, with very little data from which to learn useful completions.

For the full option lists, run `python main.py --help`, `python main.py finetune --help`, or `python main.py minify --help`. Each script under `other/` has its own entry point.

## Corpus preparation

### Download a corpus

Run `python other/download_datasets.py` to select sources, limits, and an output directory. Use **Space** to select sources, **E** to edit a limit, **F** for the full source, and **Enter** to review and confirm. **W** sets Wikipedia languages and individual quotas. **L** selects Gutenberg languages. Use `--plain` for a line-based wizard.

For a repeatable multilingual Wikipedia selection, put each language and its quota on the command line. Spanish uses `es` and Swedish uses `sv`.

```bash
python other/download_datasets.py \
  --wikipedia-language en=15 --wikipedia-language es=10 --wikipedia-language sv=10 \
  --output datasets/wiki-mix
```

Limits count saved UTF-8 text in decimal GB. A source can run out before reaching its quota. `full` consumes the available source after filtering and deduplication.

| Source | What goes into the corpus |
| --- | --- |
| Wikipedia | Article text from the pinned November 2023 snapshot, including short entries and lists. `--wikipedia 5` selects up to 5 GB of English. The interactive **W** field also accepts lists such as `en=10,sv=2,es=full`. |
| FineWeb-Edu | Filtered web prose. Select it in the wizard to combine it with Wikipedia or books. |
| Project Gutenberg | Whole books with their notices. The wizard offers multiple catalog languages and one shared quota. The final book can take the download over that target. |
| OpenBallast T2 | OpenStax textbook passages, including short definitions and examples. Use `--ballast-t2 full` or give a GB limit. The manifest records passage IDs and source information. |
| Dolly and Alpaca variants | One formatted instruction/response example per file. These can be mixed or used individually for [supervised fine-tuning](#supervised-fine-tuning). |

The downloader writes training documents under `corpus/` and source revisions, hashes, URLs, and actual sizes under `metadata/`. It drops exact duplicates across selected sources. Keep the metadata for attribution and consult the recorded dataset cards for the terms of each source.

A completed download includes `training_data.json` and a `train.sh` launcher. To continue an interrupted download, run `python other/download_datasets.py --resume datasets/wiki-mix`. Resume verifies the saved manifest, file sizes, and hashes. Keep a downloader-managed corpus intact if you plan to resume it.

### Bring your own files

Point `--rawdata` at the root to scan. Both tokenizer training and pretraining use the same recursive file selection. The repository's [default corpus config](training_data.json) includes text, Markdown, and several source-code suffixes. For a cloned repository, a custom `--data-config` like this one selects prose and C sources.

```json
{
  "suffixes": [".txt", ".md", ".c", ".h"],
  "exclude_dirs": [".git", "venv", "build"],
  "exclude_files": ["drafts/unfinished.md"]
}
```

File exclusions are exact paths relative to the corpus root. `exclude_files_from` accepts a list of filenames containing one exclusion per line. Those list filenames resolve relative to the config. Blank lines and `#` comments are ignored. Readers skip symlinks, decode text as UTF-8 with replacement, reject NUL-containing data, and omit words longer than 4,096 characters. Use `--lower-case false` to preserve case in prose or code.

### Inspect and clean the data

Use the character audit to find files that may need review before training. The default `english` policy accepts Latin letters, combining marks, and common English typography. `--policy ascii` applies a narrower character set. Use the excerpts to judge whether the flagged characters belong in your corpus.

```bash
python other/audit_corpus.py datasets/wiki-mix --output audits/wiki-mix
python other/audit_corpus.py --from-report audits/wiki-mix --exclude-above 1
```

Review `files.csv` for per-file percentages, `flagged-files.jsonl` for excerpts, `characters.csv` for character counts, and `summary.json` for estimated whole-file removal sizes by source. The second command selects files with **more than 1%** outside-policy characters. Edit `exclusions-gt-1/excluded-files.txt` and pass the adjacent `training_data.json` to the trainer to apply that selection before the validation split and tokenizer scan.

To strip Gutenberg front and back matter, run `python other/trim_gutenberg.py --gutenberg-folder path/to/books --dry-run`. On a working copy, omit `--dry-run` to keep the text between the first start marker and its following end marker and remove leading blank lines. The tool skips missing markers and empty bodies. Keep the original notices and attribution with your source copy.

### Other dataset builders

| Script under `other/` | When to use it |
| --- | --- |
| `prepare_wikipedia_dataset.py` | Build a filtered English Wikipedia subset. Defaults to 256 MiB in `wikipedia-downloaded-data`. Use `--max-mib` or `--max-files` to change the size. This builder removes recognizable code, markup, short fragments, and duplicates. |
| `prepare_wikipedia_fineweb_dataset.py` | Combine the verified Wikipedia subset with up to 1 GiB of filtered FineWeb-Edu from `sample-10BT`. Set `--wikipedia-dir`, `--wikipedia-mib`, or `--fineweb-mib` as needed. The launcher in `wikipedia-fineweb-data` defaults to a 512-wide, eight-layer model. |
| `prepare_books_and_code_dataset.py` | Build a corpus of books, Rust and C/C++ repositories, and Linux. It records revisions and licenses and offers `--fresh-downloads`. Requires Git. |
| `gutenberg_top10.py` | Download a small collection from Gutenberg's popular titles. Use `--top50` for a larger collection. |

The Wikipedia/FineWeb builder has its own recovery options. `--resume` verifies the manifest and continues collection. `--finalize-existing` verifies the files and prepares a launcher from the text already collected, even if the download fell short of the target. For an older partial build, `--fineweb-revision COMMIT` supplies its known source revision.

## Tokenizer

BPE starts with individual characters and learns merges for frequent adjacent symbols. The resulting vocabulary determines which strings can be represented by a single token. New tokenizers preserve spaces, tabs, blank lines, and line endings. An existing `--tokenizer` file is loaded with its saved encoding rules. Choose fresh tokenizer **and model** paths when changing the vocabulary.

| Setting | Effect when training a new tokenizer |
| --- | --- |
| `--vocab_size` | Target number of vocabulary entries, including base characters, reserved symbols, and learned merges. |
| `--bpe-part` | Fraction of training text chunks used for merge statistics. Files are shuffled before traversal, then the first fraction of chunks is used. All selected text is scanned for character coverage, including held-out files when validation is enabled. |
| `--bpe-max-words` | Cap on distinct words retained for merge learning. The default is 100,000. This bounds the word table even when the corpus is much larger. |
| `--bpe-alphabet-policy` | `limit` (default) maps rare characters to `<unk>` when the base alphabet would fill the vocabulary. The automatic cap reserves roughly half the vocabulary for merges. `expand` grows the vocabulary on overflow and `error` stops training. |
| `--bpe-max-chars` and `--bpe-keep-chars` | Set a specific base-alphabet cap and guarantee characters within it. Reserved symbols also consume slots. Characters mapped to `<unk>` lose their original spelling. |

For English, Swedish, and Spanish together, one example is `--vocab_size 16384 --lower-case false --bpe-max-chars 256 --bpe-keep-chars 'åäöÅÄÖáéíóúüñ¿¡ÁÉÍÓÚÜÑ'`. Character selection affects new tokenizers only. `--bpe-part` controls the tokenizer sample. The model trains on every complete training window.

## Model architecture

TinyGPT is a causal transformer. It adds learned position embeddings to token embeddings, passes them through a stack of attention and feed-forward blocks, and projects the result back into the vocabulary using the token embedding weights. During pretraining, the target at each position is the following token.

| Option | What it controls |
| --- | --- |
| `--context-length` | Maximum input window, measured in tokenizer tokens. |
| `--d-model`, `--num-layers`, `--n-heads` | Hidden width, depth, and number of attention heads. Width must be divisible by head count. |
| `--qk-norm` | Applies learned RMS normalization to queries and keys within each head. Statistics use FP32 and the result uses the attention dtype. |
| `--moe --num-experts 4 --moe-top-k 2 --moe-aux-loss-weight 0.01` | Replaces each dense feed-forward block with routed experts. Each token goes to its selected experts. The optimizer objective includes a load-balancing term. The reported loss measures token prediction. |

Dense blocks are the default. MoE adds expert parameters and optimizer state even though each token uses only a subset. Expert dispatch runs eagerly on one device. Q/K normalization and MoE are architecture choices for a **fresh model**. On resume and inference, saved checkpoint settings take precedence. The implementation is in [tinygpt.py](tinygpt.py).

## Pretraining

For a prepared dataset, use its launcher. It supplies the corpus root, config, tokenizer and checkpoint paths, and a baseline training configuration. Append options to override those defaults. This example uses the `datasets/wiki-mix` download from above and gives the run its own artifacts directory.

```bash
RUN=runs/wiki-mix
mkdir -p "$RUN"
bash datasets/wiki-mix/train.sh \
  --tokenizer "$RUN/tokenizer.json" --model "$RUN/model.pt" \
  --log-progress "$RUN/progress.csv" \
  --monitor-layer-norms "$RUN/layer-norms.csv" --monitor-layer-norms-every 100 \
  --vocab_size 16384 --lower-case false --bpe-max-chars 256 \
  --bpe-keep-chars 'åäöÅÄÖáéíóúüñ¿¡ÁÉÍÓÚÜÑ' \
  --d-model 768 --n-heads 12 --num-layers 16 --context-length 512 \
  --batch-size 16 --epochs 1 --qk-norm \
  --lr 0.0001 --beta1 0.9 --beta2 0.95 --lr-schedule cosine \
  --validation-fraction 0.05 --validation-every-steps 1000 \
  --validation-batches 50 --save-every-steps 1000
```

For a custom corpus, invoke `python main.py` with `--rawdata`, `--data-config`, and your chosen training options directly. A new tokenizer first counts chunks and learns merges. The reader then packs tokens into windows of `context_length + 1` and uses shifted inputs and targets. Each epoch shuffles all complete training windows and visits each once. At batch 16 and context 512, a full optimizer update has 8,192 target tokens.

The reader caches four-byte token IDs on disk and holds the tokens and shuffle indices in RAM during training. Training and validation have separate caches when a split is enabled. Changes to file selection, paths, sizes, modification times, or tokenizer settings trigger a rebuild. Verbose startup reports scanning, tokenizer preparation, and cache progress.

### Optimizer and learning rate

AdamW defaults to a constant learning rate of `0.0003` and betas of `0.9` and `0.999` on a new run. Set `--beta1` and `--beta2` independently. With `--lr-schedule cosine`, `--lr` is the peak. The rate rises linearly over `--warmup-steps` updates, then decays to `--min-lr-ratio` times the peak (default `0.1`).

The default warmup is 10% of planned updates, capped at 1,000. Set `--warmup-steps 0` to start at the peak. `--lr-decay-steps` sets the total schedule length **including warmup**. Otherwise the trainer derives it from epochs, windows, and batch size. Once reached, the floor holds for subsequent updates.

### Device, precision, and compilation

The trainer automatically selects CUDA, then MPS, then CPU according to availability. Use `--device` to choose explicitly. Pretraining `--precision auto` uses BF16 autocast on CUDA devices with native BF16 support and FP32 elsewhere. `--precision bfloat16` requires that support. `--precision float32` selects FP32 execution. Checkpoint weights and optimizer moments use FP32.

| Pretraining setting | Behavior |
| --- | --- |
| Fused AdamW | Enabled on CUDA. `--no-fused-adamw` selects standard AdamW, including when resuming. |
| Compiled training loss | The training forward pass and cross-entropy compile together by default. `--no-compile-loss` uses the eager loss. |
| `--compile-model` | Also compiles the model for validation. Combined with `--no-compile-loss`, the model compiles and cross-entropy runs eagerly. |
| `--compile-qk-norm` | Compiles Q/K normalization on a QK-normalized model. For Q/K-only pretraining compilation, add `--no-compile-loss` and omit `--compile-model`. |

For fully eager pretraining, use `--no-compile-loss` and omit both optional compilation flags. Compilation needs a working PyTorch/Inductor toolchain. Initial calls and new batch shapes can incur compilation time. Compare steady-state throughput after warmup. To measure Q/K normalization in isolation, run `python other/benchmark_qk_norm.py` on the training GPU.

## Monitoring and validation

Pretraining prints the step, percentage of the current epoch, cumulative mean training loss, and learning rate. The ten-step throughput figure includes transfers, forward pass, backward pass, and optimizer work. Batch preparation, validation, progress CSV writes, and checkpoint writes take additional wall time.

`--validation-fraction 0.05` holds out 5% of **files** before model training. With the default `--validation-seed 42`, each evaluation uses the same sample of held-out windows. Validation loss is the mean cross-entropy over target tokens. Perplexity is `exp(validation_loss)`. Compare checkpoints using the same tokenizer and evaluation sample. These measures track next-token prediction. Use task prompts to assess factual recall or instruction following.

| Output from the pretraining example | What to read |
| --- | --- |
| `progress.csv` | One row per update with batch loss, epoch mean loss, latest validation loss, learning rate used, and gradient norm before clipping. The validation value carries forward between evaluations. |
| `progress.csv.validation.csv` | One row per evaluation with global step, validation loss, perplexity, and token count. |
| `layer-norms.csv` | Per-block parameter and gradient L2 norms. `--monitor-layer-norms-every` controls sampling by global step (default 1 when monitoring is enabled). |
| `layer-norms.csv.parameters.csv` | At the same interval, norms for embeddings, final LayerNorm, and individual parameters in the first three blocks. Measurements precede clipping and the optimizer update. |

Direct `main.py` pretraining defaults to validation disabled. When enabled, it evaluates before the first update, every 1,000 steps, and at epoch end. `--validation-every-steps 0` keeps just the initial and epoch-end checks. `--validation-batches` bounds the cost per check. Enabling a split late can hold out files that a checkpoint has already seen.

```bash
tail -f "$RUN/progress.csv"
```

The plotting script accepts either progress CSV or a norms CSV. These commands save one image for each.

```bash
python other/plot_progress.py --input "$RUN/progress.csv" --output "$RUN/training.png"
python other/plot_progress.py --input "$RUN/progress.csv.validation.csv" --output "$RUN/validation.png"
python other/plot_progress.py --norms "$RUN/layer-norms.csv" \
  --series 0 --series 5 --log-gradients --output "$RUN/norms.png"
```

Omit `--output` for an interactive plot. For a named tensor, pass the `.parameters.csv` file to `--norms` and select, for example, `--series E`. Use separate log paths for separate runs. Compatible older progress headers are upgraded on append, leaving unknown historical values blank.

## Checkpoints and resume

Pretraining saves to `--model` every 1,000 optimizer updates and at each epoch end. `--save-every-steps N` changes the interval. Set it to `0` for epoch-end saves only. Each successful save atomically replaces the previous file. Copy a checkpoint to a separate path when you want to retain a particular step for evaluation.

To continue a run, launch pretraining with the same model, tokenizer, and corpus. A full checkpoint restores weights, architecture, optimizer moments, update counters, and any saved learning-rate schedule. **A mid-epoch checkpoint restarts traversal of that epoch with a fresh shuffle.** Some windows can be seen again. Checkpoint step counters and schedule position continue from the save.

On resume, omitted learning-rate and beta options retain their saved values. An explicit `--lr` overrides the constant rate or the peak of a cosine schedule. Launcher defaults count as explicit arguments. A saved cosine schedule retains its original length even if batch size or epochs change. `--lr-decay-steps` overrides the total from the original schedule start. Enabling cosine on a constant-rate checkpoint starts a schedule at its saved global step. Switching to `--lr-schedule constant` returns to the latest optimizer rate unless you also set `--lr`.

Use `minify` to export a smaller checkpoint for inference or fine-tuning. Choose a new output path.

```bash
python main.py minify --model "$RUN/model.pt" --output "$RUN/model-inference.pt"
```

The export keeps the original weight precision, model config, progress metadata, and saved response marker. It strips the AdamW state and learning-rate schedule. Keep the full checkpoint for continuing the original pretraining run, and the matching tokenizer for both files.

## Supervised fine-tuning

Pretraining learns continuations across the whole text. Supervised fine-tuning (SFT) starts from a pretrained checkpoint and teaches it to produce the response to a given instruction. **Only the answer and its end marker contribute to the loss.** The prompt supplies the context needed to predict them. Use the pretrained model's exact whitespace-preserving tokenizer.

The downloader normalizes Dolly and Alpaca records into one text file per example. Alpaca's `input` becomes optional context, `instruction` becomes the question or task, and `output` becomes the response. Here is a complete example with context.

```text
### Context
The library closes at six on Fridays.

### Instruction
When does the library close on Friday?

### Response
At six.
```

Omit the context section for a standalone instruction. The reader also accepts older instruction-first files and places their context first. Long responses are windowed so each answer token is trained once. The trainer appends `\n\n### End Response\n` and saves the marker in the checkpoint. `--response-end-marker` selects another nonempty string the tokenizer can represent exactly.

Download one of `--alpaca`, `--dolly-alpaca`, `--alpaca-gpt4`, `--alpaca-cleaned`, or `--gpt4-llm-cleaned`, or combine them. The wizard also offers the original Dolly 15k. This example fine-tunes the base model from the pretraining section using two sources.

```bash
python other/download_datasets.py --alpaca-gpt4 full --alpaca-cleaned full \
  --output datasets/instructions

python main.py finetune --data datasets/instructions/corpus \
  --base-model "$RUN/model.pt" --tokenizer "$RUN/tokenizer.json" \
  --output "$RUN/instruct.pt" --epochs 3 --batch-size 2 --lr 0.00002 \
  --validation-every-steps 500 --save-every-steps 500
```

`--data` is scanned recursively, so you can point it at one source subdirectory or a corpus containing only formatted examples. Use `main.py finetune` for response-only training. A dataset's `train.sh` performs ordinary next-token pretraining even when its files contain instructions.

Fine-tuning requires a new output path, creates a fresh AdamW optimizer, holds out 5% of examples by default, and counts training windows up front. Epoch completion and throughput appear every 100 steps and at epoch end. Throughput counts input positions, including prompt and padding. To compile fine-tuning, pass `--compile-model` or, on a QK-normalized checkpoint, `--compile-qk-norm`.

By default, fine-tuning validates and saves at epoch end. The interval flags in the example add checks and saves every 500 optimizer steps. Progress goes to `<output>.progress.csv` and evaluations to `<output>.progress.csv.validation.csv`. `--log-progress` chooses another base log path. Logs append, but step and epoch numbering start over for each new fine-tuning run. An intermediate checkpoint can serve as the base for another run with a fresh optimizer.

## Inference and chat

For a pretrained checkpoint, give `other/inference.py` a text prefix. For an instruction-tuned checkpoint, use the same section headings it saw during fine-tuning. End the prompt immediately after the response heading and newline.

```bash
python other/inference.py --model "$RUN/instruct.pt" --tokenizer "$RUN/tokenizer.json" \
  --prompt $'### Instruction\nExplain gravity in one sentence.\n\n### Response\n' \
  --max-new-tokens 100 --temperature 0.7 --top-k 40 --seed 42 --tokens-per-second
```

`--temperature 0` selects the highest-scoring next token. The defaults (`--temperature 1` and `--top-k 0`) sample across the vocabulary. `--max-new-tokens` caps the continuation. The script stops at a generated checkpoint end marker and hides it by default. Use `--show-end-marker` to display it. `--tokens-per-second` writes elapsed generation time, actual generated token count, and throughput to stderr. Timing includes prompt processing and starts after checkpoint loading.

Both `main.py --prompt` and the standalone inference script support sampling controls. For a simple factual completion check, run `python tests/test_factfulness.py --model "$RUN/model.pt" --tokenizer "$RUN/tokenizer.json"`. It scores a small built-in prompt set. To run the software regression suite, use `python -m unittest discover -s tests -v`.

The inference scripts and chat server use a key/value (KV) cache to store attention keys and values per generation by default. After the initial prompt, the model processes one new token per forward pass while the prefix fits. When the window slides, retained tokens shift absolute positions and their representations are recomputed. Pass `--no-kv-cache` to disable this optimization, including in the factuality runner.

### Local chat server

```bash
python chat_app/server.py --model-dir "$RUN" --model "$RUN/instruct.pt" \
  --tokenizer "$RUN/tokenizer.json"
```

Open `http://127.0.0.1:8000`. The browser streams the answer, shows the active checkpoint, and lets you switch between `.pt` files in the model directory that share the supplied tokenizer. Previous turns go under `### Context`, followed by the newest instruction. Older tokens fall outside the model's window as the conversation grows.

Conversations are stored in SQLite and can be deleted from the sidebar. For a separate record of the formatted input and generated output, start the server with `--log-io path/to/generations.jsonl`. That opt-in log retains its own copy when a conversation is deleted. See the [chat app guide](chat_app/details.md) for model selection, storage paths, and server options.
