# TinyGPT local chat

After the fine-tuning checkpoint has been saved, from the repository root run:

```bash
venv/bin/python chat_app/server.py
```

Open http://127.0.0.1:8000. The default checkpoint is
`dolly15k/artifacts/dolly-sft.pt` and the default tokenizer is
`artifacts-2026-11-02/tokenizer.json`. Select different files with `--model`
and `--tokenizer`; use the tokenizer used to train the checkpoint. Choose a
device with `--device cpu` or `--device cuda`, and change the port with `--port`.
To choose among checkpoints, put `.pt` files in one directory and run:

```bash
venv/bin/python chat_app/server.py --model-dir path/to/checkpoints --tokenizer path/to/tokenizer.json
```

The first filename alphabetically loads at startup, or pass `--model` with a
checkpoint inside that directory to choose the initial model. Select a model in
the chat header; use the refresh button to discover files added later. To reload
an updated checkpoint, select another model and switch back, or restart the server.
Without `--model-dir`, the initial model's folder
is used. All selectable models must use the exact same tokenizer supplied at
startup. Equal vocabulary sizes alone do not establish tokenizer compatibility.
Switching may temporarily require memory for both models; a failed load leaves
the previous model active. Switching affects all clients connected to the server.

Chats are stored locally in `chat_app/chats.sqlite3`. Use `--database` to select
another location. Use the delete button beside a conversation in the sidebar to
remove it and its messages. The server listens only on localhost and has no authentication;
it is intended for use on a trusted local machine, not exposure to the network.
It loads one model and serializes generation requests. Sampling settings are in
the right sidebar. Responses stream as they are generated; a completed exchange
is saved to history after generation finishes. Generation stops at the checkpoint's response end marker,
if present, or at the requested token limit. Trailing line breaks in the marker
are not required for stopping. Older
conversation context is truncated to fit the model's context window. Dolly was
trained on instruction/response examples rather than multi-turn conversations,
so conversational behavior may be limited. Previous messages are placed inside
the instruction as a labelled conversation, followed by the latest instruction.
