#!/usr/bin/env python3
"""Local browser interface for a TinyGPT checkpoint."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bpe import BPE
from tinygpt import MOE_DEFAULTS, TinyGPT


def format_prompt(messages):
    history = messages[:-1]
    instruction = messages[-1][1]
    if history:
        conversation = '\n\n'.join(
            f"{'User' if role == 'user' else 'Assistant'}: {content}"
            for role, content in history
        )
        return f'### Context\n{conversation}\n\n### Instruction\n{instruction}\n\n### Response\n'
    return f'### Instruction\n{instruction}\n\n### Response\n'


class ChatServer(ThreadingHTTPServer):
    def __init__(self, address, model_path, tokenizer_path, database, device, model_dir=None, log_io=None):
        super().__init__(address, Handler)
        self.database = database
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.log_io = Path(log_io).expanduser() if log_io else None
        if self.log_io:
            with self.log_io.open('a', encoding='utf-8'):
                pass
        self.device = device
        self.model_dir = Path(model_dir or Path(model_path).parent).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.response_end_marker = None
        self.tokenizer = BPE()
        self.tokenizer.load(str(tokenizer_path))
        self.load_model(Path(model_path))
        with self.connect() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);'
                             'CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, role TEXT NOT NULL, content TEXT NOT NULL);')

    def available_models(self):
        return sorted(path.name for path in self.model_dir.glob('*.pt')
                      if path.is_file() and path.resolve().parent == self.model_dir)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if (not self.tokenizer.preserve_whitespace or not self.tokenizer.vocab
                or self.tokenizer.vocab_size != config['vocab_size']
                or min(self.tokenizer.vocab.values()) < 0
                or max(self.tokenizer.vocab.values()) >= config['vocab_size']):
            raise ValueError('A matching whitespace-preserving tokenizer is required')
        model = TinyGPT(qk_norm=config.get('qk_norm', False),
                             **{key: config.get(key, value) for key, value in MOE_DEFAULTS.items()},
                             **{key: config[key] for key in ('vocab_size', 'context_length', 'd_model', 'n_heads', 'num_layers')})
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device).eval()
        info = {'model': str(model_path.resolve()), 'tokenizer': str(self.tokenizer_path), 'device': str(self.device),
                     'config': config, 'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('global_step'),
                     'parameters': sum(parameter.numel() for parameter in model.parameters()),
                     'response_end_marker': checkpoint.get('response_end_marker')}
        self.model = model
        self.response_end_marker = info['response_end_marker']
        self.info = info

    def select_model(self, name):
        if not isinstance(name, str) or name not in self.available_models():
            raise ValueError('Model must be a .pt file in the configured model directory')
        path = self.model_dir / name
        with self.lock:
            self.load_model(path)
            return self.info

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.execute('PRAGMA foreign_keys = ON')
        return db

    def generate_stream(self, messages, max_tokens, temperature, top_k, metadata=None):
        prompt = format_prompt(messages)
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError('Prompt contains no encodable tokens')
        generated = []
        emitted = 0
        with self.lock, torch.inference_mode():
            if metadata is not None:
                metadata.update(model=self.info['model'], tokenizer=self.info['tokenizer'],
                                context_length=self.model.context_length, prompt_tokens=len(ids),
                                initial_model_input=self.tokenizer.decode(ids[-self.model.context_length:]))
            for _ in range(max_tokens):
                inputs = torch.tensor([ids[-self.model.context_length:]], dtype=torch.long, device=self.device)
                logits = self.model(inputs)[0, -1]
                if temperature == 0:
                    token = logits.argmax().item()
                else:
                    logits = logits / temperature
                    if top_k:
                        cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                        logits = logits.masked_fill(logits < cutoff, float('-inf'))
                    token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                ids.append(token)
                generated.append(token)
                text = self.tokenizer.decode(generated)
                if metadata is not None:
                    metadata['raw_response'] = text
                marker = (self.response_end_marker.rstrip('\r\n') or self.response_end_marker) if self.response_end_marker else None
                if marker and marker in text:
                    visible = text.split(marker, 1)[0]
                    if visible[emitted:]:
                        yield visible[emitted:]
                    break
                held = 0
                if marker:
                    held = next((size for size in range(min(len(marker) - 1, len(text)), 0, -1)
                                 if text.endswith(marker[:size])), 0)
                visible = text[:len(text) - held] if held else text
                if visible[emitted:]:
                    yield visible[emitted:]
                    emitted = len(visible)
            else:
                text = self.tokenizer.decode(generated)
                if text[emitted:]:
                    yield text[emitted:]

    def generate(self, messages, max_tokens, temperature, top_k):
        return ''.join(self.generate_stream(messages, max_tokens, temperature, top_k))

    def recorded_generate(self, chat_id, messages, max_tokens, temperature, top_k):
        metadata = {}
        chunks = []
        completed = False
        try:
            with closing(self.generate_stream(messages, max_tokens, temperature, top_k, metadata)) as stream:
                for chunk in stream:
                    chunks.append(chunk)
                    yield chunk
            completed = True
        finally:
            if self.log_io is not None:
                record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'chat_id': chat_id,
                          'prompt': format_prompt(messages), 'response': ''.join(chunks),
                          'complete': completed, 'max_tokens': max_tokens,
                          'temperature': temperature, 'top_k': top_k, **metadata}
                with self.log_lock, self.log_io.open('a', encoding='utf-8') as output:
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/':
            payload = (Path(__file__).parent / 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        with self.server.connect() as db:
            if self.path == '/api/info':
                return self.respond(200, self.server.info)
            if self.path == '/api/models':
                return self.respond(200, {'models': self.server.available_models(), 'current': Path(self.server.info['model']).name})
            if self.path == '/api/chats':
                return self.respond(200, [dict(id=row[0], title=row[1], created=row[2]) for row in db.execute('SELECT id, title, created FROM chats ORDER BY id DESC')])
            if self.path.startswith('/api/chats/'):
                try:
                    chat_id = int(self.path.removeprefix('/api/chats/'))
                except ValueError:
                    return self.respond(404, {'error': 'Not found'})
                if not db.execute('SELECT 1 FROM chats WHERE id = ?', (chat_id,)).fetchone():
                    return self.respond(404, {'error': 'Chat not found'})
                return self.respond(200, [dict(role=row[0], content=row[1]) for row in db.execute('SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id', (chat_id,))])
        self.respond(404, {'error': 'Not found'})

    def do_DELETE(self):
        if not self.path.startswith('/api/chats/'):
            return self.respond(404, {'error': 'Not found'})
        try:
            chat_id = int(self.path.removeprefix('/api/chats/'))
        except ValueError:
            return self.respond(404, {'error': 'Not found'})
        with self.server.connect() as db:
            deleted = db.execute('DELETE FROM chats WHERE id = ?', (chat_id,)).rowcount
        if not deleted:
            return self.respond(404, {'error': 'Chat not found'})
        self.respond(200, {'deleted': True})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 100000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/models/select':
                return self.respond(200, self.server.select_model(data['model']))
            if self.path == '/api/chats':
                with self.server.connect() as db:
                    cursor = db.execute('INSERT INTO chats (title) VALUES (?)', ('New conversation',))
                    return self.respond(201, {'id': cursor.lastrowid})
            if self.path not in ('/api/message', '/api/message/stream'):
                return self.respond(404, {'error': 'Not found'})
            chat_id = int(data['chat_id'])
            content = data['content'].strip()
            max_tokens = int(data.get('max_tokens', 400))
            temperature = float(data.get('temperature', 0.8))
            top_k = int(data.get('top_k', 40))
            if not content or len(content) > 20000 or not 1 <= max_tokens <= 1024 or not math.isfinite(temperature) or not 0 <= temperature <= 3 or not 0 <= top_k <= 10000:
                raise ValueError('Invalid message or sampling settings')
            with self.server.connect() as db:
                if not db.execute('SELECT 1 FROM chats WHERE id = ?', (chat_id,)).fetchone():
                    return self.respond(404, {'error': 'Chat not found'})
                messages = list(db.execute('SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id', (chat_id,)))
            if self.path == '/api/message/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                try:
                    chunks = []
                    with closing(self.server.recorded_generate(chat_id, messages + [('user', content)], max_tokens, temperature, top_k)) as stream:
                        for chunk in stream:
                            chunks.append(chunk)
                            self.wfile.write((json.dumps({'content': chunk}) + '\n').encode())
                            self.wfile.flush()
                    answer = ''.join(chunks)
                    self.save_exchange(chat_id, content, answer, messages)
                    self.wfile.write((json.dumps({'done': True}) + '\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as error:
                    self.wfile.write((json.dumps({'error': str(error)}) + '\n').encode())
                    self.wfile.flush()
                return
            answer = ''.join(self.server.recorded_generate(chat_id, messages + [('user', content)], max_tokens, temperature, top_k))
            self.save_exchange(chat_id, content, answer, messages)
            self.respond(200, {'content': answer})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.respond(400, {'error': str(error)})
        except Exception as error:
            self.respond(500, {'error': str(error)})

    def save_exchange(self, chat_id, content, answer, messages):
        with self.server.connect() as db:
            db.executemany('INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)',
                           [(chat_id, 'user', content), (chat_id, 'assistant', answer)])
            if not messages:
                db.execute('UPDATE chats SET title = ? WHERE id = ?', (content[:70], chat_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Initial checkpoint (default: first model in --model-dir, or the Dolly checkpoint)')
    parser.add_argument('--model-dir', type=Path, help='Directory containing selectable .pt checkpoints')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts-2026-11-02/tokenizer.json')
    parser.add_argument('--database', type=Path, default=Path(__file__).parent / 'chats.sqlite3')
    parser.add_argument('--log-io', type=Path, help='Append prompts and generated responses to this JSONL file')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if model_dir and not model_dir.is_dir():
        parser.error('--model-dir must be an existing directory')
    model = args.model.expanduser().resolve() if args.model else None
    if model_dir and model and model.parent != model_dir:
        parser.error('--model must be inside --model-dir')
    if model is None and model_dir:
        models = sorted(path for path in model_dir.glob('*.pt') if path.is_file() and path.resolve().parent == model_dir)
        if not models:
            parser.error('--model-dir contains no .pt files')
        model = models[0]
    model = model or ROOT / 'dolly15k/artifacts/dolly-sft.pt'
    server = ChatServer(('127.0.0.1', args.port), model, args.tokenizer, args.database, args.device, model_dir, args.log_io)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
