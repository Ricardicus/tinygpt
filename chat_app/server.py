#!/usr/bin/env python3
"""Local browser interface for a TinyGPT checkpoint."""
import argparse
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
        instruction = (f'Previous conversation:\n{conversation}\n\n'
                       f'Now, with all that, here is the instruction:\n{instruction}')
    return f'### Instruction\n{instruction}\n\n### Response\n'


class ChatServer(ThreadingHTTPServer):
    def __init__(self, address, model_path, tokenizer_path, database, device):
        super().__init__(address, Handler)
        self.database = database
        self.lock = threading.Lock()
        self.device = device
        self.response_end_marker = None
        self.tokenizer = BPE()
        self.tokenizer.load(str(tokenizer_path))
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.response_end_marker = checkpoint.get('response_end_marker')
        config = checkpoint['config']
        if (not self.tokenizer.preserve_whitespace or not self.tokenizer.vocab
                or self.tokenizer.vocab_size != config['vocab_size']
                or min(self.tokenizer.vocab.values()) < 0
                or max(self.tokenizer.vocab.values()) >= config['vocab_size']):
            raise ValueError('A matching whitespace-preserving tokenizer is required')
        self.model = TinyGPT(qk_norm=config.get('qk_norm', False),
                             **{key: config.get(key, value) for key, value in MOE_DEFAULTS.items()},
                             **{key: config[key] for key in ('vocab_size', 'context_length', 'd_model', 'n_heads', 'num_layers')})
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device).eval()
        self.info = {'model': str(model_path), 'tokenizer': str(tokenizer_path), 'device': str(device),
                     'config': config, 'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('global_step'),
                     'parameters': sum(parameter.numel() for parameter in self.model.parameters()),
                     'response_end_marker': self.response_end_marker}
        with self.connect() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);'
                             'CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, role TEXT NOT NULL, content TEXT NOT NULL);')

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.execute('PRAGMA foreign_keys = ON')
        return db

    def generate_stream(self, messages, max_tokens, temperature, top_k):
        prompt = format_prompt(messages)
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError('Prompt contains no encodable tokens')
        generated = []
        emitted = 0
        with self.lock, torch.inference_mode():
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
                    for chunk in self.server.generate_stream(messages + [('user', content)], max_tokens, temperature, top_k):
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
            answer = self.server.generate(messages + [('user', content)], max_tokens, temperature, top_k)
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
    parser.add_argument('--model', type=Path, default=ROOT / 'dolly15k/artifacts/dolly-sft.pt')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts-2026-11-02/tokenizer.json')
    parser.add_argument('--database', type=Path, default=Path(__file__).parent / 'chats.sqlite3')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    server = ChatServer(('127.0.0.1', args.port), args.model, args.tokenizer, args.database, args.device)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
