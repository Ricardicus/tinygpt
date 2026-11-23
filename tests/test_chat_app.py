import json
from contextlib import closing
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from bpe import BPE
from chat_app.server import ChatServer, format_prompt
from tinygpt import TinyGPT


class ChatAppTests(unittest.TestCase):
    def test_interrupted_generation_records_partial_output(self):
        server = object.__new__(ChatServer)
        server.log_lock = threading.Lock()
        with tempfile.TemporaryDirectory() as directory:
            server.log_io = Path(directory) / 'log.jsonl'
            def generate(messages, max_tokens, temperature, top_k, metadata):
                metadata['model'] = 'model.pt'
                yield 'partial'
                yield 'rest'
            server.generate_stream = generate
            with closing(server.recorded_generate(1, [('user', 'Hi')], 10, 0, 0)) as stream:
                self.assertEqual(next(stream), 'partial')
            record = json.loads(server.log_io.read_text())
            self.assertFalse(record['complete'])
            self.assertEqual(record['response'], 'partial')
            self.assertEqual(record['model'], 'model.pt')

    def test_previous_turns_are_context_before_latest_instruction(self):
        self.assertEqual(format_prompt([('user', 'Hello')]),
                         '### Instruction\nHello\n\n### Response\n')
        self.assertEqual(format_prompt([('user', 'First'), ('assistant', 'Answer'), ('user', 'Next')]),
                         '### Context\nUser: First\n\nAssistant: Answer\n\n'
                         '### Instruction\nNext\n\n### Response\n')

    def test_generation_stops_at_checkpoint_marker(self):
        class Tokenizer:
            def encode(self, text):
                return [0]

            def decode(self, tokens):
                return ''.join({1: 'answer', 2: '<end>', 3: 'extra'}[token] for token in tokens)

        class Model:
            context_length = 16

            def __call__(self, inputs):
                index = min(inputs.shape[1], 3)
                logits = torch.zeros((1, inputs.shape[1], 4))
                logits[0, -1, index] = 1
                return logits

        server = object.__new__(ChatServer)
        server.tokenizer = Tokenizer()
        server.model = Model()
        server.device = 'cpu'
        server.use_kv_cache = True
        server.lock = threading.Lock()
        server.response_end_marker = '<end>\n'
        self.assertEqual(server.generate([('user', 'hi')], 4, 0, 0), 'answer')

    def test_chat_persists_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = BPE(vocab_size=64, lower_case=False)
            tokenizer.add_corpus('### Instruction\nHello\n\n### Response\nHello world!\n\n### Context\nUser: Hello\n\nAssistant: Hello')
            tokenizer.train()
            tokenizer.save(str(root / 'tokenizer.json'))
            config = dict(vocab_size=64, context_length=16, d_model=8, n_heads=2, num_layers=1)
            model = TinyGPT(**config)
            torch.save({'config': config, 'model_state_dict': model.state_dict(), 'epoch': 1}, root / 'model.pt')
            torch.save({'config': config, 'model_state_dict': model.state_dict(), 'epoch': 2, 'response_end_marker': '<end>'}, root / 'second.pt')
            bad_config = {**config, 'vocab_size': 32}
            torch.save({'config': bad_config, 'model_state_dict': {}}, root / 'bad.pt')
            log_path = root / 'generation.jsonl'
            server = ChatServer(('127.0.0.1', 0), root / 'model.pt', root / 'tokenizer.json', root / 'chats.db', 'cpu', log_io=log_path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'

            def request(path, data=None):
                body = json.dumps(data).encode() if data is not None else None
                with urlopen(Request(base + path, data=body, headers={'Content-Type': 'application/json'})) as response:
                    return json.load(response)

            try:
                self.assertEqual(request('/api/info')['epoch'], 1)
                self.assertEqual(request('/api/models')['models'], ['bad.pt', 'model.pt', 'second.pt'])
                self.assertEqual(request('/api/models/select', {'model': 'second.pt'})['epoch'], 2)
                self.assertEqual(request('/api/info')['response_end_marker'], '<end>')
                self.assertEqual(request('/api/models')['current'], 'second.pt')
                for name in ('../outside.pt', 'bad.pt'):
                    with self.assertRaises(HTTPError) as error:
                        request('/api/models/select', {'model': name})
                    self.assertEqual(error.exception.code, 400)
                self.assertEqual(request('/api/info')['epoch'], 2)
                chat = request('/api/chats', {})['id']
                body = json.dumps({'chat_id': chat, 'content': 'Hello', 'max_tokens': 2, 'temperature': 0}).encode()
                with urlopen(Request(base + '/api/message/stream', data=body, headers={'Content-Type': 'application/json'})) as response:
                    events = [json.loads(line) for line in response]
                self.assertTrue(events[-1]['done'])
                answer = ''.join(event.get('content', '') for event in events)
                messages = request(f'/api/chats/{chat}')
                self.assertEqual([message['role'] for message in messages], ['user', 'assistant'])
                self.assertEqual(messages[1]['content'], answer)
                records = [json.loads(line) for line in log_path.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['prompt'], '### Instruction\nHello\n\n### Response\n')
                self.assertEqual(records[0]['response'], answer)
                self.assertTrue(records[0]['complete'])
                self.assertEqual(records[0]['model'], str(root / 'second.pt'))
                self.assertGreater(records[0]['prompt_tokens'], 0)
                self.assertIn('initial_model_input', records[0])
                self.assertEqual(request('/api/chats')[0]['title'], 'Hello')
                request('/api/models/select', {'model': 'model.pt'})
                self.assertEqual(request(f'/api/chats/{chat}'), messages)
                second = request('/api/message', {'chat_id': chat, 'content': 'Hello', 'max_tokens': 1, 'temperature': 0})
                records = [json.loads(line) for line in log_path.read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertEqual(records[-1]['model'], str(root / 'model.pt'))
                self.assertEqual(records[-1]['response'], second['content'])
                self.assertIn('### Context\nUser: Hello', records[-1]['prompt'])
                self.assertGreater(records[-1]['prompt_tokens'], records[-1]['context_length'])
                other = request('/api/chats', {})['id']
                with urlopen(Request(base + f'/api/chats/{chat}', method='DELETE')) as response:
                    self.assertTrue(json.load(response)['deleted'])
                self.assertEqual([item['id'] for item in request('/api/chats')], [other])
                with server.connect() as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM messages WHERE chat_id = ?', (chat,)).fetchone()[0], 0)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + f'/api/chats/{chat}', method='DELETE'))
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
