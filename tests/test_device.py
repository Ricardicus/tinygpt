import unittest
from unittest.mock import patch

from main import default_device, parse_args


class DeviceTests(unittest.TestCase):
    def test_automatic_device_selection(self):
        for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'),
                                    (False, False, 'cpu')]:
            with self.subTest(device=expected):
                with patch('torch.cuda.is_available', return_value=cuda), \
                     patch('torch.backends.mps.is_available', return_value=mps):
                    self.assertEqual(default_device(), expected)
                    self.assertEqual(parse_args([]).device, expected)
                    self.assertEqual(parse_args(['--device', 'cpu']).device, 'cpu')
