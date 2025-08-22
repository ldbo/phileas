import unittest
from pathlib import Path

from phileas import mock_instruments


class TestMockInstruments(unittest.TestCase):
    def test_present_encryption(self):
        vectors = [
            (0x0000000000000000, 0x00000000000000000000, 0x5579C1387B228445),
            (0x0000000000000000, 0xFFFFFFFFFFFFFFFFFFFF, 0xE72C46C0F5945049),
            (0xFFFFFFFFFFFFFFFF, 0x00000000000000000000, 0xA112FFC72F68417B),
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFFFFFF, 0x3333DCD3213210D2),
        ]

        for plaintext, key, true_ciphertext in vectors:
            present = mock_instruments.SimulatedPRESENTImplementation(
                Path("/dev/ttyTest"), 115200
            )
            present.key = key
            our_ciphertext = present.encrypt(plaintext)
            print(f"true: {true_ciphertext:b}\nours: {our_ciphertext:b}")
            self.assertEqual(our_ciphertext, true_ciphertext)
