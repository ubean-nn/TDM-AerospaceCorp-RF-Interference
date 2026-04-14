"""unit tests for transforms"""

import unittest

import torch

from radiomana.transforms import LogNoise


class TestCustomTransforms(unittest.TestCase):
    def test_log_noise(self):
        """is LogNoise custom transform working as expected?"""
        transform = LogNoise(noise_power_db=-90, p=0.5)
        # Add your test cases here
        batch = (torch.randn(16, 512, 243), torch.randint(0, 9, (16,)))
        output = transform(batch)

        self.assertEqual(batch[0].shape, output[0].shape)
        # ensure labels are unchanged
        self.assertTrue(torch.equal(batch[1], output[1]))
        # ensure no nan values in output
        self.assertFalse(torch.isnan(output[0]).any())
