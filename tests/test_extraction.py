import numpy as np
import pytest
import torch

from angionet.functional import combine_patches, extract_patches


class TestExtractPatches:
    @pytest.mark.parametrize(
        ["dim", "stride", "expected"],
        [([32, 32, (32, 32)]), ([16, 10, (16, 16)]), (128, 1, (128, 128))],
    )
    def test_patch_size(self, dim, stride, expected):
        img = torch.empty((1, 1, 128, 128))
        patches = extract_patches(img, dim, stride)

        assert np.array_equal(list(patches.shape[-2:]), expected)

    @pytest.mark.parametrize(
        ["height", "width", "dim", "stride", "expected"],
        [([10, 10, 5, 5, 4]), ([10, 10, 5, 3, 9])],
    )
    def test_patch_count(self, height, width, dim, stride, expected):
        img = torch.empty((1, 1, height, width))
        patches = extract_patches(img, dim, stride)

        assert patches.size(1) == expected


class TestCombinePatches:
    @pytest.mark.parametrize(
        ["channels", "height", "width", "dim", "stride"],
        [
            (1, 128, 128, 32, 28),
            (1, 128, 128, 128, 128),
            (1, 97, 127, 32, 32),
            (1, 143, 99, 33, 18),
            (2, 128, 128, 32, 28),
            (2, 128, 128, 128, 128),
            (2, 97, 127, 32, 32),
            (2, 143, 99, 33, 18),
        ],
    )
    def test_output_same_as_input(self, channels, height, width, dim, stride):
        orig = torch.ones((1, channels, height, width))
        patches = extract_patches(orig, dim, stride)
        rec = combine_patches(orig.shape, patches, dim, stride)

        assert (orig == rec).all()
