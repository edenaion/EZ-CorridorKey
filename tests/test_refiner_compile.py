"""Tests for selective compile support in the tiled CNN refiner."""

import pytest
import torch

timm = pytest.importorskip("timm", reason="timm not installed")

from CorridorKeyModule.core.model_transformer import CNNRefinerModule


class TestCNNRefinerModule:
    def test_single_tile_matches_full_frame(self):
        """When input fits in one tile, output should be identical to full-frame."""
        torch.manual_seed(0)
        refiner = CNNRefinerModule().eval()

        img = torch.randn(1, 3, 64, 64)
        coarse_pred = torch.randn(1, 4, 64, 64)

        with torch.no_grad():
            expected = refiner(img, coarse_pred)

            refiner._tile_size = 128  # larger than input — single tile path
            refiner._tile_overlap = 80
            actual = refiner(img, coarse_pred)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_tiled_matches_full_frame(self):
        """Multi-tile output should match full-frame when overlap covers receptive field."""
        torch.manual_seed(0)
        refiner = CNNRefinerModule().eval()

        img = torch.randn(1, 3, 256, 256)
        coarse_pred = torch.randn(1, 4, 256, 256)

        with torch.no_grad():
            expected = refiner(img, coarse_pred)

            refiner._tile_size = 128
            refiner._tile_overlap = 80  # > 65px receptive field
            actual = refiner(img, coarse_pred)

        assert actual.shape == expected.shape
        # Tiled blending is an approximation — overlap ramps produce small
        # differences at tile boundaries (<1% of value range).
        assert torch.allclose(actual, expected, atol=0.15, rtol=0.02)

    def test_compile_tile_kernel_compiles_once_and_preserves_output(self, monkeypatch):
        refiner = CNNRefinerModule().eval()
        x = torch.randn(1, 7, 64, 64)
        expected = refiner._process_tile_impl(x)

        compile_calls = []

        def fake_compile(fn, **kwargs):
            compile_calls.append(kwargs)

            def wrapped(tile):
                return fn(tile)

            return wrapped

        monkeypatch.setattr(torch, "compile", fake_compile)

        refiner.compile_tile_kernel()
        refiner.compile_tile_kernel()
        actual = refiner._process_tile(x)

        assert len(compile_calls) == 1
        assert compile_calls[0]["dynamic"] is False
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
