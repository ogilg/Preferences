"""Tests for the model interface."""

import pytest
from typing import Protocol


class TestModelProtocol:
    """Tests that define the Model protocol contract."""

    def test_model_protocol_exists(self):
        """Model protocol should exist."""
        from src.model import Model

        assert Model is not None

    def test_model_is_protocol(self):
        """Model should be a Protocol."""
        from src.model import Model

        assert issubclass(Model, Protocol)

    def test_model_has_generate_method(self):
        """Model should define a generate method."""
        from src.model import Model

        assert hasattr(Model, "generate")

    def test_model_has_get_logprobs_method(self):
        """Model should define a get_logprobs method."""
        from src.model import Model

        assert hasattr(Model, "get_logprobs")
