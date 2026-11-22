"""Shared test setup, picked up by pytest for every test in this folder.

No test may spend API credit. Each test file stubs the parts it uses, but a
new code path that a test forgets to stub would otherwise reach the real API
quietly - which happened once, when the follow-up question was added. This
makes any such call fail loudly, before a request leaves the machine.
"""

import anthropic._base_client as base
import pytest


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError(
            "a test tried to call the Anthropic API - stub the path it took")

    monkeypatch.setattr(base.SyncAPIClient, "request", refuse)
