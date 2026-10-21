"""Classify tests that need omitted execution archives, not mathematical tests."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_fedbcert_rg1_runner.py":
            item.add_marker(pytest.mark.archive)
