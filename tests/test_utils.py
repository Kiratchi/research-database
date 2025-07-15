# tests/test_utils.py
from es_tools.utils import validate_year_range


def test_year_range():
    assert validate_year_range({"gte": 2020, "lte": 2024})
    assert not validate_year_range({"gte": 2025, "lte": 1900})
