from document_ocr_benchmarks.normalization import (
    normalize_date,
    normalize_field,
    normalize_name,
    normalize_number,
    validate_nin,
    validate_passport_number,
)


def test_normalize_name_collapses_and_uppercases():
    assert normalize_name("  Ada   Okafor ") == "ADA OKAFOR"


def test_normalize_date_formats_to_iso():
    assert normalize_date("14/03/1990") == "1990-03-14"
    assert normalize_date("14 Mar 1990") == "1990-03-14"
    assert normalize_date("1990-03-14") == "1990-03-14"


def test_normalize_number_keeps_passport_prefix():
    assert normalize_number("A 0123 4567") == "A01234567"
    assert normalize_number("123-456-789 01") == "12345678901"


def test_normalize_field_dispatch():
    assert normalize_field("nin", "123 456 789 01") == "12345678901"
    assert normalize_field("gender", "m") == "MALE"


def test_validators():
    assert validate_nin("12345678901")[0] is True
    assert validate_nin("123")[0] is False
    assert validate_passport_number("A01234567")[0] is True
    assert validate_passport_number("12345")[0] is False
