from document_ocr_benchmarks.providers.vlm_openai import _extract_json


def test_extract_json_plain():
    assert _extract_json('{"documentType": "nin_slip", "confidence": 0.9}') == {
        "documentType": "nin_slip",
        "confidence": 0.9,
    }


def test_extract_json_fenced():
    text = "Here you go:\n```json\n{\"a\": 1}\n```\nThanks"
    assert _extract_json(text) == {"a": 1}


def test_extract_json_embedded():
    text = 'prefix {"full_name": {"value": "Ada", "confidence": 0.8}} suffix'
    parsed = _extract_json(text)
    assert parsed["full_name"]["value"] == "Ada"


def test_extract_json_garbage_returns_none():
    assert _extract_json("no json here") is None
    assert _extract_json("") is None
