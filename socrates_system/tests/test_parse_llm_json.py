import pytest
from socrates_system.utils.parse_llm_json import parse_llm_json


# ---------------------------------------------------------------------------
# Happy-path: clean JSON
# ---------------------------------------------------------------------------

def test_plain_array():
    assert parse_llm_json('[{"a": 1}]') == [{"a": 1}]

def test_plain_object():
    assert parse_llm_json('{"key": "value"}') == {"key": "value"}

def test_nested():
    assert parse_llm_json('{"claims": [{"text": "x", "confidence": 0.9}]}') == {
        "claims": [{"text": "x", "confidence": 0.9}]
    }


# ---------------------------------------------------------------------------
# Markdown fences
# ---------------------------------------------------------------------------

def test_json_fence():
    assert parse_llm_json('```json\n{"x": 1}\n```') == {"x": 1}

def test_plain_fence():
    assert parse_llm_json('```\n[1, 2]\n```') == [1, 2]

def test_bare_backticks():
    assert parse_llm_json('`{"a": "b"}`') == {"a": "b"}

def test_leading_json_prefix():
    assert parse_llm_json('json\n{"k": "v"}') == {"k": "v"}


# ---------------------------------------------------------------------------
# Prose preamble / postamble
# ---------------------------------------------------------------------------

def test_prose_before_array():
    result = parse_llm_json('Here are the claims:\n[{"claim": "Paris is in France"}]')
    assert result == [{"claim": "Paris is in France"}]

def test_prose_after_object():
    result = parse_llm_json('{"verdict": "TRUE"}\nLet me know if you need more.')
    assert result == {"verdict": "TRUE"}


# ---------------------------------------------------------------------------
# Escape artefacts
# ---------------------------------------------------------------------------

def test_escaped_underscore():
    assert parse_llm_json('{"key\\_name": "val"}') == {"key_name": "val"}

def test_literal_backslash_n():
    result = parse_llm_json('{"text": "hello\\nworld"}')
    assert result["text"] == "hello world"

def test_embedded_newlines_in_prose():
    result = parse_llm_json('Some prose\n{"a": 1}\nmore prose')
    assert result == {"a": 1}


# ---------------------------------------------------------------------------
# Bare-word identifiers
# ---------------------------------------------------------------------------

def test_bare_confidence_level():
    result = parse_llm_json('{"confidence": High, "x": 1}')
    assert result["confidence"] == "High"

def test_bare_enum_value():
    result = parse_llm_json('{"type_hint": EXTERNAL_KNOWLEDGE_REQUIRED, "c": 0.9}')
    assert result["type_hint"] == "EXTERNAL_KNOWLEDGE_REQUIRED"

def test_json_keywords_not_quoted():
    result = parse_llm_json('{"active": true, "deleted": false, "value": null}')
    assert result == {"active": True, "deleted": False, "value": None}


# ---------------------------------------------------------------------------
# Arrays preferred over objects when both present
# ---------------------------------------------------------------------------

def test_array_preferred_over_object():
    # outer prose contains a stray { } pair; the real data is in the array
    text = 'Result: [{"a": 1}, {"a": 2}] (using default config {})'
    result = parse_llm_json(text)
    assert result == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_raises_on_no_json():
    with pytest.raises(ValueError):
        parse_llm_json("This response contains no JSON at all.")

def test_raises_on_empty_string():
    with pytest.raises(ValueError):
        parse_llm_json("")

def test_raises_on_unclosable_json():
    with pytest.raises(ValueError):
        parse_llm_json('{"key": "value"')
