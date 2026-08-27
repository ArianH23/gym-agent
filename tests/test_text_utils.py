from agent.text_utils import strip_code_fences


def test_plain_text_with_no_fence():
    result = strip_code_fences("result = df.mean()")
    assert result == "result = df.mean()"


def test_multiline_text_with_no_fence():
    text = "some_variable = 5\nresult = some_variable + 1"
    result = strip_code_fences(text)
    assert result == text

def test_plain_fence_is_stripped():
    raw = "```\nresult = df.mean()\n```"
    result = strip_code_fences(raw)
    assert result == "result = df.mean()"


def test_python_tagged_fence_is_stripped():
    raw = "```python\nresult = df.mean()\n```"
    result = strip_code_fences(raw)
    assert result == "result = df.mean()"


def test_json_tagged_fence_is_stripped():
    raw = '```json\n{"Age": 30}\n```'
    result = strip_code_fences(raw)
    assert result == '{"Age": 30}'
