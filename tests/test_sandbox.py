import pytest

from agent.sandbox import run_sandboxed


def test_legitimate_pandas_style_code_still_works():
    local_vars = {"numbers": [10, 20, 30]}
    code = "result = round(sum(numbers) / len(numbers), 2)"
    run_sandboxed(code, local_vars)
    assert local_vars["result"] == 20.0


def test_import_is_blocked():
    local_vars = {}
    code = "import os\nresult = os.popen('echo pwned').read()"
    with pytest.raises(ImportError):
        run_sandboxed(code, local_vars)


def test_dunder_import_is_blocked():
    local_vars = {}
    code = "result = __import__('os').popen('echo pwned').read()"
    with pytest.raises(NameError):
        run_sandboxed(code, local_vars)


def test_open_is_not_available():
    local_vars = {}
    code = "result = open('some_file.txt')"
    with pytest.raises(NameError):
        run_sandboxed(code, local_vars)