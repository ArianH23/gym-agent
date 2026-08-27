def strip_code_fences(text):
    if text.startswith("```"):
        return text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text