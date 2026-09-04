safe_builtins = {
    "len": len,
    "sum": sum,
    "round": round,
    "abs": abs,
    "max": max,
    "min": min,
    "str": str,
    "int": int,
    "float": float,
}

def run_sandboxed(code, local_vars):
    exec(code, {"__builtins__": safe_builtins}, local_vars)
