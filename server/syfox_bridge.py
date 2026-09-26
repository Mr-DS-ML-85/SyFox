"""SyFox ctypes bridge — Python tool-side access to the SI substrate core."""
import ctypes
import json
import os
import threading

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "build", "libsyfox_core.so"))
_LIB.syfox_engine_create.restype = ctypes.c_void_p
_LIB.syfox_engine_create.argtypes = [ctypes.c_char_p]
_LIB.syfox_engine_free.argtypes = [ctypes.c_void_p]
_LIB.syfox_decide.restype = ctypes.POINTER(ctypes.c_char)
_LIB.syfox_decide.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
_LIB.syfox_string_free.argtypes = [ctypes.POINTER(ctypes.c_char)]


class SyFoxEngine:
    """One engine instance = one loaded SI substrate model."""

    def __init__(self, model_dir: str):
        self._h = _LIB.syfox_engine_create(model_dir.encode())
        if not self._h:
            raise RuntimeError(f"syfox: cannot load model from {model_dir}")
        self._lock = threading.Lock()

    def decide(self, state: str, questions: dict):
        payload = json.dumps(questions, ensure_ascii=False).encode()
        with self._lock:
            raw = _LIB.syfox_decide(self._h, state.encode(), payload)
            if not raw:
                raise RuntimeError("syfox: decide failed")
            data = json.loads(ctypes.cast(raw, ctypes.c_char_p).value.decode())
            _LIB.syfox_string_free(raw)
        return data.get("answers", {}), data.get("usage", {})

    def __del__(self):
        if getattr(self, "_h", None):
            _LIB.syfox_engine_free(self._h)
