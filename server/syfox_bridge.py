"""SyFox ctypes bridge — Python tool-side access to the SI substrate core.

Exposes the full v1.0 -> v3.0 decide-side surface the CLI has:
  - decide with per-call options (M3 evidence JSON, M1 energy-norm, SI modes)
  - script detection for --lang auto routing (v2.2)
  - trigram lane policy toggle (v2.2)
  - fabric info (nodes/lanes/evidence records)
The Python layer is a TOOL: HTTP/JSON marshaling only. Every decision is made
by the C++ SI substrate core (energy injection -> settle -> resonance readout
-> honest silence). No transformer, no classifier lives here.
"""
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
_LIB.syfox_decide_ex.restype = ctypes.POINTER(ctypes.c_char)
_LIB.syfox_decide_ex.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                 ctypes.c_char_p]
_LIB.syfox_engine_info.restype = ctypes.POINTER(ctypes.c_char)
_LIB.syfox_engine_info.argtypes = [ctypes.c_void_p]
_LIB.syfox_detect_script.restype = ctypes.POINTER(ctypes.c_char)
_LIB.syfox_detect_script.argtypes = [ctypes.c_char_p]
_LIB.syfox_engine_set_energy_norm.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_engine_set_source_modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LIB.syfox_engine_set_parallel_settle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LIB.syfox_engine_set_ngrams.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_engine_set_semantics.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_engine_set_retrieval.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_engine_set_retrieval_topk.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_engine_set_retrieval_dose.argtypes = [ctypes.c_void_p, ctypes.c_double]
_LIB.syfox_engine_set_hierarchy.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.syfox_core_version.restype = ctypes.c_char_p
_LIB.syfox_string_free.argtypes = [ctypes.POINTER(ctypes.c_char)]


def core_version() -> str:
    """Version string of the loaded core library (e.g. '3.0.0')."""
    return _LIB.syfox_core_version().decode()


def detect_script(text: str) -> str:
    """Unicode script-family slug for text ('latin', 'bengali', ..., 'unknown')."""
    raw = _LIB.syfox_detect_script(text.encode())
    if not raw:
        return "unknown"
    slug = ctypes.cast(raw, ctypes.c_char_p).value.decode()
    _LIB.syfox_string_free(raw)
    return slug


class SyFoxEngine:
    """One engine instance = one loaded SI substrate model."""

    def __init__(self, model_dir: str):
        self._h = _LIB.syfox_engine_create(model_dir.encode())
        if not self._h:
            raise RuntimeError(f"syfox: cannot load model from {model_dir}")
        self.model_dir = model_dir
        self._lock = threading.Lock()

    def decide(self, state: str, questions: dict, opts: dict | None = None):
        """Decide via the substrate. opts (all optional, CLI-equal semantics):
            evidence        bool   -> M3 machine-auditable evidence JSON
            energy_norm     bool   -> M1 measurement gain
            salience_gating bool   -> SI salience source mode
            miller_window   bool   -> SI Miller [cap-4,cap] window
            ngrams          "on"|"off" -> v2.2 trigram lane policy
            semantics       bool   -> v3.2 semantic field (resonance + ctx lanes)
            retrieval       bool   -> v3.2 retrieval-by-default priming
            retrieval_topk  int    -> v3.2 memories to prime with (default 5)
            retrieval_dose  float  -> v3.2 prime dose x inject (default 0.30)
            hierarchy       bool   -> v3.2 Stage-3 category gating
            question_gate   bool   -> v3.3 question-conditioned readout (opt-in,
                                       default off; measured trade-off: fixes
                                       reasoning probes, hurts tickets-cal 0.9533->0.9000)
            question_gate_floor float -> v3.3 floor for gated candidates (0..1, default 0.25)
            defer_margin    float  -> v3.4 near-tie defer margin (default 0.05 engine-side)
            no_defer        bool   -> v3.4 disable the near-tie defer
            hops            int    -> v3.4 multi-hop readout walk depth (1 = legacy, default)
            ctx_gate        bool   -> v3.4 question-context two-stage settle (opt-in)
            ctx_alpha       float  -> v3.4 context weight in the composed field (0..1, default 0.5)
        Returns (answers, usage) plus "evidence" key when requested.
        usage carries "retrieval": [{label, resonance}] when priming fired.
        """
        payload = json.dumps(questions, ensure_ascii=False).encode()
        with self._lock:
            if opts:
                raw = _LIB.syfox_decide_ex(self._h, state.encode(), payload,
                                           json.dumps(opts).encode())
            else:
                raw = _LIB.syfox_decide(self._h, state.encode(), payload)
            if not raw:
                raise RuntimeError("syfox: decide failed")
            data = json.loads(ctypes.cast(raw, ctypes.c_char_p).value.decode())
            _LIB.syfox_string_free(raw)
        if "error" in data:
            raise RuntimeError(f"syfox: {data['error']}")
        return data.get("answers", {}), data.get("usage", {}), \
            data.get("evidence") if "evidence" in data else None

    # -- engine-scoped knobs (sticky; same flags as the CLI) ----------------

    def set_energy_norm(self, on: bool):
        _LIB.syfox_engine_set_energy_norm(self._h, int(on))

    def set_source_modes(self, salience_gating: bool, miller_window: bool):
        _LIB.syfox_engine_set_source_modes(self._h, int(salience_gating), int(miller_window))

    def set_parallel_settle(self, on: bool, threads: int = 0):
        _LIB.syfox_engine_set_parallel_settle(self._h, int(on), int(threads))

    def set_ngrams(self, on: bool):
        _LIB.syfox_engine_set_ngrams(self._h, int(on))

    # -- v3.2 semantic layer / retrieval / hierarchy ------------------------

    def set_semantics(self, on: bool):
        """Runtime kill switch for the semantic field (v3.2)."""
        _LIB.syfox_engine_set_semantics(self._h, int(on))

    def set_retrieval(self, on: bool):
        """Toggle retrieval-by-default priming (v3.2; inert without memories)."""
        _LIB.syfox_engine_set_retrieval(self._h, int(on))

    def set_retrieval_topk(self, topk: int):
        _LIB.syfox_engine_set_retrieval_topk(self._h, int(topk))

    def set_retrieval_dose(self, dose: float):
        _LIB.syfox_engine_set_retrieval_dose(self._h, float(dose))

    def set_hierarchy(self, on: bool):
        _LIB.syfox_engine_set_hierarchy(self._h, int(on))

    def info(self) -> dict:
        raw = _LIB.syfox_engine_info(self._h)
        if not raw:
            raise RuntimeError("syfox: engine info failed")
        data = json.loads(ctypes.cast(raw, ctypes.c_char_p).value.decode())
        _LIB.syfox_string_free(raw)
        return data

    def __del__(self):
        if getattr(self, "_h", None):
            _LIB.syfox_engine_free(self._h)
