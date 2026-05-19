from __future__ import annotations

try:
    from adjustment.class_maps import CODE_TO_LABEL, decode as _decode
except Exception:
    CODE_TO_LABEL = {}

    def _decode(key: str, code: str) -> str | None:
        return None


def decode(key: str, value: str) -> str:
    return _decode(key, str(value)) or str(value)

