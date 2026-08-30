from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class MVRoMaRequest:
    scope: str
    input_model: Path
    images: Path
    selection: Path
    intrinsics: Path
    run_dir: Path
    runtime_python: Path
    runtime_root: Path
    weight: Path
    continuation_receipt: Path | None = None
    coarse_size: tuple[int, int] = (378, 672)
    target_size: tuple[int, int] = (756, 1344)
    certainty_threshold: float = 0.5
    max_samples_per_group: int = 1024

    def with_target_size(self, value: tuple[int, int]) -> "MVRoMaRequest":
        return replace(self, target_size=value)

    def validate_resolution(self) -> None:
        for name, size in (("coarse", self.coarse_size), ("target", self.target_size)):
            if any(value <= 0 or value % 14 for value in size):
                raise ValueError(f"MV-RoMa {name} dimensions must be positive and divisible by 14")
        coarse_ratio = self.coarse_size[1] / self.coarse_size[0]
        target_ratio = self.target_size[1] / self.target_size[0]
        native_ratio = 2688 / 1512
        if abs(coarse_ratio - native_ratio) > 0.01 or abs(target_ratio - native_ratio) > 0.01:
            raise ValueError("MV-RoMa resolutions must preserve the River aspect ratio")

    def validate_continuation(self) -> None:
        if self.scope != "full":
            return
        if self.continuation_receipt is None:
            raise ValueError("full MV-RoMa requires a targeted continuation receipt")
        payload = json.loads(self.continuation_receipt.read_text(encoding="utf-8"))
        if not bool((payload.get("mapping_gate") or {}).get("passes")):
            raise ValueError("targeted MV-RoMa continuation receipt did not pass")


__all__ = ["MVRoMaRequest"]
