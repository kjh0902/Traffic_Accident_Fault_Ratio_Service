from __future__ import annotations

from .schemas import AdjustmentResult, BaseRatioResult, Evidence, SceneAnchor


def generate_report(anchor: SceneAnchor, base: BaseRatioResult, adjusted: AdjustmentResult, evidence: Evidence) -> str:
    return f"""## 분석 결과

최종 과실비율은 **A {adjusted.ratio_a:.0f}% / B {adjusted.ratio_b:.0f}%** 입니다.

### 사고 상황
- 장소: {anchor.accident_place}
- 장소 특징: {anchor.accident_place_feature}
- A 진행 정보: {anchor.vehicle_a_progress_info}
- B 진행 정보: {anchor.vehicle_b_progress_info}
- 분류 신뢰도: {anchor.confidence:.3f}

### 과실비율 계산
- 기본 과실비율: A {base.ratio_a:.0f}% / B {base.ratio_b:.0f}%
- 보정값: A {adjusted.adjustment_a:+.2f}

### Tracking Evidence
- 진입 순서: {evidence.entry_order}
- 선진입 강도: {evidence.first_entry_strength or "unknown"} ({evidence.first_entry_conf:.2f})
- 충돌 상대 위치: {evidence.collision_relative_position or "unknown"}
"""

