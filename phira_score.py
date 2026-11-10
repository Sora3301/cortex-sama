"""phira_score.py —— 按 Phira 的判定与计分逻辑打分。

依据 ~/dev/phira/prpr/src/judge.rs：

判定窗口（秒）：
    Perfect 0.08
    Good    0.16
    Bad     0.22
    另有 0.07 的 EARLY_OFFSET，只作用于打晚的音符。

计分：
    accuracy = (Perfect + Good * 0.65) / 音符总数
    score    = round((0.9 * accuracy + 0.1 * max_combo / 音符总数) * 1_000_000)
    Perfect 等于音符总数时直接给 1_000_000。

Combo：
    Perfect / Good 连击 +1，Bad / Miss 清零；结算取最大连击。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

LIMIT_PERFECT = 0.08
LIMIT_GOOD = 0.16
LIMIT_BAD = 0.22
EARLY_OFFSET = 0.07
GOOD_ACC_WEIGHT = 0.65
TOTAL_SCORE = 1_000_000


class Judgement(IntEnum):
    """顺序与 Phira 的 counts 数组一致。"""

    PERFECT = 0
    GOOD = 1
    BAD = 2
    MISS = 3


def as_judgement(value: Judgement | str | int) -> Judgement:
    if isinstance(value, Judgement):
        return value
    if isinstance(value, str):
        return Judgement[value.upper()]
    return Judgement(value)


def judge_tap(offset: float, speed: float = 1.0) -> Judgement:
    """判定一个 Tap。

    offset = 点击时刻 - 音符时刻（秒），正数表示打晚，负数表示打早。
    speed 对应 Phira 的游戏速度，默认 1.0。

    复刻源码中的点击判定：
        dt = (note.time - touch.time) / speed
        if dt < 0:  # 打晚
            dt = min(dt + EARLY_OFFSET, 0)
        再按 abs(dt) 与 0.08 / 0.16 / 0.22 比较。
    """
    if speed <= 0:
        raise ValueError("speed 必须大于 0")

    dt = -offset / speed
    if dt < 0:
        dt = min(dt + EARLY_OFFSET, 0.0)
    error = abs(dt)

    if error <= LIMIT_PERFECT:
        return Judgement.PERFECT
    if error <= LIMIT_GOOD:
        return Judgement.GOOD
    if error <= LIMIT_BAD:
        return Judgement.BAD
    return Judgement.MISS


@dataclass
class PhiraScore:
    """逐音符 commit 判定，最后输出 Phira 风格的结算数据。

    num_notes 是谱面音符总数；不传时按已 commit 的判定数计算。
    """

    num_notes: int | None = None
    perfect: int = 0
    good: int = 0
    bad: int = 0
    miss: int = 0
    combo: int = 0
    max_combo: int = 0

    @property
    def total_notes(self) -> int:
        if self.num_notes is not None:
            return self.num_notes
        return self.perfect + self.good + self.bad + self.miss

    @property
    def accuracy(self) -> float:
        n = self.total_notes
        if n <= 0:
            return 0.0
        return (self.perfect + self.good * GOOD_ACC_WEIGHT) / n

    @property
    def score(self) -> int:
        n = self.total_notes
        if n <= 0:
            return 0
        # 与 Phira 一致：全 Perfect 直接满分
        if self.perfect == n and self.good == self.bad == self.miss == 0:
            return TOTAL_SCORE

        raw = (
            0.9 * self.accuracy + 0.1 * self.max_combo / n
        ) * TOTAL_SCORE
        # Rust 的 f64::round 对正数是四舍五入；Python 的 round 是银行家舍入
        return math.floor(raw + 0.5)

    @property
    def full_combo(self) -> bool:
        return self.total_notes > 0 and self.max_combo == self.total_notes

    def commit(self, judgement: Judgement | str | int) -> None:
        """提交一个音符的最终判定，并更新连击。"""
        j = as_judgement(judgement)
        if j is Judgement.PERFECT:
            self.perfect += 1
        elif j is Judgement.GOOD:
            self.good += 1
        elif j is Judgement.BAD:
            self.bad += 1
        else:
            self.miss += 1

        if j in (Judgement.PERFECT, Judgement.GOOD):
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
        else:
            self.combo = 0

    def commit_tap(self, offset: float, speed: float = 1.0) -> Judgement:
        """按 Tap 时间偏差判定并提交。"""
        j = judge_tap(offset, speed)
        self.commit(j)
        return j

    def result(self) -> dict[str, object]:
        n = self.total_notes
        return {
            "counts": {
                "perfect": self.perfect,
                "good": self.good,
                "bad": self.bad,
                "miss": self.miss,
            },
            "accuracy": self.accuracy,
            "accuracy_pct": round(self.accuracy * 100.0, 2),
            "score": self.score,
            "combo": self.max_combo,
            "current_combo": self.combo,
            "max_combo": self.max_combo,
            "full_combo": self.full_combo,
            "num_notes": n,
        }


def score_judgements(
    judgements: Iterable[Judgement | str | int],
    num_notes: int | None = None,
) -> dict[str, object]:
    """直接按判定序列打分。"""
    items = list(judgements)
    scorer = PhiraScore(num_notes=len(items) if num_notes is None else num_notes)
    for judgement in items:
        scorer.commit(judgement)
    return scorer.result()


def score_tap_offsets(
    offsets: Iterable[float],
    num_notes: int | None = None,
    speed: float = 1.0,
) -> dict[str, object]:
    """按 Tap 时间偏差序列判定并打分。offset 正数表示打晚。"""
    items = list(offsets)
    scorer = PhiraScore(num_notes=len(items) if num_notes is None else num_notes)
    for offset in items:
        scorer.commit_tap(offset, speed)
    return scorer.result()
