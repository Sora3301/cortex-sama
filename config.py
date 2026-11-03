#  常量
from enum import IntEnum

n_slot: int = 4
# 槽位动作类别定义 (n_types = 4)
class Action_type(IntEnum):
    TYPE_IDLE = 0   # 空置
    TYPE_DOWN = 1   # 新按下 (click / drag / hold 起始帧)
    TYPE_HOLD = 2   # 持续按住或移动 (hold / drag 后续帧)
    TYPE_FLICK = 3  # 划动音符起始帧 (flick)

coord_w: float = 2.0                                      # coord_loss 相对 type_loss 的权重
class_weight: list[float] = [0.1, 1.0, 2.0, 4.0]          # 四种note的权重

num_threads: int = 6
batch_size: int = 16     # cpu视频解码
pin_memory: bool = True

seq_len: int = 16
warmup_len: int = 16
