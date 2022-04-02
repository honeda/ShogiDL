# MCTS: Monte Carlo tree search, モンテカルロ木探索
import time
import math

import numpy as np
import torch

from cshogi import (
    Board, BLACK, NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR, move_to_usi
)
from src.features import FEATURES_NUM, make_input_features, make_move_label
from src.uct.uct_node import NodeTree
from src.network.policy_value_resnet import PolicyValueNetwork
from src.player.base_player import BasePlayer


DEFAULT_GPU_ID = 0
DEFAULT_BATCH_SIZE = 32
DEFAULT_RESIGN_THRESHOLD = 0.01  # デフォルト投了閾値
DEFAULT_C_PUCT = 1.0             # デフォルトPUCTの定数
DEFAULT_TEMPETATURE = 1.0        # デフォルト温度パラメータ
DEFAULT_TIME_MARGIN = 1000       # ms
DEFAULT_BYOYOMI_MARGIN = 100     # ms
DEFAULT_PV_INTERVAL = 500        # ms
DEFAULT_CONST_PLAYOUT = 1000     # デフォルトプレイアウト数

# 勝ちを表す定数（数値に意味はない）
VALUE_WIN = 10000
# 負けを表す定数（数値に意味はない）
VALUE_LOSE = -10000
# 引き分けを表す定数（数値に意味はない）
VALUE_LOSE = 20000
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = -1
# 探索を破棄するときの戻り値（数値に意味はない）
DISCARDED = -2
# Virtual Loss
VIRTUAL_LOSS = 1


def softmax_temperature_with_normalize(logits, temperature):
    """温度パラメータを適用した確率分布を取得

    Args:
        logits (_type_): _description_
        temperature (_type_): _description_
    """
    logits /= temperature

    # 確率を計算（オーバフローを防止するため最大値で引く）
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるよう正規化
    probabilities = probabilities / sum(probabilities)

    return probabilities


def update_result(current_node, next_index, result):
    """Update node.

    Args:
        current_node (_type_): _description_
        next_index (_type_): _description_
        result (_type_): _description_
    """
    current_node.sum_value += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_sum_value[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS


class EvalQueueElement:
    # 評価待ちキューの要素
    def set(self, node, color):
        self.node = node
        self.color = color


class MCTSPlayer(BasePlayer):
    """Monte Carlo tree search player"""

    name = "python-dlshogi2"
    DEFAULT_MODELFILE = "checkpoints/checkpoint.pth"

    def __init__(self) -> None:
        """
        Attributes:
            modelfile: Path of model file. It can be change by `setoption()`
            model: Model instance.
            features: Input features. It's made by `isready()`
            eval_queue: Queues waiting for evalation of a phase to be evaluated
                by the neural network.
            current_batch_index: Indicates the number of stored queue
                waiting for evaluation.
                評価待ちキューの何番目まで格納したかを示す.
            root_board: Root node for search.
            tree: Game tree.
            playout_count: Number of playout.
                (Number of how many simulations were performed.)
            halt: Number of playout times to interrupt the search.
                It's set by `set_limits()`, `stop()`, `poderhit()`.
            gpu_id: GPU's ID. It can be change by `setoption()`.
            devide: Device.
            batch_size: Batch size for the neural network.
                It can be change by `setoption()`.
            resign_threshold: Threshold of the resignation.
                It can be change by `setoption()`.
            c_puct: Constants of the PUCT algorithm.
                It can be change by `setoption()`.
            temperature: Temperature of the policy.
                It can be change by `setoption()`.
            time_margin: Margin of the time condition.
                It can be change by `setoption()`.
            byoyomi_margin: Margin of the byoyomi.
                It can be change by `setoption()`.
            pv_interval: Interval that indicating thinking moves on the
                GUI software. It can be change by `setoption()`.
            debug: Flag of displaying debug massage.
                It can be change by `setoption()`.
        """
        super().__init__()
        self.modelfile = self.DEFAULT_MODELFILE
        self.model = None
        self.features = None
        self.eval_queue = None
        self.current_batch_index = 0

        self.root_board = Board()
        self.tree = NodeTree()
        self.playout_count = 0
        self.halt = None

        self.gpu_id = DEFAULT_GPU_ID
        self.device = None
        self.batch_size = DEFAULT_BATCH_SIZE

        self.resign_threshold = DEFAULT_RESIGN_THRESHOLD
        self.c_puct = DEFAULT_C_PUCT
        self.temperature = DEFAULT_TEMPETATURE
        self.time_margin = DEFAULT_TIME_MARGIN
        self.byoyomi_margin = DEFAULT_BYOYOMI_MARGIN
        self.pv_interval = DEFAULT_PV_INTERVAL

        self.debug = False