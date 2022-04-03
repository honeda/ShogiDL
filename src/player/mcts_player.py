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
            root_board: Root node for the search.
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

    def usi(self):
        o = "option name"
        print(f"id name: {self.name}")
        print(f"{o} USI_Ponder type check default false")
        print(f"{o} name modelfile type string default {self.DEFAULT_MODELFILE}")
        print(f"{o} gpu_id type spin default {DEFAULT_GPU_ID} min -1 max 7")
        print(f"{o} batchsize type spin default {DEFAULT_BATCH_SIZE} min 1 max 256")
        print(f"{o} resign_threshold type spin default {int(DEFAULT_RESIGN_THRESHOLD * 100)}"
              " min 0 max 100")
        print(f"{o} c_puct type spin default {int(DEFAULT_C_PUCT * 100)} min 10 max 1000")
        print(f"{o} temperature type spin default {int(DEFAULT_TEMPETATURE * 100)}"
              " min 10 max 1000")
        print(f"{o} time_margin type spin default {DEFAULT_TIME_MARGIN} min 0 max 1000")
        print(f"{o} byoyomi_margin type spin default {DEFAULT_BYOYOMI_MARGIN} min 0 max 1000")
        print(f"{o} pv_interval type spin default {DEFAULT_PV_INTERVAL} min 0 max 10000")
        print(f"{o} debug type check default false")

    def setoption(self, args):
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "batchsize":
            self.batch_size = int(args[3])
        elif args[1] == "resign_threshold":
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == "c_puct":
            self.c_puct = int(args[3]) / 100
        elif args[1] == "temperature":
            self.temperature = int(args[3]) / 100
        elif args[1] == "time_margin":
            self.time_margin = int(args[3])
        elif args[1] == "byoyomi_margin":
            self.byoyomi_margin = int(args[3])
        elif args[1] == "pv_interval":
            self.pv_interval = int(args[3])
        elif args[1] == "debug":
            self.debug = (args[3] == "true")

    def load_model(self):
        self.model = PolicyValueNetwork()
        self.model.to(self.device)
        checkpoint = torch.load(self.modelfile, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def init_features(self):
        """Initialize input features"""
        self.features = torch.empty(
            (self.batch_size, FEATURES_NUM, 9, 9),
            dtype=torch.float32,
            pin_memory=(self.gpu_id >= 0)  # GPUのときはTrueのが速い. CPUでTrueだとエラー
        )

    def isready(self):
        # set device
        if self.gpu_id >= 0:
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.load_model()

        # Initialized board
        self.root_board.reset()
        self.tree.reset_to_position(
            self.root_board.zobrist_hash(),
            moves=[]  # 初期局面なのでmovesは空
        )

        # 入力特徴量と評価待ちキューを初期化
        self.init_features()
        self.eval_queue = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index = 0

        # モデルをキャッシュして初回推論を速くする
        # 初期局面をbatch_size分用意して推論させるだけ
        current_node = self.tree.current_head
        current_node.expand_node(board=self.root_board)
        for _ in range(self.batch_size):
            self.queue_node(self.root_board, current_node)
        self.eval_node()

    def position(self, sfen, usi_moves):
        """
        Args:
            sfen (str): sfen code of the positons
            usi_moves (list): usi move list
        """
        if sfen == "startpos":
            # 開始局面から開始している場合
            self.root_board.reset()
        elif sfen[5:] == "sfen ":
            # sfen形式で局面が指定されている場合
            self.root_board.set_sfen(sfen[5:])

        moves = []
        for usi_move in usi_moves:
            move = self.root_board.push_usi(usi_move)
            moves.append(move)
        self.tree.reset_to_position(self.root_board.zobrist_hash(), moves)

        if self.debug:
            print(self.root_board)

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None,
                   winc=None, nodes=None, infinite=False, ponder=False):
        """
        Args:
            btime (int, optional): black's time condition. [ms]
                Defaults to None.
            wtime (int, optional): white's time condition. [ms]
                Defaults to None.
            byoyomi (int, optional): byo-yomi time. [ms] Defaults to None.
            binc (int, optional): black's time added per move under
                the Fischer rule. Defaults to None.
            winc (int, optional): white's time added per move under
                the Fischer rule. Defaults to None.
            nodes (_type_, optional): _description_. Defaults to None.
            infinite (bool, optional): if True, no time limit.
                Defaults to Flase.
            ponder (bool, optional): if True, USI_Ponder is ON.
                Defaults to False.
        """
        pass

    def make_input_features(self, board):
        make_input_features(board, self.features.numpy()[self.current_batch_index])

    def queue_node(self, board, node):
        """Add a node to the queue.

        Args:
            board (cshogi.Board): board
            node (UctNode): node
        """
        self.make_input_features(board)

        self.eval_queue[self.current_batch_index].set(node, board.turn)
        self.current_batch_index += 1

    def infer(self):
        """Predict"""
        with torch.no_grad():
            x = self.features[0:self.current_batch_index].to(self.device)
            policy_logits, value_logits = self.model(x)

            return (
                policy_logits.cpu().numpy(),
                torch.sigmoid(value_logits).cpu().numpy()
            )

    def make_move_label(self, move, color):
        return make_move_label(move, color)

    def eval_node(self):
        """Evalate board status."""

        # predict
        policy_logits, values = self.infer()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue[i].node
            color = self.eval_queue[i].color

            # all legal moves
            legal_move_probabilities = np.empty(len(current_node.chile_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann distribution
            probabilities = softmax_temperature_with_normalize(
                legal_move_probabilities,
                self.temperature
            )

            # update node's values
            current_node.policy = probabilities
            current_node.value = float(value)
