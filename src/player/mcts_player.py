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
VALUE_DRAW = 20000
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
    DEFAULT_MODELFILE = "data/checkpoints/checkpoint-002.pth"

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
        elif sfen[:5] == "sfen ":
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
        # 探索回数の閾値を設定
        if infinite or ponder:
            # infiniteもしくはponderの場合は、探索を打ち切らないため、32bit整数の最大値とする
            self.halt = 2**31 - 1
        elif nodes:
            # 探索してよいノード数を指定された場合？
            self.halt = nodes
        else:
            self.remaining_time, inc = (
                (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            )
            if self.remaining_time is None and byoyomi is None and inc is None:
                # 時間指定がない場合
                self.halt = DEFAULT_CONST_PLAYOUT
            else:
                self.minimum_time = 0
                self.remaining_time = int(self.remaining_time) if self.remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = (
                    # この式の詳細はp.153, 囲碁ソフトEricaを参考にしている
                    self.remaining_time / (14 + max(0, 30 - self.root_board.move_number)) + inc
                )
                if byoyomi:
                    byoyomi = int(byoyomi) - self.byoyomi_margin
                    self.minimum_time = byoyomi
                    # time_limit が秒読み以下の場合、秒読みに設定
                    if self.time_limit < byoyomi:
                        self.time_limit = byoyomi

                # extend_timeがTrueなら最善手と次善手が僅差の場合に時間延長する
                self.extend_time = self.time_limit > self.minimum_time
                self.halt = None

    def go(self):
        """
        Returns:
            str: USI move or "resign" or "win"
            str or None: opponent move for the ponder
        """

        # 探索開始時間の記録
        self.begin_time = time.time()

        if self.root_board.is_game_over():
            # 投了
            return "resign", None

        if self.root_board.is_nyugyoku():
            # 入玉宣言勝ち
            return "win", None

        current_node = self.tree.current_head

        # 詰みの確認
        if current_node.value == VALUE_WIN:
            # 3手詰みの場合はルートノードのvalueがVALUE_WINに設定されている
            matemove = self.root_board.mate_move(3)  # 3手詰め確認
            if matemove != 0:
                print(f"info score mate 3 pv {move_to_usi(matemove)}", flush=True)
                return move_to_usi(matemove), None
        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print(f"info score mate 1 pv {move_to_usi(matemove)}", flush=True)
                return move_to_usi(matemove), None

        # clear playout count
        self.playout_count = 0

        # ルートノードが未展開の場合は展開
        if current_node.child_move is None:
            current_node.expand_node(self.root_node)

        # 候補手が1つの場合はその手を返す
        if self.halt is None and len(current_node.child_move) == 1:
            if current_node.child_move_count[0] > 0:
                bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
                return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None
            else:
                return move_to_usi(current_node.child_move[0]), None

        # ルートノードが未評価の場合、評価する
        if current_node.policy is None:
            print("ここにいつくる？")
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        self.search()

        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()

        # for debug
        if self.debug:
            for i in range(len(current_node.child_move)):
                print("{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}".format(
                    i, move_to_usi(current_node.child_move[i]),
                    current_node.child_move_count[i],
                    current_node.policy[i],
                    (current_node.child_sum_value[i] / current_node.child_move_count[i]
                     if current_node.child_move_count[i] > 0 else 0)
                ))

        # 閾値未満の場合投了
        if bestvalue < self.resign_threshold:
            return "resign", None

        return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None

    def stop(self):
        # すぐに中断する
        self.halt = 0

    def ponderhit(self, last_limits):
        self.begin_time = time.time()
        self.last_pv_print_time = 0

        self.playout_count = 0
        self.set_limits(**last_limits)

    def quit(self):
        self.stop()

    def search(self):
        self.last_pv_print_time = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search(board, self.tree.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1増やす
                    self.playout_count += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した、もしくはバックアップ済みのため廃棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result

            # 探索を打ち切るか確認
            if self.check_interruption():
                return

            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time) * 1000)
                if elapsed_time > self.last_pv_print_time + self.pv_interval:
                    self.last_pv_print_time = elapsed_time
                    self.get_bestmove_and_print_pv()

    def uct_search(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合、初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Loss を加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))

        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの確認
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち、もしくは優越局面の場合
                    # これらの場合、この局面ではこの手が一番良いことが確定するので最大値とする
                    # 子ノード（相手の局面）に対しての勝敗のためresultが反転している
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:  # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け、もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めのチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                # ここに来るのはQUEUINGを以前に返しているときのみなので
                # もう一度キューに入れないようにしている
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result
        else:
            # 探索結果の反映.
            update_result(current_node, next_index, result)
            return 1.0 - result

    def select_max_ucb_child(self, node):
        q = np.divide(node.child_sum_value, node.child_move_count,
                      out=np.zeros(len(node.child_move), np.float32),
                      where=node.child_move_count != 0)
        if node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(node.move_count)) / (1 + node.child_move_count)
        ucb = q + self.c_puct * u * node.policy

        return np.argmax(ucb)

    def get_bestmove_and_print_pv(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time

        # 訪問回数最大の手を選択する
        current_node = self.tree.current_head
        selected_index = np.argmax(current_node.child_move_count)

        # 選択した着手の勝率の算出
        bestvalue = (current_node.child_sum_value[selected_index]
                     / current_node.child_move_count[selected_index])
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30_000
        elif bestvalue == 0.0:
            cp = -30_000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)  # Ponanzaが使っていた式

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if (pv_node is None) or (pv_node.child_move is None) or (pv_node.move_count == 0):
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += " " + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print(f"info "
              f"nps {int(self.playout_count / finish_time) if finish_time > 0 else 0} "
              f"time {int(finish_time * 1000)} "
              f"nodes {current_node.move_count}"
              f"score cp {cp} pv {pv}",
              flush=True)

        return bestmove, bestvalue, ponder_move

    def check_interruption(self):
        """Check to interrupt search."""

        # プレイアウト数が閾値を超えている
        if self.halt is not None:
            return self.playout_count >= self.halt

        # 候補手が1つの場合は中断
        current_node = self.tree.current_head
        if len(current_node.child_move) == 1:
            return True

        # 消費時間
        spend_time = int((time.time() - self.begin_time) * 1000)

        # 消費時間が短すぎる場合、もしくは秒読みの場合は打ち切らない
        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False

        # 探索打ち切り
        #   残りの時間で可能なプレイアウト数をすべて2番目に訪問回数の多いノードに費やしても
        #   1番訪問回数が多いノードを超えない場合は、残りのプレイアウトは無駄になるので打ち切る.
        # 探索回数が最も多い手と次に多い手を求める
        child_move_count = current_node.child_move_count
        second_idx, first_idx = np.argpartition(child_move_count, -2)[-2:]
        second, first = child_move_count[[second_idx, first_idx]]

        # 探索測度から残りの時間で探索できるプレイアウト数を見積もる
        rest = int(self.playout_count * ((self.time_limit - spend_time) / spend_time))

        # 残りの探索で次善手が最善手を超える可能性がある場合は打ち切らない
        if first - second <= rest:
            return False

        # 探索延長
        #   21手目以降かつ、残り時間がある場合、
        #   最善手の探索回数が次善手の探索回数の1.5倍未満もしくは、勝率が逆なら探索延長する
        if (
            self.extend_time
            and self.root_board.move_number > 20
            and self.remaining_time > self.time_limit * 2
            and (first < second * 1.5
                 or (current_node.child_sum_value[first_idx] / child_move_count[first_idx]
                     < current_node.child_sum_value[second_idx] / child_move_count[second_idx]))
        ):
            # 探索時間を2倍に延長
            self.time_limit *= 2
            # 探索延長は1回のみ
            self.extend_time = False
            print("info string extend_time")
            return False

        return True

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
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
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

if __name__ == "__main__":
    player = MCTSPlayer()
    player.run()
