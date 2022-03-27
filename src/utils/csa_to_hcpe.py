# hcpeはAperyで使用されている形式

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from cshogi import HuffmanCodedPosAndEval, Board, BLACK, move16, CSA


class IligalMoveError(Exception):
    """不正な指し手を検出したときに使うエラー"""


parser = argparse.ArgumentParser()
parser.add_argument("csa_dir")
parser.add_argument("hcpe_train")
parser.add_argument("hcpe_test")
parser.add_argument("--filter_moves", type=int, default=50)
parser.add_argument("--filter_rating", type=int, default=3500)
parser.add_argument("--test_ratio", type=float, default=0.1)
args = parser.parse_args()

csa_file_list = [str(f) for f in Path(args.csa_dir).glob("**/*.csa")]

file_list_train, file_list_test = train_test_split(csa_file_list, test_size=args.test_ratio)

hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

f_train = open(args.hcpe_train, "wb")
f_test = open(args.hcpe_test, "wb")

board = Board()
for file_list, f, m in zip(
    [file_list_train, file_list_test], [f_train, f_test], ["train", "test"]
):
    kif_num = 0
    position_num = 0
    for filepath in file_list:
        for kif in CSA.Parser.parse_file(filepath):
            if kif.endgame not in ("%TORYO", "%SENNICHITE", "%KACHI"):
                # 投了、千日手、宣言勝ちで終了した棋譜を除外
                continue
            elif len(kif.moves) < args.filter_moves:
                # 手数が少ない棋譜を除外
                continue
            elif args.filter_rating > 0 and min(kif.ratings) < args.filter_rating:
                # レートの低いエンジンの対局を除外
                continue

            # 開始局面を設定
            board.set_sfen(kif.sfen)
            p = 0
            try:
                for i, (move, score, comment) in enumerate(
                    zip(kif.moves, kif.scores, kif.comments)
                ):
                    # 不正な指し手のある棋譜を除外
                    if not board.is_legal(move):
                        raise IligalMoveError()
                    hcpe = hcpes[p]
                    p += 1
                    # 局面はhcpに変換
                    board.to_hcp(hcpe["hcp"])
                    # 16bitに収まるようクリッピングする
                    eval_ = min(32767, max(score, -32767))
                    # 手番側の評価値にする
                    hcpe["eval"] = eval_ if board.turn == BLACK else -eval_
                    # 指し手の32bit数値を16bitに切り捨てる
                    hcpe["bestMove16"] = move16(move)
                    # 勝敗結果
                    hcpe["gameResult"] = kif.win
                    board.push(move)

            except Exception:
                print(f"skip {filepath}")
                continue

            if p == 0:
                # いつここにくる？
                print(f"p==0, skip {filepath}")
                continue

            hcpes[:p].tofile(f)

            kif_num += 1
            position_num += p

    print(f"{m}_kif_num:", kif_num)
    print(f"{m}_position_num:", position_num)
