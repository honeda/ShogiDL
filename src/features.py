import cshogi

# 移動方向を表す定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 入力特徴量の数
FEATURES_NUM = len(cshogi.PIECE_TYPES) * 2 + sum(cshogi.MAX_PIECES_IN_HAND) * 2

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(cshogi.HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81


def make_input_features(board, features):
    """

    Args:
        board (cshogi.Board): _description_
        features (ndarray): 3 dimensions array.
                Shape is (num of feature channel, 9, 9)
    """
    # initialize features
    features.fill(0)

    # pieces on board
    if board.turn == cshogi.BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        # 先手持ち駒のリスト、後手持ち駒のリストの順番なのでreversed
        pieces_in_hand = reversed(board.pieces_in_hand)

    # piece in hand
    i = 28  # 盤面の駒で28チャネル使うので
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, cshogi.MAX_PIECES_IN_HAND):
            features[i:i + num].fill(1)  # 30チャネル目が.fill(1)されたら歩を2枚持っている.
            i += max_num
