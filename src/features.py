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


def make_move_label(move, color):
    """make output labels for the policy network from move and color
    information.

    Args:
        move (int): integer for the movement of pieces
        color (int): 0(black, sente) or 1(white, gote)
    """
    if not cshogi.move_is_drop(move):
        to_sq = cshogi.move_to(move)
        from_sq = cshogi.move_from(move)

        # rotate the board if WHITE
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # direction of movement
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y

        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_x < 0:
                move_direction = UP_LEFT
            else:  # dir_x > 0
                move_direction = UP_RIGHT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = LEFT
            else:  # dir_x > 0
                move_direction = RIGHT
        elif dir_y > 0:
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_LEFT
            else:  # dir_x > 0
                move_direction = DOWN_RIGHT
        else:  # dir_y == -2 （桂馬）
            if dir_x == -1:
                move_direction = UP2_LEFT
            else:  # dir_x == 1
                move_direction = UP2_RIGHT

        # promotion（成り）
        if cshogi.move_is_promotion(move):
            move_direction += 10

    else:
        # drop the piece （駒打ち）
        to_sq = cshogi.move_to(move)
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq

        move_direction = len(MOVE_DIRECTION) + cshogi.move_drop_hand_piece(move)

    return move_direction * 81 + to_sq


def make_result(game_result, color):
    """return a label of game result for value netwaork

    Args:
        game_result (_type_): _description_
        color (_type_): _description_
    return:
        float: if win the color, return 1.0, lose -> 0.0, draw -> 0.5
    """
    if color == cshogi.BLACK:
        if game_result == cshogi.BLACK_WIN:
            return 1.0
        if game_result == cshogi.WHITE_WIN:
            return 0.0
    else:
        if game_result == cshogi.BLACK_WIN:
            return 0.0
        if game_result == cshogi.WHITE_WIN:
            return 1.0
    return 0.5
