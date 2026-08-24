"""The four exact Breakthrough transformations used for augmentation.

A symmetry is the tuple ``(swap_players, reflect_left_right)``. Swapping players
rotates the board by 180 degrees and negates every pawn; a Player-1 win therefore
becomes a Player-2 win. Left-right reflection does not change the winner.
"""

import numpy as np

from .game import Breakthrough


SYMMETRIES = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]


def transform_square(square, size, symmetry):
    swap_players, reflect = symmetry
    row, col = divmod(square, size)
    if swap_players:
        row = size - 1 - row
        col = size - 1 - col
    if reflect:
        col = size - 1 - col
    return row * size + col


def transform_move(move, size, symmetry):
    from_square, to_square = move
    return (
        transform_square(from_square, size, symmetry),
        transform_square(to_square, size, symmetry),
    )


def transform_state(game, symmetry):
    swap_players, unused_reflect = symmetry
    size = game.board_size
    board = [0] * (size * size)
    for square in range(len(game.board)):
        new_square = transform_square(square, size, symmetry)
        piece = game.board[square]
        if swap_players:
            piece = -piece
        board[new_square] = piece

    # State labels must transform with the pieces. This is what makes the
    # augmented position a genuinely equivalent game state.
    player = game.player_to_move
    winner = game.winner
    if swap_players:
        player = -player
        if winner is not None:
            winner = -winner
    return Breakthrough(size, game.starting_rows, board, player, winner)


def transform_action(game, action, symmetry):
    # Decode, transform, and encode again so policy targets obey exactly the
    # same mover-relative action convention as the transformed position.
    move = game.decode(action)
    new_game = transform_state(game, symmetry)
    new_move = transform_move(move, game.board_size, symmetry)
    return new_game.encode_move(new_move)


def transform_policy(game, policy, symmetry):
    if policy.shape != (game.action_size,):
        raise ValueError("policy has the wrong shape")
    new_policy = np.zeros_like(policy)
    for action in range(len(policy)):
        probability = policy[action]
        try:
            new_action = transform_action(game, action, symmetry)
            new_policy[new_action] = probability
        except ValueError:
            # Some policy-head cells point off the board. They are never legal,
            # so a valid target must assign them zero probability.
            if probability != 0:
                raise
    return new_policy
