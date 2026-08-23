"""The four exact Breakthrough symmetries used for data augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import Breakthrough, Move


@dataclass(frozen=True)
class Symmetry:
    swap_players: bool = False
    reflect_left_right: bool = False

    def square(self, square: int, size: int) -> int:
        row, col = divmod(square, size)
        if self.swap_players:
            row, col = size - 1 - row, size - 1 - col
        if self.reflect_left_right:
            col = size - 1 - col
        return row * size + col

    def move(self, move: Move, size: int) -> Move:
        return Move(self.square(move.from_sq, size), self.square(move.to_sq, size))

    def state(self, game: Breakthrough) -> Breakthrough:
        size = game.board_size
        board = [0] * (size * size)
        piece_sign = -1 if self.swap_players else 1
        for square, piece in enumerate(game.board):
            board[self.square(square, size)] = piece * piece_sign
        player = -game.player_to_move if self.swap_players else game.player_to_move
        winner = game.winner
        if winner is not None and self.swap_players:
            winner = -winner
        return Breakthrough(
            size,
            game.starting_rows,
            board=board,
            player_to_move=player,
            winner=winner,
        )

    def action(self, game: Breakthrough, action: int) -> int:
        move = game.decode(action)
        transformed = self.state(game)
        return transformed.encode_move(self.move(move, game.board_size))

    def policy(self, game: Breakthrough, policy: np.ndarray) -> np.ndarray:
        if policy.shape != (game.action_size,):
            raise ValueError("policy has the wrong shape")
        transformed = np.zeros_like(policy)
        for action, probability in enumerate(policy):
            try:
                transformed[self.action(game, action)] = probability
            except ValueError:
                # Off-board output cells are never legal and carry no useful target mass.
                if probability != 0:
                    raise
        return transformed


SYMMETRIES = (
    Symmetry(),
    Symmetry(reflect_left_right=True),
    Symmetry(swap_players=True),
    Symmetry(swap_players=True, reflect_left_right=True),
)
