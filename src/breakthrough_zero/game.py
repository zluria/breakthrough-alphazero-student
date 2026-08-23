"""Rules for Breakthrough.

The board is deliberately an ordinary flat Python list. Rows grow in Player 1's
forward direction. Player 1 therefore moves toward the last row; Player 2 moves
toward row zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1
ONGOING = None


@dataclass(frozen=True, order=True)
class Move:
    """A move between two compact, zero-based square indices."""

    from_sq: int
    to_sq: int


@dataclass(frozen=True)
class _Undo:
    move: Move
    captured: int
    previous_player: int
    previous_winner: int | None


class Breakthrough:
    """Mutable Breakthrough position with readable make/unmake operations.

    ``status()`` returns ``1`` for a Player-1 win, ``-1`` for a Player-2 win,
    and ``None`` while play is ongoing. A terminal move intentionally leaves
    ``player_to_move`` equal to the player who made that move.
    """

    def __init__(
        self,
        board_size: int = 5,
        starting_rows: int | None = None,
        *,
        board: Iterable[int] | None = None,
        player_to_move: int = PLAYER_1,
        winner: int | None = None,
    ) -> None:
        if board_size < 3:
            raise ValueError("board_size must be at least 3")
        if starting_rows is None:
            starting_rows = 2 if board_size == 8 else 1
        if not 1 <= starting_rows < board_size / 2:
            raise ValueError("starting_rows must leave space between the armies")
        if player_to_move not in (PLAYER_1, PLAYER_2):
            raise ValueError("player_to_move must be 1 or -1")

        self.board_size = board_size
        self.starting_rows = starting_rows
        self.player_to_move = player_to_move
        self.winner = winner
        self._history: list[_Undo] = []

        if board is None:
            self.board = [EMPTY] * (board_size * board_size)
            for row in range(starting_rows):
                for col in range(board_size):
                    self.board[self.square(row, col)] = PLAYER_1
                    self.board[self.square(board_size - 1 - row, col)] = PLAYER_2
        else:
            self.board = list(board)
            if len(self.board) != board_size * board_size:
                raise ValueError("board has the wrong number of squares")
            if any(piece not in (PLAYER_2, EMPTY, PLAYER_1) for piece in self.board):
                raise ValueError("board entries must be -1, 0, or 1")

    @property
    def action_size(self) -> int:
        """Number of relative policy outputs: one origin and three directions."""

        return self.board_size * self.board_size * 3

    def square(self, row: int, col: int) -> int:
        return row * self.board_size + col

    def row_col(self, square: int) -> tuple[int, int]:
        return divmod(square, self.board_size)

    def clone(self) -> "Breakthrough":
        clone = Breakthrough(
            self.board_size,
            self.starting_rows,
            board=self.board,
            player_to_move=self.player_to_move,
            winner=self.winner,
        )
        clone._history = self._history.copy()
        return clone

    def legal_moves(self) -> list[Move]:
        if self.winner is not None:
            return []
        return self._legal_moves_for(self.player_to_move)

    def _legal_moves_for(self, player: int) -> list[Move]:
        size = self.board_size
        forward = 1 if player == PLAYER_1 else -1
        moves: list[Move] = []
        for from_sq, piece in enumerate(self.board):
            if piece != player:
                continue
            row, col = self.row_col(from_sq)
            to_row = row + forward
            if not 0 <= to_row < size:
                continue
            for delta_col in (-1, 0, 1):
                to_col = col + delta_col
                if not 0 <= to_col < size:
                    continue
                to_sq = self.square(to_row, to_col)
                target = self.board[to_sq]
                if delta_col == 0:
                    if target == EMPTY:
                        moves.append(Move(from_sq, to_sq))
                elif target != player:
                    # Diagonal steps may enter an empty square or capture an enemy.
                    moves.append(Move(from_sq, to_sq))
        return moves

    def make_move(self, move: Move) -> None:
        if self.winner is not None:
            raise ValueError("cannot move after the game is over")
        if move not in self.legal_moves():
            raise ValueError(f"illegal move: {move}")

        player = self.player_to_move
        captured = self.board[move.to_sq]
        self._history.append(_Undo(move, captured, player, self.winner))
        self.board[move.from_sq] = EMPTY
        self.board[move.to_sq] = player

        goal_row = self.board_size - 1 if player == PLAYER_1 else 0
        to_row, _ = self.row_col(move.to_sq)
        if to_row == goal_row or -player not in self.board:
            self.winner = player
            return

        # A player unable to reply loses. Terminal moves never switch the player.
        if not self._legal_moves_for(-player):
            self.winner = player
            return

        self.player_to_move = -player

    def unmake_move(self, move: Move | None = None) -> Move:
        if not self._history:
            raise ValueError("no move to unmake")
        undo = self._history.pop()
        if move is not None and move != undo.move:
            self._history.append(undo)
            raise ValueError("move does not match the latest move")
        self.board[undo.move.from_sq] = undo.previous_player
        self.board[undo.move.to_sq] = undo.captured
        self.player_to_move = undo.previous_player
        self.winner = undo.previous_winner
        return undo.move

    def status(self) -> int | None:
        return self.winner

    def outcome(self) -> int | None:
        """Compatibility name used by the final-project handout."""

        return self.status()

    def _canonical_square(self, square: int) -> int:
        if self.player_to_move == PLAYER_1:
            return square
        return self.board_size * self.board_size - 1 - square

    def _absolute_square(self, canonical_square: int) -> int:
        # A 180-degree rotation is its own inverse.
        return self._canonical_square(canonical_square)

    def encode_move(self, move: Move) -> int:
        """Map an absolute move to a mover-relative policy action."""

        origin = self._canonical_square(move.from_sq)
        destination = self._canonical_square(move.to_sq)
        from_row, from_col = self.row_col(origin)
        to_row, to_col = self.row_col(destination)
        if to_row - from_row != 1 or to_col - from_col not in (-1, 0, 1):
            raise ValueError(f"move is not a one-step relative forward move: {move}")
        direction = to_col - from_col + 1
        return origin * 3 + direction

    def decode(self, action: int) -> Move:
        """Translate a relative policy action into an absolute move."""

        if not 0 <= action < self.action_size:
            raise ValueError("action is outside the policy head")
        canonical_from, direction = divmod(action, 3)
        row, col = self.row_col(canonical_from)
        to_row = row + 1
        to_col = col + direction - 1
        if not (0 <= to_row < self.board_size and 0 <= to_col < self.board_size):
            raise ValueError("action points outside the board")
        return Move(
            self._absolute_square(canonical_from),
            self._absolute_square(self.square(to_row, to_col)),
        )

    def legal_actions(self) -> list[int]:
        return [self.encode_move(move) for move in self.legal_moves()]

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def encode(self) -> np.ndarray:
        """Return exactly two mover-relative binary planes, flattened.

        The first plane contains the mover's pawns and the second the opponent's.
        The side to move is implicit because Player-2 positions are rotated.
        """

        from .neural import canonical_planes

        return canonical_planes(self).reshape(-1)

    @classmethod
    def from_rows(
        cls,
        rows: list[str],
        *,
        player_to_move: int = PLAYER_1,
        starting_rows: int = 1,
        winner: int | None = None,
    ) -> "Breakthrough":
        """Convenient position constructor used in tests and diagnostics.

        ``1`` is a Player-1 pawn, ``2`` is a Player-2 pawn, and ``.`` is empty.
        """

        size = len(rows)
        if any(len(row) != size for row in rows):
            raise ValueError("rows must form a square board")
        pieces = {"1": PLAYER_1, "2": PLAYER_2, ".": EMPTY}
        try:
            board = [pieces[cell] for row in rows for cell in row]
        except KeyError as error:
            raise ValueError(f"unknown board character: {error.args[0]}") from error
        return cls(
            size,
            starting_rows,
            board=board,
            player_to_move=player_to_move,
            winner=winner,
        )

    def to_rows(self) -> list[str]:
        symbols = {PLAYER_1: "1", PLAYER_2: "2", EMPTY: "."}
        size = self.board_size
        return [
            "".join(symbols[p] for p in self.board[row * size : (row + 1) * size])
            for row in range(size)
        ]

