"""Simple Breakthrough rules on a flat Python list.

A move is a tuple: ``(from_square, to_square)``. Squares are ordinary
zero-based indices into ``board``.
"""

import numpy as np


EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1


class Breakthrough:
    def __init__(
        self,
        board_size=5,
        starting_rows=None,
        board=None,
        player_to_move=PLAYER_1,
        winner=None,
    ):
        if board_size < 3:
            raise ValueError("board_size must be at least 3")
        if starting_rows is None:
            if board_size == 8:
                starting_rows = 2
            else:
                starting_rows = 1
        if starting_rows < 1 or starting_rows >= board_size / 2:
            raise ValueError("starting_rows must leave space between the armies")
        if player_to_move not in (PLAYER_1, PLAYER_2):
            raise ValueError("player_to_move must be 1 or -1")

        self.board_size = board_size
        self.starting_rows = starting_rows
        self.player_to_move = player_to_move
        self.winner = winner
        self.action_size = board_size * board_size * 3
        self.history = []

        if board is not None:
            self.board = list(board)
            if len(self.board) != board_size * board_size:
                raise ValueError("board has the wrong number of squares")
            for piece in self.board:
                if piece not in (PLAYER_2, EMPTY, PLAYER_1):
                    raise ValueError("board entries must be -1, 0, or 1")
            return

        self.board = [EMPTY] * (board_size * board_size)
        for row in range(starting_rows):
            for col in range(board_size):
                self.board[self.square(row, col)] = PLAYER_1
                other_row = board_size - 1 - row
                self.board[self.square(other_row, col)] = PLAYER_2

    def square(self, row, col):
        return row * self.board_size + col

    def row_col(self, square):
        return divmod(square, self.board_size)

    def clone(self):
        copy = Breakthrough(
            self.board_size,
            self.starting_rows,
            self.board,
            self.player_to_move,
            self.winner,
        )
        copy.history = list(self.history)
        return copy

    def legal_moves(self):
        if self.winner is not None:
            return []
        return self.legal_moves_for(self.player_to_move)

    def legal_moves_for(self, player):
        moves = []
        if player == PLAYER_1:
            forward = 1
        else:
            forward = -1

        for from_square in range(len(self.board)):
            if self.board[from_square] != player:
                continue
            row, col = self.row_col(from_square)
            next_row = row + forward
            if next_row < 0 or next_row >= self.board_size:
                continue

            for col_change in (-1, 0, 1):
                next_col = col + col_change
                if next_col < 0 or next_col >= self.board_size:
                    continue
                to_square = self.square(next_row, next_col)
                target = self.board[to_square]
                if col_change == 0 and target == EMPTY:
                    moves.append((from_square, to_square))
                if col_change != 0 and target != player:
                    moves.append((from_square, to_square))
        return moves

    def make_move(self, move):
        if self.winner is not None:
            raise ValueError("cannot move after the game is over")
        if move not in self.legal_moves():
            raise ValueError("illegal move: " + str(move))

        from_square, to_square = move
        player = self.player_to_move
        captured = self.board[to_square]
        self.history.append((move, captured, player, self.winner))

        self.board[from_square] = EMPTY
        self.board[to_square] = player

        if player == PLAYER_1:
            goal_row = self.board_size - 1
        else:
            goal_row = 0
        to_row, unused_col = self.row_col(to_square)

        if to_row == goal_row or -player not in self.board:
            self.winner = player
            return
        if not self.legal_moves_for(-player):
            self.winner = player
            return

        self.player_to_move = -player

    def unmake_move(self, move=None):
        if not self.history:
            raise ValueError("no move to unmake")
        undo = self.history.pop()
        old_move, captured, old_player, old_winner = undo
        if move is not None and move != old_move:
            self.history.append(undo)
            raise ValueError("move does not match the latest move")

        from_square, to_square = old_move
        self.board[from_square] = old_player
        self.board[to_square] = captured
        self.player_to_move = old_player
        self.winner = old_winner
        return old_move

    def status(self):
        return self.winner

    def outcome(self):
        return self.winner

    def canonical_square(self, square):
        if self.player_to_move == PLAYER_1:
            return square
        return self.board_size * self.board_size - 1 - square

    def encode_move(self, move):
        from_square, to_square = move
        relative_from = self.canonical_square(from_square)
        relative_to = self.canonical_square(to_square)
        from_row, from_col = self.row_col(relative_from)
        to_row, to_col = self.row_col(relative_to)
        if to_row - from_row != 1:
            raise ValueError("move is not one step forward")
        col_change = to_col - from_col
        if col_change not in (-1, 0, 1):
            raise ValueError("move has an invalid direction")
        return relative_from * 3 + col_change + 1

    def decode(self, action):
        if action < 0 or action >= self.action_size:
            raise ValueError("action is outside the policy head")
        relative_from, direction = divmod(action, 3)
        row, col = self.row_col(relative_from)
        to_row = row + 1
        to_col = col + direction - 1
        if to_row >= self.board_size or to_col < 0 or to_col >= self.board_size:
            raise ValueError("action points outside the board")

        relative_to = self.square(to_row, to_col)
        from_square = self.canonical_square(relative_from)
        to_square = self.canonical_square(relative_to)
        return (from_square, to_square)

    def legal_actions(self):
        actions = []
        for move in self.legal_moves():
            actions.append(self.encode_move(move))
        return actions

    def legal_action_mask(self):
        mask = np.zeros(self.action_size, dtype=np.bool_)
        for action in self.legal_actions():
            mask[action] = True
        return mask

    def encode(self):
        from .neural import canonical_planes

        return canonical_planes(self).reshape(-1)

    def to_rows(self):
        symbols = {PLAYER_1: "1", PLAYER_2: "2", EMPTY: "."}
        rows = []
        for row in range(self.board_size):
            text = ""
            start = row * self.board_size
            for piece in self.board[start : start + self.board_size]:
                text += symbols[piece]
            rows.append(text)
        return rows


def game_from_rows(rows, player_to_move=PLAYER_1, starting_rows=1, winner=None):
    """Build a test position from strings containing 1, 2, and dots."""

    size = len(rows)
    board = []
    symbols = {"1": PLAYER_1, "2": PLAYER_2, ".": EMPTY}
    for row in rows:
        if len(row) != size:
            raise ValueError("rows must form a square board")
        for symbol in row:
            if symbol not in symbols:
                raise ValueError("unknown board character: " + symbol)
            board.append(symbols[symbol])
    return Breakthrough(size, starting_rows, board, player_to_move, winner)
