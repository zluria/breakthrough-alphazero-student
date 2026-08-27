"""Baseline agents for comparison and position solving."""

import math
import random
import time

from .game import PLAYER_1


class RandomAgent:
    def choose_move(self, game):
        moves = game.legal_moves()
        if not moves:
            raise ValueError("position has no legal move")
        return random.choice(moves)


def rollout_outcome(game):
    """Play to the end and return 1 for Player 1 or -1 for Player 2."""

    state = game.clone()
    while state.status() is None:
        move = random.choice(state.legal_moves())
        state.make_move(move)
    return state.status()


class AlphaBetaAgent:
    """An alpha-beta player with absolute Player-1 scores.

    Player 1 maximizes the score and Player 2 minimizes it. This is the same
    absolute-value convention used by PUCT.
    """

    def __init__(self, depth=4, time_limit_s=None):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.time_limit_s = time_limit_s
        self.stats = {"nodes": 0, "completed_depth": 0}
        self.deadline = math.inf

    def choose_move(self, game):
        moves = game.legal_moves()
        if not moves:
            raise ValueError("position has no legal move")

        self.stats = {"nodes": 0, "completed_depth": 0}
        if self.time_limit_s is None:
            self.deadline = math.inf
            unused_value, move = self.search(game, self.depth)
            self.stats["completed_depth"] = self.depth
            return move

        self.deadline = time.perf_counter() + self.time_limit_s
        best_move = moves[0]
        depth = 1
        while True:
            try:
                unused_value, move = self.search(game, depth)
            except TimeoutError:
                break
            best_move = move
            self.stats["completed_depth"] = depth
            depth += 1
        return best_move

    def search(self, game, depth=None):
        if depth is None:
            depth = self.depth
        value, move = self.alpha_beta(game, depth, -math.inf, math.inf)
        if move is None:
            raise ValueError("position has no legal move")
        return value, move

    def alpha_beta(self, game, depth, alpha, beta):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError

        self.stats["nodes"] += 1
        if game.status() is not None:
            # A proven result must dominate every finite horizon evaluation.
            if game.status() == PLAYER_1:
                return math.inf, None
            return -math.inf, None
        if depth == 0:
            return evaluate_position(game), None

        moves = ordered_moves(game)
        best_move = moves[0]

        if game.player_to_move == PLAYER_1:
            best_value = -math.inf
            for move in moves:
                game.make_move(move)
                # Always restore the position, including when a time limit
                # interrupts the recursive search.
                try:
                    value, unused_move = self.alpha_beta(
                        game, depth - 1, alpha, beta
                    )
                finally:
                    game.unmake_move()
                if value > best_value:
                    best_value = value
                    best_move = move
                if best_value > alpha:
                    alpha = best_value
                if alpha >= beta:
                    break
        else:
            best_value = math.inf
            for move in moves:
                game.make_move(move)
                # The same restoration invariant is required in the minimizing
                # branch.
                try:
                    value, unused_move = self.alpha_beta(
                        game, depth - 1, alpha, beta
                    )
                finally:
                    game.unmake_move()
                if value < best_value:
                    best_value = value
                    best_move = move
                if best_value < beta:
                    beta = best_value
                if alpha >= beta:
                    break

        return best_value, best_move


def ordered_moves(game):
    """Put goal moves and captures first."""

    player = game.player_to_move
    if player == PLAYER_1:
        goal_row = game.board_size - 1
    else:
        goal_row = 0

    def order_key(move):
        to_row, unused_col = game.row_col(move[1])
        wins = int(to_row == goal_row)
        captures = int(game.board[move[1]] == -player)
        return (-wins, -captures)

    return sorted(game.legal_moves(), key=order_key)


def evaluate_position(game):
    """A material, progress, and mobility evaluation."""

    size = game.board_size
    player_1_rows = []
    player_2_rows = []
    for square in range(len(game.board)):
        row, unused_col = game.row_col(square)
        if game.board[square] == 1:
            player_1_rows.append(row)
        if game.board[square] == -1:
            player_2_rows.append(row)

    material = (len(player_1_rows) - len(player_2_rows)) / max(1, size)
    player_1_progress = sum(player_1_rows)
    player_2_progress = 0
    for row in player_2_rows:
        player_2_progress += size - 1 - row
    progress = (player_1_progress - player_2_progress) / max(1, size * size)
    mobility = (
        len(game.legal_moves_for(1)) - len(game.legal_moves_for(-1))
    ) / max(1, 3 * size)

    # Material dominates, progress recognizes pawn races, and mobility breaks
    # ties between positions with similar pieces and advancement.
    value = 0.55 * material + 0.35 * progress + 0.10 * mobility
    return value


def solve_exact(game, cache=None):
    """Solve a position by trying every continuation.

    Breakthrough has no draws. If the mover has any continuation that produces
    its own absolute result, the position is a win; otherwise it is a loss.
    """

    if cache is None:
        cache = {}
    if game.status() is not None:
        return game.status()

    key = (tuple(game.board), game.player_to_move)
    if key in cache:
        return cache[key]

    desired_result = game.player_to_move
    for move in game.legal_moves():
        game.make_move(move)
        result = solve_exact(game, cache)
        game.unmake_move()
        if result == desired_result:
            cache[key] = desired_result
            return desired_result

    cache[key] = -desired_result
    return -desired_result
