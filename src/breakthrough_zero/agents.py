"""Trusted, intentionally simple baseline agents."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from .game import Breakthrough, Move, PLAYER_1


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, game: Breakthrough) -> Move:
        moves = game.legal_moves()
        if not moves:
            raise ValueError("position has no legal move")
        return self.rng.choice(moves)


class TacticalRolloutAgent(RandomAgent):
    """Prefer an immediate win, then a capture, then a seeded random move."""

    def choose_move(self, game: Breakthrough) -> Move:
        moves = game.legal_moves()
        if not moves:
            raise ValueError("position has no legal move")
        captures: list[Move] = []
        for move in moves:
            is_capture = game.board[move.to_sq] == -game.player_to_move
            game.make_move(move)
            won = game.status() is not None
            game.unmake_move(move)
            if won:
                return move
            if is_capture:
                captures.append(move)
        return self.rng.choice(captures or moves)


def rollout_outcome(game: Breakthrough, rng: random.Random, tactical: bool = False) -> int:
    """Play a complete seeded rollout and return an absolute Player-1 outcome."""

    state = game.clone()
    agent = TacticalRolloutAgent() if tactical else None
    if agent is not None:
        agent.rng = rng
    while state.status() is None:
        moves = state.legal_moves()
        move = agent.choose_move(state) if agent is not None else rng.choice(moves)
        state.make_move(move)
    return int(state.status())


@dataclass
class SearchStats:
    nodes: int = 0
    completed_depth: int = 0


class AlphaBetaAgent:
    """Small alpha-beta baseline whose scores are absolute for Player 1."""

    def __init__(self, depth: int = 4, time_limit_s: float | None = None) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.time_limit_s = time_limit_s
        self.stats = SearchStats()
        self._deadline = math.inf

    def choose_move(self, game: Breakthrough) -> Move:
        moves = game.legal_moves()
        if not moves:
            raise ValueError("position has no legal move")
        self.stats = SearchStats()
        self._deadline = (
            time.perf_counter() + self.time_limit_s
            if self.time_limit_s is not None
            else math.inf
        )
        best_move = moves[0]
        max_depth = self.depth if self.time_limit_s is None else 100
        for depth in range(1, max_depth + 1):
            try:
                _, move = self.search(game, depth)
            except TimeoutError:
                break
            best_move = move
            self.stats.completed_depth = depth
        return best_move

    def search(self, game: Breakthrough, depth: int | None = None) -> tuple[float, Move]:
        depth = self.depth if depth is None else depth
        value, move = self._alphabeta(game, depth, -math.inf, math.inf)
        if move is None:
            raise ValueError("position has no legal move")
        return value, move

    def _alphabeta(
        self, game: Breakthrough, depth: int, alpha: float, beta: float
    ) -> tuple[float, Move | None]:
        if time.perf_counter() >= self._deadline:
            raise TimeoutError
        self.stats.nodes += 1
        outcome = game.status()
        if outcome is not None:
            return float(outcome), None
        if depth == 0:
            return self.evaluate(game), None

        moves = self._ordered_moves(game)
        if game.player_to_move == PLAYER_1:
            best_value = -math.inf
            best_move = moves[0]
            for move in moves:
                game.make_move(move)
                try:
                    value, _ = self._alphabeta(game, depth - 1, alpha, beta)
                finally:
                    game.unmake_move(move)
                if value > best_value:
                    best_value, best_move = value, move
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
        else:
            best_value = math.inf
            best_move = moves[0]
            for move in moves:
                game.make_move(move)
                try:
                    value, _ = self._alphabeta(game, depth - 1, alpha, beta)
                finally:
                    game.unmake_move(move)
                if value < best_value:
                    best_value, best_move = value, move
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
        return best_value, best_move

    @staticmethod
    def _ordered_moves(game: Breakthrough) -> list[Move]:
        """Immediate goals and captures first; stable indices break ties."""

        player = game.player_to_move
        goal = game.board_size - 1 if player == PLAYER_1 else 0

        def key(move: Move) -> tuple[int, int, int]:
            row, _ = game.row_col(move.to_sq)
            wins = int(row == goal)
            captures = int(game.board[move.to_sq] == -player)
            return (-wins, -captures, move.from_sq * game.board_size**2 + move.to_sq)

        return sorted(game.legal_moves(), key=key)

    @staticmethod
    def evaluate(game: Breakthrough) -> float:
        size = game.board_size
        p1 = [game.row_col(i)[0] for i, piece in enumerate(game.board) if piece == 1]
        p2 = [game.row_col(i)[0] for i, piece in enumerate(game.board) if piece == -1]
        material = (len(p1) - len(p2)) / max(1, size)
        progress = (sum(p1) - sum(size - 1 - row for row in p2)) / max(1, size**2)
        mobility = (
            len(game._legal_moves_for(1)) - len(game._legal_moves_for(-1))
        ) / max(1, 3 * size)
        return float(max(-0.99, min(0.99, 0.55 * material + 0.35 * progress + 0.1 * mobility)))


def solve_exact(game: Breakthrough, cache: dict | None = None) -> int:
    """Brute-force a tractable position, returning its absolute outcome."""

    if cache is None:
        cache = {}
    outcome = game.status()
    if outcome is not None:
        return int(outcome)
    key = (tuple(game.board), game.player_to_move)
    if key in cache:
        return cache[key]
    desired = game.player_to_move
    fallback = -desired
    for move in game.legal_moves():
        game.make_move(move)
        value = solve_exact(game, cache)
        game.unmake_move(move)
        if value == desired:
            cache[key] = desired
            return desired
    cache[key] = fallback
    return fallback
