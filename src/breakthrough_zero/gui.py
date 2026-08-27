"""Graphical play between humans, search baselines, and saved networks."""

import math
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk

import keras

from .agents import AlphaBetaAgent
from .game import Breakthrough, PLAYER_1, PLAYER_2
from .neural import GameNetwork, NeuralBoundary
from .puct import PUCTPlayer, RolloutEvaluator, best_action


CELL_SIZE = 64
BOARD_MARGIN = 24
HUMAN = "Human"
ALPHA_BETA = "Alpha-beta"
ROLLOUT_MCTS = "Vanilla MCTS (rollouts)"


def move_text(game, move):
    names = []
    for square in move:
        row, col = game.row_col(square)
        names.append(chr(ord("a") + col) + str(row + 1))
    return names[0] + "-" + names[1]


def board_row_from_display(board_size, display_row):
    """Convert a row counted from the top of the window to a board row."""

    return board_size - 1 - display_row


def find_checkpoints(directory):
    """Return the Keras model files below a directory."""

    paths = []
    if not directory or not os.path.isdir(directory):
        return paths
    for current_directory, unused_directories, filenames in os.walk(directory):
        for filename in filenames:
            lower_name = filename.lower()
            if lower_name.endswith((".keras", ".h5", ".hdf5")):
                path = os.path.join(current_directory, filename)
                paths.append(os.path.abspath(path))
    paths.sort()
    return paths


def checkpoint_label(path, model_directory):
    """Use a short path in the player selector while retaining unique names."""

    if model_directory:
        relative = os.path.relpath(path, model_directory)
    else:
        relative = os.path.basename(path)
    if relative == ".." or relative.startswith(".." + os.sep):
        relative = os.path.basename(path)
    name = os.path.splitext(relative)[0]
    return "Neural: " + name.replace(os.sep, "/")


def checkpoint_choices(paths, model_directory):
    """Map the names shown in the GUI to their saved model files."""

    choices = {}
    for path in paths:
        label = checkpoint_label(path, model_directory)
        if label in choices:
            label = "Neural: " + os.path.abspath(path)
        choices[label] = os.path.abspath(path)
    return choices


class GameWindow:
    def __init__(
        self,
        root,
        board_size,
        model_choices,
        simulations,
        loaded_networks=None,
    ):
        self.root = root
        self.board_size = board_size
        self.model_choices = model_choices
        self.networks = loaded_networks or {}
        self.game = None
        self.players = {}
        self.selected = None
        self.moves = []
        self.positions = []
        self.replay_ply = 0
        self.thinking = False
        self.running = False
        self.pending_move = None

        self.simulation_text = tk.StringVar(value=str(simulations))
        self.alpha_beta_time_text = tk.StringVar(value="0.1")
        self.player_1_text = tk.StringVar(value=HUMAN)
        neural_names = list(model_choices)
        if neural_names:
            player_2 = neural_names[0]
        else:
            player_2 = ALPHA_BETA
        self.player_2_text = tk.StringVar(value=player_2)

        root.title("Breakthrough arena")
        root.configure(padx=14, pady=14)
        root.columnconfigure(0, weight=1)

        self.make_controls()
        self.make_board()
        self.make_report()
        root.bind("<Left>", self.previous_key)
        root.bind("<Right>", self.next_key)

        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, font=("Segoe UI", 11)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self.new_game()

    def player_options(self):
        return [HUMAN, ALPHA_BETA, ROLLOUT_MCTS] + list(self.model_choices)

    def make_controls(self):
        controls = ttk.Frame(self.root)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Player 1 (X):").grid(row=0, column=0, sticky="w")
        self.player_1_box = ttk.Combobox(
            controls,
            textvariable=self.player_1_text,
            values=self.player_options(),
            state="readonly",
            width=34,
        )
        self.player_1_box.grid(row=0, column=1, sticky="w", padx=(5, 18))
        self.player_1_box.bind(
            "<<ComboboxSelected>>", lambda unused_event: self.new_game()
        )

        ttk.Label(controls, text="Player 2 (O):").grid(row=0, column=2, sticky="w")
        self.player_2_box = ttk.Combobox(
            controls,
            textvariable=self.player_2_text,
            values=self.player_options(),
            state="readonly",
            width=34,
        )
        self.player_2_box.grid(row=0, column=3, sticky="w", padx=(5, 0))
        self.player_2_box.bind(
            "<<ComboboxSelected>>", lambda unused_event: self.new_game()
        )

        ttk.Label(controls, text="MCTS simulations:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            controls,
            textvariable=self.simulation_text,
            values=(25, 50, 100, 200, 256, 500, 1000),
            width=7,
        ).grid(row=1, column=1, sticky="w", padx=(5, 18), pady=(8, 0))

        ttk.Label(controls, text="Alpha-beta seconds:").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            controls,
            textvariable=self.alpha_beta_time_text,
            values=(0.05, 0.1, 0.25, 0.5, 1.0),
            width=7,
        ).grid(row=1, column=3, sticky="w", padx=(5, 0), pady=(8, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=2, column=0, columnspan=4, sticky="w", pady=(9, 0))
        ttk.Button(buttons, text="New game", command=self.new_game).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(buttons, text="Swap players", command=self.swap_players).pack(
            side="left", padx=(0, 18)
        )
        ttk.Button(buttons, text="Play", command=self.play).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(buttons, text="Pause", command=self.pause).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(buttons, text="Step", command=self.step).pack(side="left")

    def make_board(self):
        board_pixels = self.board_size * CELL_SIZE + BOARD_MARGIN
        self.canvas = tk.Canvas(
            self.root,
            width=board_pixels,
            height=board_pixels,
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self.board_clicked)

    def make_report(self):
        report = ttk.Frame(self.root)
        report.grid(row=1, column=1, sticky="n", padx=(18, 0))

        self.search_title = tk.StringVar(value="Search for the selected move")
        ttk.Label(
            report,
            textvariable=self.search_title,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self.summary = tk.StringVar(value="No search yet.")
        ttk.Label(report, textvariable=self.summary).pack(anchor="w", pady=(2, 6))

        columns = ("move", "prior", "visits", "q", "u", "score")
        self.table = ttk.Treeview(report, columns=columns, show="headings", height=13)
        headings = ("Move", "P", "N", "Q (P1)", "U", "player·Q + U")
        widths = (75, 60, 55, 75, 65, 105)
        for index in range(len(columns)):
            self.table.heading(columns[index], text=headings[index])
            self.table.column(columns[index], width=widths[index], anchor="center")
        self.table.tag_configure("chosen", background="#d7f4d2")
        self.table.pack()
        ttk.Label(
            report,
            text=(
                "P is the policy prior; it is uniform for vanilla MCTS. "
                "N is the visit count. Q is absolute from Player 1's viewpoint, "
                "and U is the exploration bonus."
            ),
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(7, 12))

        move_heading = ttk.Frame(report)
        move_heading.pack(fill="x")
        ttk.Label(move_heading, text="Moves", font=("Segoe UI", 11, "bold")).pack(
            side="left"
        )
        ttk.Button(move_heading, text="|<", width=3, command=self.first_position).pack(
            side="left", padx=(18, 3)
        )
        ttk.Button(move_heading, text="<", width=3, command=self.previous_position).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(move_heading, text=">", width=3, command=self.next_position).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(move_heading, text="Live", command=self.go_live).pack(side="left")
        move_frame = ttk.Frame(report)
        move_frame.pack(fill="x", pady=(3, 0))
        move_columns = ("number", "player_1", "player_2")
        self.move_table = ttk.Treeview(
            move_frame,
            columns=move_columns,
            show="headings",
            height=9,
        )
        move_headings = ("#", "X", "O")
        move_widths = (42, 165, 165)
        for index in range(len(move_columns)):
            self.move_table.heading(move_columns[index], text=move_headings[index])
            self.move_table.column(
                move_columns[index], width=move_widths[index], anchor="center"
            )
        move_scrollbar = ttk.Scrollbar(
            move_frame, orient="vertical", command=self.move_table.yview
        )
        self.move_table.configure(yscrollcommand=move_scrollbar.set)
        self.move_table.pack(side="left")
        move_scrollbar.pack(side="right", fill="y")
        self.move_table.bind("<ButtonRelease-1>", self.move_clicked)

    def selected_name(self, player):
        if player == PLAYER_1:
            return self.player_1_text.get()
        return self.player_2_text.get()

    def has_human(self):
        return HUMAN in (self.player_1_text.get(), self.player_2_text.get())

    def read_settings(self):
        simulations = int(self.simulation_text.get())
        move_seconds = float(self.alpha_beta_time_text.get())
        if simulations < 1:
            raise ValueError("MCTS simulations must be positive")
        if move_seconds <= 0:
            raise ValueError("alpha-beta seconds must be positive")
        return simulations, move_seconds

    def load_network(self, name):
        path = self.model_choices[name]
        if path not in self.networks:
            self.status.set("Loading " + name + "...")
            self.root.update_idletasks()
            model = keras.models.load_model(path, compile=False)
            board_size = int(model.input_shape[1])
            self.networks[path] = GameNetwork(board_size, model=model)
        network = self.networks[path]
        if network.board_size != self.board_size:
            raise ValueError(
                name
                + " uses a "
                + str(network.board_size)
                + "x"
                + str(network.board_size)
                + " board, but this window uses "
                + str(self.board_size)
                + "x"
                + str(self.board_size)
            )
        return network

    def make_player(self, name, simulations, move_seconds):
        if name == HUMAN:
            return None
        if name == ALPHA_BETA:
            return AlphaBetaAgent(4, move_seconds)
        if name == ROLLOUT_MCTS:
            return PUCTPlayer(RolloutEvaluator(), simulations, 1.5)
        network = self.load_network(name)
        return PUCTPlayer(NeuralBoundary(network), simulations, 1.5)

    def new_game(self):
        self.pause()
        try:
            simulations, move_seconds = self.read_settings()
            players = {}
            players[PLAYER_1] = self.make_player(
                self.player_1_text.get(), simulations, move_seconds
            )
            players[PLAYER_2] = self.make_player(
                self.player_2_text.get(), simulations, move_seconds
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot start game", str(error))
            self.status.set("Choose compatible players and start a new game.")
            return

        self.players = players
        self.game = Breakthrough(self.board_size)
        self.selected = None
        self.moves = []
        self.positions = [self.game.clone()]
        self.replay_ply = 0
        self.thinking = False
        self.clear_search()
        for item in self.move_table.get_children():
            self.move_table.delete(item)
        self.draw_board()
        self.continue_game()

    def swap_players(self):
        first = self.player_1_text.get()
        self.player_1_text.set(self.player_2_text.get())
        self.player_2_text.set(first)
        self.new_game()

    def clear_search(self):
        for item in self.table.get_children():
            self.table.delete(item)
        self.search_title.set("Search for the selected move")
        self.summary.set("No search yet.")

    def play(self):
        if self.game is None:
            return
        self.go_live()
        if self.game.status() is not None:
            return
        self.running = True
        self.continue_game()

    def pause(self):
        self.running = False
        if self.pending_move is not None:
            self.root.after_cancel(self.pending_move)
            self.pending_move = None
            self.thinking = False
        if self.game is not None and self.game.status() is None:
            self.set_turn_status()

    def step(self):
        if self.game is None:
            return
        self.go_live()
        if self.game.status() is not None:
            return
        self.running = False
        if self.selected_name(self.game.player_to_move) == HUMAN:
            self.set_turn_status()
            return
        self.schedule_computer_move(20)

    def board_clicked(self, event):
        self.canvas.focus_set()
        if self.game is None or self.thinking or self.game.status() is not None:
            return
        if not self.is_live():
            return
        if self.selected_name(self.game.player_to_move) != HUMAN:
            return
        col = (event.x - BOARD_MARGIN) // CELL_SIZE
        display_row = (event.y - BOARD_MARGIN) // CELL_SIZE
        if display_row < 0 or display_row >= self.game.board_size:
            return
        if col < 0 or col >= self.game.board_size:
            return
        row = board_row_from_display(self.game.board_size, display_row)
        self.square_clicked(self.game.square(row, col))

    def square_clicked(self, square):
        player = self.game.player_to_move
        if self.selected is None:
            if self.game.board[square] == player:
                self.selected = square
                self.status.set("Now select a highlighted destination.")
                self.draw_board()
            return

        move = (self.selected, square)
        if move in self.game.legal_moves():
            name = move_text(self.game, move)
            self.game.make_move(move)
            self.record_move(move, name, None)
            self.selected = None
            self.draw_board()
            self.continue_game()
            return

        if self.game.board[square] == player:
            self.selected = square
            self.status.set("Now select a highlighted destination.")
        else:
            self.status.set("That is not a legal destination.")
        self.draw_board()

    def schedule_computer_move(self, delay):
        if self.thinking or self.pending_move is not None:
            return
        self.thinking = True
        self.pending_move = self.root.after(delay, self.computer_move)

    def computer_move(self):
        self.pending_move = None
        if self.game.status() is not None:
            self.thinking = False
            return

        player = self.game.player_to_move
        name = self.selected_name(player)
        agent = self.players[player]
        if agent is None:
            self.thinking = False
            self.set_turn_status()
            return

        try:
            simulations, move_seconds = self.read_settings()
        except ValueError as error:
            self.thinking = False
            messagebox.showerror("Invalid search setting", str(error))
            return

        self.status.set(name + " is thinking...")
        self.root.update_idletasks()
        started = time.perf_counter()

        if name == ALPHA_BETA:
            agent.time_limit_s = move_seconds
            move = agent.choose_move(self.game)
            elapsed = time.perf_counter() - started
            self.show_alpha_beta(agent, move, elapsed)
        else:
            agent.simulations = simulations
            result = agent.search(self.game)
            elapsed = time.perf_counter() - started
            action = best_action(result)
            move = self.game.decode(action)
            self.show_search(result, action, name, elapsed)

        text = move_text(self.game, move)
        search_report = self.save_search_report()
        self.game.make_move(move)
        self.record_move(move, text, search_report)
        self.selected = None
        self.thinking = False
        self.draw_board()
        self.continue_game()

    def record_move(self, move, move_name, search_report):
        self.moves.append((move, move_name, search_report))
        self.positions.append(self.game.clone())
        self.replay_ply = len(self.moves)

        move_number = (self.replay_ply + 1) // 2
        item = "move-" + str(move_number)
        if self.replay_ply % 2 == 1:
            self.move_table.insert(
                "", "end", iid=item, values=(move_number, move_name, "")
            )
        else:
            values = self.move_table.item(item, "values")
            self.move_table.item(item, values=(move_number, values[1], move_name))
        self.move_table.see(item)
        self.show_search_for_position()

    def is_live(self):
        return self.replay_ply == len(self.moves)

    def move_clicked(self, event):
        item = self.move_table.identify_row(event.y)
        column = self.move_table.identify_column(event.x)
        if not item or column not in ("#1", "#2", "#3"):
            return

        values = self.move_table.item(item, "values")
        move_number = int(values[0])
        if column == "#2":
            ply = 2 * move_number - 1
        elif column == "#3" and values[2]:
            ply = 2 * move_number
        else:
            ply = min(2 * move_number, len(self.moves))
        self.show_position(ply)

    def first_position(self):
        self.show_position(0)

    def previous_position(self):
        self.show_position(self.replay_ply - 1)

    def next_position(self):
        self.show_position(self.replay_ply + 1)

    def previous_key(self, unused_event):
        if isinstance(self.root.focus_get(), ttk.Combobox):
            return
        self.previous_position()
        return "break"

    def next_key(self, unused_event):
        if isinstance(self.root.focus_get(), ttk.Combobox):
            return
        self.next_position()
        return "break"

    def go_live(self):
        self.show_position(len(self.moves))

    def show_position(self, ply):
        if self.game is None:
            return
        self.pause()
        self.replay_ply = max(0, min(ply, len(self.moves)))
        self.selected = None
        self.draw_board()
        self.show_search_for_position()
        self.set_replay_status()

    def save_search_report(self):
        rows = []
        for item in self.table.get_children():
            rows.append(
                (self.table.item(item, "values"), self.table.item(item, "tags"))
            )
        return (self.search_title.get(), self.summary.get(), rows)

    def show_search_for_position(self):
        for item in self.table.get_children():
            self.table.delete(item)
        if self.replay_ply == 0:
            self.search_title.set("Search for the selected move")
            self.summary.set("The initial position has no preceding search.")
            return

        move_name = self.moves[self.replay_ply - 1][1]
        report = self.moves[self.replay_ply - 1][2]
        if report is None:
            self.search_title.set("No search — human move")
            self.summary.set(move_name + " was selected by the human player.")
            return

        title, summary, rows = report
        self.search_title.set(title)
        self.summary.set(summary)
        for values, tags in rows:
            self.table.insert("", "end", values=values, tags=tags)
        self.table.yview_moveto(0)

    def set_replay_status(self):
        if self.is_live():
            if self.game.status() is not None:
                self.show_winner()
            else:
                self.set_turn_status()
            return
        if self.replay_ply == 0:
            self.status.set("Replay: initial position.")
            return

        name = self.moves[self.replay_ply - 1][1]
        move_number = (self.replay_ply + 1) // 2
        if self.replay_ply % 2 == 1:
            prefix = str(move_number) + ". "
        else:
            prefix = str(move_number) + "... "
        self.status.set(
            "Replay: "
            + prefix
            + name
            + " (ply "
            + str(self.replay_ply)
            + " of "
            + str(len(self.moves))
            + ")."
        )

    def continue_game(self):
        if self.game.status() is not None:
            self.show_winner()
            return

        if self.selected_name(self.game.player_to_move) == HUMAN:
            self.set_turn_status()
            return

        if self.has_human() or self.running:
            self.schedule_computer_move(180)
        else:
            self.set_turn_status()

    def set_turn_status(self):
        if self.game is None or self.game.status() is not None:
            return
        if not self.is_live():
            self.set_replay_status()
            return
        player = self.game.player_to_move
        symbol = "X" if player == PLAYER_1 else "O"
        name = self.selected_name(player)
        if name == HUMAN:
            self.status.set(symbol + " is human. Select a pawn, then its destination.")
        else:
            self.status.set(symbol + " · " + name + " to move. Click Play or Step.")

    def show_search(self, result, chosen_action, name, elapsed):
        for item in self.table.get_children():
            self.table.delete(item)
        actions = list(result["visit_counts"])
        actions.sort(
            key=lambda action: result["visit_counts"][action], reverse=True
        )

        parent_visits = result["root_visits"]
        player = self.game.player_to_move
        for action in actions:
            prior = result["priors"][action]
            visits = result["visit_counts"][action]
            q_value = result["q_values"][action]
            exploration = 1.5 * prior * math.sqrt(parent_visits) / (1 + visits)
            score = player * q_value + exploration
            values = (
                move_text(self.game, self.game.decode(action)),
                "%.3f" % prior,
                visits,
                "%+.3f" % q_value,
                "%.3f" % exploration,
                "%+.3f" % score,
            )
            tags = ("chosen",) if action == chosen_action else ()
            self.table.insert("", "end", values=values, tags=tags)

        chosen_move = move_text(self.game, self.game.decode(chosen_action))
        self.search_title.set("PUCT search — " + name)
        self.summary.set(
            "Selected "
            + chosen_move
            + " · root N="
            + str(parent_visits)
            + " · root Q(P1)=%+.3f" % result["root_q"]
            + " · %.2fs" % elapsed
        )

    def show_alpha_beta(self, agent, move, elapsed):
        for item in self.table.get_children():
            self.table.delete(item)
        self.search_title.set("Alpha-beta search")
        self.summary.set(
            "Selected "
            + move_text(self.game, move)
            + " · completed depth "
            + str(agent.stats["completed_depth"])
            + " · "
            + str(agent.stats["nodes"])
            + " nodes · %.2fs" % elapsed
        )

    def draw_board(self):
        if self.game is None:
            return
        shown_game = self.positions[self.replay_ply]
        if self.replay_ply == 0:
            last_move = None
        else:
            last_move = self.moves[self.replay_ply - 1][0]

        destinations = []
        if self.is_live() and self.selected is not None:
            for move in self.game.legal_moves():
                if move[0] == self.selected:
                    destinations.append(move[1])

        self.canvas.delete("all")
        for display_row in range(self.game.board_size):
            self.canvas.create_text(
                BOARD_MARGIN / 2,
                BOARD_MARGIN + display_row * CELL_SIZE + CELL_SIZE / 2,
                text=str(self.game.board_size - display_row),
            )
        for col in range(self.game.board_size):
            self.canvas.create_text(
                BOARD_MARGIN + col * CELL_SIZE + CELL_SIZE / 2,
                BOARD_MARGIN / 2,
                text=chr(ord("a") + col),
            )

        for display_row in range(self.game.board_size):
            row = board_row_from_display(self.game.board_size, display_row)
            for col in range(self.game.board_size):
                square = self.game.square(row, col)
                x = BOARD_MARGIN + col * CELL_SIZE
                y = BOARD_MARGIN + display_row * CELL_SIZE
                # As on a chessboard, a1 is the dark square at lower left.
                color = "#b58863" if (row + col) % 2 == 0 else "#f0d9b5"
                if last_move is not None and square in last_move:
                    color = "#d8c768"
                if square == self.selected:
                    color = "#f6e56f"
                if square in destinations:
                    color = "#9fd68f"
                self.canvas.create_rectangle(
                    x,
                    y,
                    x + CELL_SIZE,
                    y + CELL_SIZE,
                    fill=color,
                    outline="",
                )
                piece = shown_game.board[square]
                if piece != 0:
                    symbol = "X" if piece == PLAYER_1 else "O"
                    piece_color = "#17365d" if piece == PLAYER_1 else "#8b1a1a"
                    self.canvas.create_text(
                        x + CELL_SIZE / 2,
                        y + CELL_SIZE / 2,
                        text=symbol,
                        fill=piece_color,
                        font=("Segoe UI", 26, "bold"),
                    )

    def show_winner(self):
        self.running = False
        winner = self.game.status()
        symbol = "X" if winner == PLAYER_1 else "O"
        name = self.selected_name(winner)
        self.status.set(
            symbol
            + " wins — "
            + name
            + " after "
            + str(len(self.game.history))
            + " plies."
        )


def run_gui(
    checkpoint=None,
    simulations=100,
    model_directory="checkpoints",
    board_size=None,
):
    paths = []
    loaded_networks = {}
    if checkpoint:
        checkpoint = os.path.abspath(checkpoint)
        paths.append(checkpoint)
        if board_size is None:
            model = keras.models.load_model(checkpoint, compile=False)
            board_size = int(model.input_shape[1])
            loaded_networks[checkpoint] = GameNetwork(board_size, model=model)

    for path in find_checkpoints(model_directory):
        if path not in paths:
            paths.append(path)
    if board_size is None:
        board_size = 8

    models = checkpoint_choices(paths, model_directory)
    root = tk.Tk()
    width = board_size * CELL_SIZE + 560
    height = max(board_size * CELL_SIZE + 175, 735)
    root.geometry(str(width) + "x" + str(height))
    root.lift()
    root.attributes("-topmost", True)
    root.after(1000, lambda: root.attributes("-topmost", False))
    GameWindow(root, board_size, models, simulations, loaded_networks)
    root.mainloop()
