"""Local graphical play with the final PUCT statistics for each move."""

import math
import tkinter as tk
from tkinter import ttk

from .game import Breakthrough, PLAYER_1, PLAYER_2
from .neural import GameNetwork, NeuralBoundary
from .puct import PUCTPlayer, best_action


CELL_SIZE = 72
BOARD_MARGIN = 24


def move_text(game, move):
    names = []
    for square in move:
        row, col = game.row_col(square)
        names.append(chr(ord("a") + col) + str(row + 1))
    return names[0] + "-" + names[1]


class GameWindow:
    def __init__(self, root, network, simulations):
        self.root = root
        self.network = network
        self.game = None
        self.human_player = PLAYER_1
        self.selected = None
        self.thinking = False
        self.simulation_text = tk.StringVar(value=str(simulations))
        self.computer = PUCTPlayer(NeuralBoundary(network), simulations, 1.5)

        root.title("Breakthrough AlphaZero")
        root.configure(padx=14, pady=14)

        controls = ttk.Frame(root)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Button(
            controls, text="New game as X", command=lambda: self.new_game(PLAYER_1)
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            controls, text="New game as O", command=lambda: self.new_game(PLAYER_2)
        ).pack(side="left")
        simulation_choices = ttk.Combobox(
            controls,
            textvariable=self.simulation_text,
            values=(25, 50, 100, 200),
            state="readonly",
            width=5,
        )
        simulation_choices.pack(side="right")
        ttk.Label(controls, text="PUCT simulations: ").pack(side="right")

        board_pixels = network.board_size * CELL_SIZE + BOARD_MARGIN
        self.canvas = tk.Canvas(
            root,
            width=board_pixels,
            height=board_pixels,
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self.board_clicked)

        report = ttk.Frame(root)
        report.grid(row=1, column=1, sticky="n", padx=(18, 0))
        ttk.Label(
            report,
            text="PUCT after the network's last search",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self.summary = tk.StringVar(value="No search yet.")
        ttk.Label(report, textvariable=self.summary).pack(anchor="w", pady=(2, 6))

        columns = ("move", "prior", "visits", "q", "u", "score")
        self.table = ttk.Treeview(report, columns=columns, show="headings", height=15)
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
                "P is the neural prior. N is the visit count. Q is absolute "
                "from Player 1's viewpoint. U is the exploration bonus."
            ),
            wraplength=430,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, font=("Segoe UI", 11)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self.new_game(PLAYER_1)

    def new_game(self, human_player):
        self.game = Breakthrough(self.network.board_size, 1)
        self.human_player = human_player
        self.selected = None
        self.thinking = False
        for item in self.table.get_children():
            self.table.delete(item)
        self.summary.set("No search yet.")
        if human_player == PLAYER_1:
            self.status.set("You are X. Select a pawn, then its destination.")
        else:
            self.status.set("You are O. The network moves first.")
        self.draw_board()
        if self.game.player_to_move != self.human_player:
            self.root.after(100, self.computer_move)

    def board_clicked(self, event):
        if self.thinking or self.game.status() is not None:
            return
        if self.game.player_to_move != self.human_player:
            return
        col = (event.x - BOARD_MARGIN) // CELL_SIZE
        row = (event.y - BOARD_MARGIN) // CELL_SIZE
        if row < 0 or row >= self.game.board_size:
            return
        if col < 0 or col >= self.game.board_size:
            return
        self.square_clicked(self.game.square(row, col))

    def square_clicked(self, square):
        if self.selected is None:
            if self.game.board[square] == self.human_player:
                self.selected = square
                self.status.set("Now select a highlighted destination.")
                self.draw_board()
            return

        move = (self.selected, square)
        if move in self.game.legal_moves():
            self.game.make_move(move)
            self.selected = None
            self.draw_board()
            if self.game.status() is not None:
                self.show_winner()
            else:
                self.status.set("The network is thinking...")
                self.thinking = True
                self.root.after(50, self.computer_move)
            return

        if self.game.board[square] == self.human_player:
            self.selected = square
            self.status.set("Now select a highlighted destination.")
        else:
            self.status.set("That is not a legal destination.")
        self.draw_board()

    def computer_move(self):
        if self.game.status() is not None:
            return
        self.thinking = True
        self.status.set("The network is thinking...")
        self.root.update_idletasks()

        self.computer.simulations = int(self.simulation_text.get())
        result = self.computer.search(self.game)
        action = best_action(result)
        move = self.game.decode(action)
        name = move_text(self.game, move)
        self.show_search(result, action)
        self.game.make_move(move)

        self.thinking = False
        self.draw_board()
        if self.game.status() is not None:
            self.show_winner()
        else:
            self.status.set("The network played " + name + ". Your turn.")

    def show_search(self, result, chosen_action):
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
        self.summary.set(
            "Selected "
            + chosen_move
            + " · root N="
            + str(parent_visits)
            + " · root Q(P1)=%+.3f" % result["root_q"]
        )

    def draw_board(self):
        destinations = []
        if self.selected is not None:
            for move in self.game.legal_moves():
                if move[0] == self.selected:
                    destinations.append(move[1])

        self.canvas.delete("all")
        for row in range(self.game.board_size):
            self.canvas.create_text(
                BOARD_MARGIN / 2,
                BOARD_MARGIN + row * CELL_SIZE + CELL_SIZE / 2,
                text=str(row + 1),
            )
        for col in range(self.game.board_size):
            self.canvas.create_text(
                BOARD_MARGIN + col * CELL_SIZE + CELL_SIZE / 2,
                BOARD_MARGIN / 2,
                text=chr(ord("a") + col),
            )

        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                square = self.game.square(row, col)
                x = BOARD_MARGIN + col * CELL_SIZE
                y = BOARD_MARGIN + row * CELL_SIZE
                color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
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
                piece = self.game.board[square]
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
        if self.game.status() == self.human_player:
            self.status.set("You win.")
        else:
            self.status.set("The network wins.")


def run_gui(checkpoint, simulations):
    import keras

    model = keras.models.load_model(checkpoint, compile=False)
    network = GameNetwork(int(model.input_shape[1]), model=model)
    root = tk.Tk()
    root.geometry("900x475")
    root.lift()
    root.attributes("-topmost", True)
    root.after(1000, lambda: root.attributes("-topmost", False))
    GameWindow(root, network, simulations)
    root.mainloop()
