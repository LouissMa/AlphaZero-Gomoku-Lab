# -*- coding: utf-8 -*-
"""
GUI for AlphaZero Gomoku using Pygame
"""

import pygame
import time
import pickle
from alphazero_gomoku.game import Board, Game
from alphazero_gomoku.mcts_pure import MCTSPlayer as MCTS_Pure
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer
from alphazero_gomoku.policy_value_net_numpy import PolicyValueNetNumpy


class GUI(object):
    def __init__(self, board_width=8, board_height=8):
        self.width = board_width
        self.height = board_height
        self.grid_size = 40
        self.margin = 40
        self.board_width_px = self.grid_size * (self.width - 1) + 2 * self.margin
        self.board_height_px = self.grid_size * (self.height - 1) + 2 * self.margin

        pygame.init()
        self.screen = pygame.display.set_mode((self.board_width_px, self.board_height_px))
        pygame.display.set_caption("AlphaZero Gomoku")
        self.font = pygame.font.Font(None, 36)

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.bg_color = (200, 150, 100)  # Wood color

    def draw_board(self, board):
        self.screen.fill(self.bg_color)

        # Draw grid lines
        for i in range(self.width):
            start = (self.margin + i * self.grid_size, self.margin)
            end = (
                self.margin + i * self.grid_size,
                self.margin + (self.height - 1) * self.grid_size,
            )
            pygame.draw.line(self.screen, self.black, start, end, 1)

        for i in range(self.height):
            start = (self.margin, self.margin + i * self.grid_size)
            end = (
                self.margin + (self.width - 1) * self.grid_size,
                self.margin + i * self.grid_size,
            )
            pygame.draw.line(self.screen, self.black, start, end, 1)

        # Draw pieces
        for move, player in board.states.items():
            h, w = board.move_to_location(move)
            center = (self.margin + w * self.grid_size, self.margin + h * self.grid_size)
            if player == board.players[0]:
                pygame.draw.circle(self.screen, self.black, center, self.grid_size // 2 - 2)
            else:
                pygame.draw.circle(self.screen, self.white, center, self.grid_size // 2 - 2)

        # Highlight last move
        if board.last_move != -1:
            h, w = board.move_to_location(board.last_move)
            center = (self.margin + w * self.grid_size, self.margin + h * self.grid_size)
            pygame.draw.circle(self.screen, (255, 0, 0), center, 4)

        pygame.display.flip()

    def get_human_move(self, board):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return -1
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    # Convert mouse pos to board coordinate
                    # Using round to find the nearest intersection
                    w = round((mouse_x - self.margin) / self.grid_size)
                    h = round((mouse_y - self.margin) / self.grid_size)

                    if 0 <= w < self.width and 0 <= h < self.height:
                        move = board.location_to_move([h, w])
                        if move in board.availables:
                            return move
            time.sleep(0.05)


class Human(object):
    def __init__(self, gui):
        self.gui = gui
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        return self.gui.get_human_move(board)

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = "models/best_policy_8_8_5.model"

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        gui = GUI(width, height)

        # Load AI model
        try:
            policy_param = pickle.load(open(model_file, "rb"))
        except:
            policy_param = pickle.load(open(model_file, "rb"), encoding="bytes")

        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        human = Human(gui)

        # Start play
        # Modified start_play to use GUI drawing instead of terminal print

        # We need to manually control the game loop to integrate with GUI better
        # or we can modify Game.start_play to accept a callback.
        # Here I will reimplement a simple game loop using the existing game logic

        board.init_board(start_player=0)
        gui.draw_board(board)

        p1, p2 = board.players
        human.set_player_ind(p1)
        mcts_player.set_player_ind(p2)
        players = {p1: human, p2: mcts_player}

        while True:
            current_player = board.get_current_player()
            player_in_turn = players[current_player]

            # Handle events to keep window responsive during AI thinking
            pygame.event.pump()

            if isinstance(player_in_turn, Human):
                move = player_in_turn.get_action(board)
                if move == -1:
                    break  # Quit
            else:
                move = player_in_turn.get_action(board)

            board.do_move(move)
            gui.draw_board(board)

            end, winner = board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                    text = "Winner: " + ("Human" if winner == p1 else "AI")
                else:
                    print("Game end. Tie")
                    text = "Tie"

                # Show result on screen
                font = pygame.font.Font(None, 48)
                text_surface = font.render(text, True, (0, 0, 255))
                text_rect = text_surface.get_rect(
                    center=(gui.board_width_px // 2, gui.board_height_px // 2)
                )
                gui.screen.blit(text_surface, text_rect)
                pygame.display.flip()

                # Wait for click to exit
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                            waiting = False
                break

    except KeyboardInterrupt:
        print("\n\rquit")
    finally:
        pygame.quit()


if __name__ == "__main__":
    run()
