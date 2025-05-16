import chess
import chess.svg
import cairosvg

def matrix_to_fen(matrix):
    fen_rows = []
    for row in matrix:
        fen_row = ""
        empty = 0
        for piece in row:
            if piece == "":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    fen_position = "/".join(fen_rows)
    return fen_position + " w KQkq - 0 1"

import chess
import chess.svg
import cairosvg
import numpy as np
import cv2
import io
from PIL import Image

def render_board_from_matrix(matrix, return_numpy=False, output_png=None):
    fen = matrix_to_fen(matrix)
    board = chess.Board(fen)
    svg = chess.svg.board(board=board)

    # Convert SVG to PNG bytes in memory
    png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))

    # Load PNG image from memory
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_np = np.array(image)

    if output_png:
        image.save(output_png)
        print(f"Board image saved to: {output_png}")

    if return_numpy:
        return img_np

if __name__ == "__main__":
    board_matrix = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]

    render_board_from_matrix(board_matrix, output_png="chessboard_example.png")
