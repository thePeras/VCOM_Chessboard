import json
from collections import defaultdict

def pieces_to_fen(pieces):
    board = [['' for _ in range(8)] for _ in range(8)]

    piece_map = {
        0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K', # white pieces
        6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k', # black pieces
        12: '.'
    }

    def pos_to_coords(pos):
        letter = pos[0]
        rank = pos[1]
        col = ord(letter) - ord('a')
        row = 8 - int(rank)
        return row, col

    for piece in pieces:
        row, col = pos_to_coords(piece['chessboard_position'])
        piece_char = piece_map.get(piece['category_id'], '?')
        board[row][col] = piece_char

    # Convert board to FEN string
    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen = '/'.join(fen_rows)
    return fen

def main():
    with open('complete_dataset/annotations.json', 'r') as f:
        data = json.load(f)

    # Collect all image ids from the three splits
    splits = data.get('splits', {}).get('chessred2k', {})
    test_ids = set(splits.get('test', {}).get('image_ids', []))
    train_ids = set(splits.get('train', {}).get('image_ids', []))
    val_ids = set(splits.get('val', {}).get('image_ids', []))

    valid_ids = test_ids | train_ids | val_ids

    # Group pieces by image_id
    pieces = data.get('annotations', {}).get('pieces', [])
    images = data.get('images', [])
    pieces_by_image = defaultdict(list)
    for piece in pieces:
        if piece['image_id'] in valid_ids:
            pieces_by_image[piece['image_id']].append(piece)

    # Create fen_annotations dict
    fen_annotations = {}
    for image_id in valid_ids:
        fen_annotations[images[image_id]["file_name"]] = pieces_to_fen(pieces_by_image.get(image_id, []))

    with open('annotations_fen.json', 'w') as f:
        json.dump(fen_annotations, f, indent=2)

if __name__ == '__main__':
    main()
