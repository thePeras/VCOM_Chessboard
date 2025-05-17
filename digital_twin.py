import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json, argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2 as transforms
from sklearn.metrics import accuracy_score, f1_score
from torchsummary import summary
from board_draw import render_board_from_matrix

random.seed(42)

data_aug = transforms.Compose([
    transforms.ToImage(),
    # full-range rotation (keep entire image, fill border with mean colour)
    transforms.RandomRotation(
        degrees=(-180, 180),
        expand=True,                         # keep corners
        fill=(123, 117, 104),                # roughly ImageNet mean in 0-255
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.04),
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.1, 1.5))],
        p=0.3,
    ),

    # sizing (keep whole board in view)
    transforms.Resize(290),
    transforms.CenterCrop(224),

    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),

    # occlusion (done on the tensor, after normalisation)
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value='random'),
])

# Normalize images
# data_aug = transforms.Compose([
#     transforms.ToImage(),
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop((224, 224)),
#     transforms.ToDtype(torch.float32, scale=True),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

data_in = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def chesspos2number(chesspos):
    col = ord(chesspos[0])-ord('a')
    row = 8 - int(chesspos[1])  # can be confusing to visualize (since a1 corresponds to row 0, col 0)
    return row, col

def piece_name_to_char(name):
    piece_names = {
        "pawn": "p",
        "rook": "r",
        "knight": "n",
        "bishop": "b",
        "queen": "q",
        "king": "k",
    }
    color_names = {
        "white": True,
        "black": False,
    }
    piece_char = None
    for piece_name, letter in piece_names.items():
        if name.endswith(piece_name):
            piece_char = letter

    for color_name, upper_case in color_names.items():
        if name.startswith(color_name):
            if upper_case and piece_char is not None:
                piece_char = piece_char.upper()
            break

    if piece_char is None:
        return ""
    return piece_char

def chess_piece_id_to_char(id):
    # annotations: hardcoded from dataset["categories"]
    anns = [{'id': 0, 'name': 'white-pawn'}, {'id': 1, 'name': 'white-rook'}, {'id': 2, 'name': 'white-knight'}, {'id': 3, 'name': 'white-bishop'}, {'id': 4, 'name': 'white-queen'}, {'id': 5, 'name': 'white-king'}, {'id': 6, 'name': 'black-pawn'}, {'id': 7, 'name': 'black-rook'}, {'id': 8, 'name': 'black-knight'}, {'id': 9, 'name': 'black-bishop'}, {'id': 10, 'name': 'black-queen'}, {'id': 11, 'name': 'black-king'}, {'id': 12, 'name': 'empty'}]
    cats = [None] * len(anns)
    for cat in anns:
        cats[cat["id"]] = cat["name"]
    return piece_name_to_char(cats[id])

def board_to_chars(board):
    board_chars = [[''] * 8 for _ in range(8)]
    for ri, row in enumerate(board):
        for ci, id in enumerate(row):
            board_chars[ri][ci] = chess_piece_id_to_char(id)
    return board_chars

class ChessDataset(Dataset):
    def __init__(self, root_dir, images_dir, partition, transform=None):
        self.anns = json.load(open(os.path.join(root_dir, 'annotations.json')))
        self.categories = [c['name'] for c in self.anns['categories']]
        self.root = root_dir
        self.images_dir = images_dir
        self.ids = []
        self.file_names = []
        for x in self.anns['images']:
            self.file_names.append(x['path'])
            self.ids.append(x['id'])
        self.file_names = np.asarray(self.file_names)
        self.ids = np.asarray(self.ids)
        self.occupancy_boards = torch.zeros((len(self.file_names), 8, 8))
        self.boards = torch.full((len(self.file_names), 8, 8), 12, dtype=torch.long)

        for piece in self.anns['annotations']['pieces']:
            idx = np.where(self.ids == piece['image_id'])[0][0]
            row, col = chesspos2number(piece['chessboard_position'])
            self.occupancy_boards[idx][row][col] = 1
            self.boards[idx][row][col] = piece["category_id"]   # [0,11] means a specific piece, 12 means empty

        if partition == 'train':
            self.split_ids = np.asarray(self.anns['splits']['chessred2k']['train']['image_ids']).astype(int)
        elif partition == 'valid':
            self.split_ids = np.asarray(self.anns['splits']['chessred2k']['val']['image_ids']).astype(int)
        else:
            self.split_ids = np.asarray(self.anns['splits']['chessred2k']['test']['image_ids']).astype(int)

        intersect = np.isin(self.ids, self.split_ids)
        self.split_ids = np.where(intersect)[0]
        self.file_names = self.file_names[self.split_ids]
        self.occupancy_boards = self.occupancy_boards[self.split_ids]
        self.boards = self.boards[self.split_ids]
        self.num_pieces = torch.sum(self.occupancy_boards.view(len(self.occupancy_boards), 64), axis=-1)
        self.num_pieces = F.one_hot(self.num_pieces.long()-1, 32)
        self.ids = self.ids[self.split_ids]

        self.transform = transform
        print(f"Number of {partition} images: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        # about 750x750 (3000 / 4)
        image = cv2.imread(os.path.join(self.images_dir, self.file_names[i]), cv2.IMREAD_REDUCED_COLOR_4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        num_pieces = self.num_pieces[i]
        board = self.boards[i]
        occupancy_board = self.occupancy_boards[i]

        return image, num_pieces.float(), board, occupancy_board

class ChessboardPredictor(nn.Module):
    def __init__(self, backbone, in_channels):
        super().__init__()
        self.backbone = backbone
        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
        #     nn.Conv2d(512, 13, 1),
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), # 7→14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # 14→28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((8, 8)),   # robust to any input crop/resize
            nn.Dropout(0.3),
            nn.Conv2d(128, 13, 1)            # logits
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.decoder(x)

def epoch_iter(model, dataloader, loss_fn, optimizer=None, is_training=True, device='cuda'):
    model.train() if is_training else model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    for images, _, boards, __ in dataloader:
        images = images.to(device)               # (B, 3, 224, 224)
        targets = boards.long().to(device)       # (B, 8, 8)

        outputs = model(images)                  # (B, 13, 8, 8)
        loss = loss_fn(outputs, targets)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)            # (B, 8, 8)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy = (all_preds == all_targets).float().mean().item()

    return avg_loss, accuracy

def train_model(model, train_loader, valid_loader, optimizer, scheduler=None, device='cuda', epochs=10) -> nn.Module:
    best_val_acc = 0.0
    best_model_state = None
    print(">> Training model")
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = epoch_iter(model, train_loader, loss_fn, optimizer=optimizer, is_training=True, device=device)

        # Validate
        val_loss, val_acc = epoch_iter(model, valid_loader, loss_fn, is_training=False, device=device)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if scheduler:
            scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def experiment(dataloader):
    for batch in dataloader:
        # Get images of the batch and print their dimensions
        imgs = batch[0]
        imgs = imgs.permute(0, 2, 3, 1)*torch.tensor([[[0.229, 0.224, 0.225]]]) + torch.tensor([[[0.485, 0.456, 0.406]]])

        # Get labels of each image in the batch and print them
        labels = batch[1]
        print("labels")
        print(labels[0])

        boards = batch[2]
        print("boards shape")
        print(boards.shape)

        print(boards[0])
        chars_board = board_to_chars(boards[0].cpu().long().numpy())
        print(chars_board)
        render_board_from_matrix(chars_board)

        occupancy_boards = batch[3]
        occ_board = occupancy_boards[0]
        print(occ_board)

        # Show first image of the batch
        plt.imshow(imgs[0])
        plt.axis('off')
        plt.savefig("figure.png")
        break

# def visualize(images, preds):
def denormalise(img_tensor):
    """Undo ImageNet normalisation and return uint8 RGB numpy array."""
    img = img_tensor.cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = torch.clamp(img, 0, 1).numpy()
    return (img * 255).astype(np.uint8)

def make_board_img(board, label):
    """Render an 8x8 board (using your helper) and write a small title on top."""
    board_img = render_board_from_matrix(board_to_chars(board), return_numpy=True)
    # add a white margin for the text
    margin = 30
    board_img = np.vstack([255 * np.ones((margin, board_img.shape[1], 3), np.uint8),
                           board_img])
    cv2.putText(board_img, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return board_img

def visualise_sample(img_tensor, pred_board, gt_board, out_file):
    """Stack: original image (photo), prediction, ground-truth and write to disk."""
    photo = denormalise(img_tensor)
    h, w  = photo.shape[:2]

    pred_img = make_board_img(pred_board, "Prediction")
    gt_img   = make_board_img(gt_board,   "Ground truth")

    # resize chessboards so they have the same width as the photo
    pred_img = cv2.resize(pred_img, (w, w), interpolation=cv2.INTER_AREA)
    gt_img   = cv2.resize(gt_img,   (w, w), interpolation=cv2.INTER_AREA)

    canvas = np.concatenate([photo, pred_img, gt_img], axis=0)
    cv2.imwrite(out_file, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train",
                        help="Whether to train a new model or run inference")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Path to saved model weights for inference")
    args = parser.parse_args()

    root_dir = "complete_dataset"
    images_dir = os.path.join(root_dir, "chessred2k")

    train_dataset = ChessDataset(root_dir, images_dir, 'train', data_aug)
    valid_dataset = ChessDataset(root_dir, images_dir, 'valid', data_in)
    test_dataset = ChessDataset(root_dir, images_dir, 'test', data_in)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    batch_size = 16
    num_workers = 2

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    # experiment(train_dataloader)

    backbone = nn.Sequential(*list(
            models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-2])  # → (B, 2048, 7, 7)
    model = ChessboardPredictor(backbone, in_channels=2048).to(device)

    if args.mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        model = train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, device, epochs=1)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "infer":
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        VISUALISATION_DIR = "digital_twin_visualisations"
        # Inference loop on test data
        with torch.no_grad():
            all_preds = []
            all_labels = []
            os.makedirs(VISUALISATION_DIR, exist_ok=True)
            img_count = 0
            with torch.no_grad():
                for images, _, boards, __ in test_dataloader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu()

                    for i in range(images.size(0)):
                        visualise_sample(
                            images[i],                     # tensor
                            preds[i].numpy(),              # predicted board  (8x8)
                            boards[i].numpy(),             # ground-truth board
                            os.path.join(VISUALISATION_DIR, f"viz_{img_count:04d}.png")
                        )
                        img_count += 1
                        if img_count >= 50:                # limit number of visualizations
                            break
                    if img_count >= 50:
                        break

        print(f"Inference complete, visualisations available at: {VISUALISATION_DIR}")

if __name__ == "__main__":
    main()
