"""
Script for training and running computer vision models, mainly for task 2 (number of pieces in a chessboard).

It also contains a few functions to train a model end-to-end to predict the chessboard (digital twin),
but this resulted in bad performances. It was not used as the main part of the digital twin task.

Required packages: matplotlib, numpy, torch, torchvision, scikit-learn, optuna


To run, assuming that best_model.pth is in the same directory as the source code, just do:
`python3 main.py`

To check what arguments you can pass, run `python3 main.py --help`
In addition, the images are assumed to be saved in their original sizes (about 3000x3000).
"""

import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json, argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2 as transforms
from sklearn.metrics import accuracy_score, f1_score
from torchsummary import summary
from board_draw import render_board_from_matrix
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.widgets import Button
import optuna

from typing import Optional
import pickle
from copy import deepcopy

random.seed(42)

# manual augmentations (not as great performance as RandAugment): not used anymore
manual_data_aug = transforms.Compose([
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

    transforms.Resize((384, 384)),  # modify according to the used model (depends on the used pretraining settings)
    transforms.CenterCrop((384, 384)),

    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# "automatic" data augmentation
# replaces the manual jitter/rotate/etc.
data_aug = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.AutoAugment(   # can use this instead of RandAugment (leads to worse performance)
    #     policy=transforms.AutoAugmentPolicy.IMAGENET
    # ),
    transforms.RandAugment(     
        num_ops=2,      # how many transforms to apply per image
        magnitude=9     # overall strength (0-10)
    ),
    transforms.Resize((384, 384)),
    transforms.CenterCrop((384, 384)),

    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

data_in = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((384, 384)),      # modify according to model used
    transforms.CenterCrop((384, 384)),
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
    def __init__(self, root_dir, images_dir, partition, transform=None, use_2k_dataset=False):
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

        splits = self.anns["splits"]["chessred2k"] if use_2k_dataset else self.anns["splits"]
        if partition == 'train':
            self.split_ids = np.asarray(splits['train']['image_ids']).astype(int)
        elif partition == 'valid':
            self.split_ids = np.asarray(splits['val']['image_ids']).astype(int)
        else:
            self.split_ids = np.asarray(splits['test']['image_ids']).astype(int)

        intersect = np.isin(self.ids, self.split_ids)
        self.split_ids = np.where(intersect)[0]
        self.file_names = self.file_names[self.split_ids]
        self.occupancy_boards = self.occupancy_boards[self.split_ids]
        self.boards = self.boards[self.split_ids]
        self.num_pieces = torch.sum(self.occupancy_boards.view(len(self.occupancy_boards), 64), axis=-1)
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

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_channels, 8*8*13),
        )

    def forward(self, x):
        x = self.backbone(x)
        flat_x = self.layers(x)
        batch_size = flat_x.shape[0]
        x = flat_x.view(batch_size, 13, 8, 8)
        return x

def get_chessboard_predictor_targets(batch):
    return batch[2].long()

def get_num_pieces_predictor_targets(batch):
    return batch[1]

def calculate_accuracy(all_preds, all_labels):
    return (all_preds == all_labels).float().mean().item()

def calculate_mae(all_preds, all_labels):
    return mean_absolute_error(all_labels, all_preds.detach().numpy())

def get_preds_chessboard(outputs):
    preds = outputs.argmax(dim=1)
    return preds

class NumPiecesPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activation = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor(1.0))    # Learnable scaling and bias for ReLU
        self.bias = nn.Parameter(torch.tensor(2.0))

    # Check https://docs.pytorch.org/vision/main/models.html for specific models and configs (e.g. pre-training image size)
    @staticmethod
    def create_efficient_net():
        conv_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        conv_model.classifier[1] = nn.Linear(conv_model.classifier[1].in_features, out_features=1)
        return NumPiecesPredictor(conv_model)

    @staticmethod
    def create_swin():
        conv_model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
        conv_model.head = nn.Linear(in_features=conv_model.head.in_features, out_features=1)
        return NumPiecesPredictor(conv_model)
    
    @staticmethod
    def create_resnet():
        conv_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        conv_model.fc = nn.Linear(in_features=conv_model.fc.in_features, out_features=1)
        return NumPiecesPredictor(conv_model)

    @staticmethod
    def create_resnext():
        conv_model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
        conv_model.fc = nn.Linear(conv_model.fc.in_features, out_features=1)
        return NumPiecesPredictor(conv_model)
    
    def _forward_sigmoid(self, x):  # initial sigmoid setup
        x = self.model(x)
        x = torch.sigmoid(x)          # (B, 1): [0, 1]
        x = x * 30 + 2                # Scale to (2, 32)
        return x.squeeze(1)           # Final shape: (B,)

    def _forward_relu(self, x):     # basic ReLU setup
        x = self.model(x)
        x = self.activation(x)
        return x.squeeze(1)

    def _learnable_forward_relu(self, x):    # The most promising results
        x = self.model(x)
        x = self.activation(x)
        x = x * self.scale + self.bias  # scale it given learnable parameters
        return x.squeeze(1)

    def forward(self, x):
        return self._learnable_forward_relu(x)

def epoch_iter(
    model,
    dataloader,
    loss_fn,
    get_targets_fn,
    calculate_metric_fn,
    get_preds_fn=None,
    optimizer=None,
    is_training=True,
    device='cuda',
):
    model.train() if is_training else model.eval()
    context = torch.enable_grad() if is_training else torch.no_grad()
    total_loss = 0
    all_preds = []
    all_targets = []

    with context:
        for batch in dataloader:
            images = batch[0].to(device)
            targets = get_targets_fn(batch).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            preds = outputs.detach() if get_preds_fn is None else get_preds_fn(outputs.detach())

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric = calculate_metric_fn(all_preds, all_targets)

    return avg_loss, metric, all_preds, all_targets

def epoch_iter_num_pieces(model, dataloader, loss_fn, optimizer=None, is_training=True, device='cuda'):
    return epoch_iter(
        model,
        dataloader,
        loss_fn,
        get_num_pieces_predictor_targets,
        calculate_mae,
        optimizer=optimizer,
        is_training=is_training,
        device=device,
    )

def epoch_iter_chessboard(model, dataloader, loss_fn, optimizer=None, is_training=True, device='cuda'):
    return epoch_iter(
        model,
        dataloader,
        loss_fn,
        get_targets_fn=get_chessboard_predictor_targets,
        calculate_metric_fn=calculate_accuracy,
        get_preds_fn=get_preds_chessboard,
        optimizer=optimizer,
        is_training=is_training,
        device=device,
    )

def plot_train_history(train_values, val_values, ylabel: str, filename: str, title: Optional[str] = None):
    """
    Plots training and validation values (e.g., loss or accuracy) over epochs.
    """
    if title is None:
        title = f"Training & Validation {ylabel}"
    epochs = range(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, 'bo-', label='Training ' + ylabel)
    plt.plot(epochs, val_values, 'ro-', label='Validation ' + ylabel)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    
    plt.grid(True)
    plt.savefig(filename)

def train_model_chessboard(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler=None,
    device='cuda',
    epochs=10,
    save_dir: Optional[str] = None,
) -> nn.Module:
    best_val_acc = 0.0
    best_model_state = None

    train_losses, val_losses = [], []
    train_scores, val_scores = [], []

    loss_fn = nn.CrossEntropyLoss()
    print(">> Training chessboard (digital twin) model")
    for epoch in range(epochs):
        # Train
        train_loss, train_acc, _, __ = epoch_iter_chessboard(model, train_loader, loss_fn, optimizer=optimizer, is_training=True, device=device)

        # Validate
        val_loss, val_acc, _, __ = epoch_iter_chessboard(model, valid_loader, loss_fn, is_training=False, device=device)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_acc)
        val_scores.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    if save_dir:
        plot_train_history(train_losses, val_losses, "Loss", os.path.join(save_dir, "loss.png"))
        plot_train_history(train_scores, val_scores, "Accuracy", os.path.join(save_dir, "accuracy.png"))
        print(f"Saved training history plots in {save_dir}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def train_model_num_pieces(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_fn,
    scheduler=None,
    device='cuda',
    epochs=10,
    save_dir: Optional[str] = None,
) -> nn.Module:
    best_val_mae = float('inf')
    best_model_state = None

    train_losses, val_losses = [], []
    train_scores, val_scores = [], []

    best_val_preds, val_labels = [], []

    print(">> Training Number of Pieces model")
    for epoch in range(epochs):
        # Train
        train_loss, train_mae, _, __ = epoch_iter_num_pieces(model, train_loader, loss_fn, optimizer=optimizer, is_training=True, device=device)

        # Validate
        val_loss, val_mae, val_preds, val_labels = epoch_iter_num_pieces(model, valid_loader, loss_fn, is_training=False, device=device)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_mae)
        val_scores.append(val_mae)

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = deepcopy(model.state_dict())
            best_val_preds = val_preds
            val_labels = val_labels

        if scheduler:
            scheduler.step()

    print(f"\nBest Validation MAE: {best_val_mae:.4f}")

    if save_dir:
        save_model_results(filename=os.path.join(save_dir, f"valid"), preds=best_val_preds.numpy(), true=val_labels.numpy())
        print(f"Saved best validation results in {save_dir}")

        plot_train_history(train_losses, val_losses, "Loss", os.path.join(save_dir, "loss.png"))
        plot_train_history(train_scores, val_scores, "MAE", os.path.join(save_dir, "mae.png"))
        print(f"Saved training history plots in {save_dir}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def experiment(dataloader):
    """
    Function to experiment and see the image results of a specific dataloader.
    Mostly useful to check how the images are "coming out" of our data augmentations
    """
    for batch in dataloader:
        imgs = batch[0]
        labels = batch[1]
        boards = get_chessboard_predictor_targets(batch)
        occupancy_boards = batch[3]

        # Undo normalization
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        imgs = torch.clamp(imgs, 0, 1)  # Ensure values are in [0,1] for display

        num_images = imgs.shape[0]
        current_idx = [0]

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        img_display = ax.imshow(imgs[current_idx[0]].numpy())
        ax.axis('off')

        def update_image():
            ax.clear()
            ax.imshow(imgs[current_idx[0]].numpy())
            ax.axis('off')
            fig.canvas.draw_idle()

        def next_image(event):
            current_idx[0] = (current_idx[0] + 1) % num_images
            update_image()

        def quit_viewer(event):
            plt.close(fig)

        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        btn_next = Button(ax_next, 'Next')
        btn_next.on_clicked(next_image)

        ax_quit = plt.axes([0.81, 0.05, 0.1, 0.075])
        btn_quit = Button(ax_quit, 'Quit')
        btn_quit.on_clicked(quit_viewer)

        plt.show()
        break

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

def visualise_chessboard_sample(img_tensor, pred_board, gt_board, out_file):
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

def visualise_num_pieces_sample(img_tensor, pred_count, gt_count, out_file):
    """Display original image with predicted and ground-truth piece counts as text."""
    photo = denormalise(img_tensor)  # (H, W, 3) in RGB, float in [0, 1] or [0, 255]
    if photo.max() <= 1.0:
        photo = (photo * 255).astype(np.uint8)

    photo_bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
    h, w = photo_bgr.shape[:2]

    text_pred = f"Predicted: {pred_count:.2f}"
    text_gt   = f"Ground Truth: {gt_count:.0f}"

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color_gt   = (0, 255, 0)     # green
    color_pred = (255, 150, 150) if pred_count.round() == gt_count else (0, 0, 255) # light blue if correct, red otherwise
    text_gt_pos   = (10, 25)
    text_pred_pos = (10, 50)

    x1, y1 = 5, 5
    x2, y2 = min(170, w - 1), min(60, h - 1)
    sub_img = photo_bgr[y1:y2, x1:x2]       # sub region updated with transparent rectangle for text
    black_rect = np.zeros(sub_img.shape, dtype=photo_bgr.dtype)
    cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 0, dst=sub_img)

    cv2.putText(photo_bgr, text_gt, text_gt_pos, font, font_scale, color_gt, thickness)
    cv2.putText(photo_bgr, text_pred, text_pred_pos, font, font_scale, color_pred, thickness)
    cv2.imwrite(out_file, photo_bgr)

def visualise_preds(model, dataloader, device, viz_sample_fn, viz_dir, get_targets_fn, get_preds_fn, max_viz_cnt=50):
    os.makedirs(viz_dir, exist_ok=True)
    img_count = 0
    with torch.no_grad():
        for batch in dataloader:    # batch = images, num_pieces, board, occupancy_board
            images = batch[0].to(device)
            outputs = model(images)
            targets = get_targets_fn(batch)
            preds = (outputs if get_preds_fn is None else get_preds_fn(outputs)).cpu()

            for i in range(images.size(0)):
                viz_sample_fn(
                    images[i],
                    preds[i].numpy(),
                    targets[i].numpy(),
                    os.path.join(viz_dir, f"viz_{img_count:04d}.png"),
                )
                img_count += 1
                if img_count >= max_viz_cnt:        # limit number of visualizations
                    break
            if img_count >= max_viz_cnt:
                break

    print(f"Inference complete, visualisations available at {viz_dir}")

def get_model_results_save_dir(base_model_path: str):
    return f"results-{base_model_path}"


def save_model_results(filename: str, preds, true):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump({"preds": preds, "true": true}, f)


def main_chessboard(args, train_dataloader, valid_dataloader, test_dataloader, device):
    save_dir = get_model_results_save_dir(args.model_name)
    model_save_path = os.path.join(save_dir, args.model_name + ".pth")

    backbone = nn.Sequential(*list(
            models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-2])  # → (B, 2048, 7, 7)
    model = ChessboardPredictor(backbone, in_channels=2048).to(device)

    if args.mode == "train":
        os.makedirs(save_dir, exist_ok=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        model = train_model_chessboard(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            device,
            epochs=20,
            save_dir=save_dir,
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    elif args.mode == "infer-test":
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc, all_preds, all_labels = epoch_iter_chessboard(
            model,
            test_dataloader,
            loss_fn=loss_fn,
            is_training=False,
            device=device
        )
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        visualise_preds(
            model,
            test_dataloader,
            device,
            viz_sample_fn=visualise_chessboard_sample,
            viz_dir="digital_twin_visualisations",
            get_targets_fn=get_chessboard_predictor_targets,
            get_preds_fn=get_preds_chessboard,
        )

def objective(trial, train_dataloader, valid_dataloader, device, epochs=15):
    """
    Defines a single trial for Optuna hyperparameter tuning.
    """
    # Suggest hyperparameters for optimizer, loss, etc.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    loss_name = trial.suggest_categorical("loss", ["L1Loss", "MSELoss", "SmoothL1Loss"])

    optimizer_name = "AdamW"    # To reduce the search space, we only used AdamW
    scheduler_name = trial.suggest_categorical("scheduler", [
        "StepLR",
        "CosineAnnealingLR",
    ])

    # Initialize model, loss function and optimizer
    model = NumPiecesPredictor.create_efficient_net().to(device)
    loss_fn = {"L1Loss": nn.L1Loss(), "MSELoss": nn.MSELoss(), "SmoothL1Loss": nn.SmoothL1Loss()}[loss_name]
    optimizer = {
        "AdamW": lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
    }[optimizer_name]()

    # Conditionally define scheduler and its hyperparameters
    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 10)
        gamma = trial.suggest_float("gamma", 0.1, 0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        # T_max is the total number of epochs. No tuning needed here.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_mae = float('inf')

    for epoch in range(epochs):
        # Train
        epoch_iter_num_pieces(model, train_dataloader, loss_fn, optimizer=optimizer, is_training=True, device=device)

        # Validate
        _, val_mae, _, __ = epoch_iter_num_pieces(model, valid_dataloader, loss_fn, is_training=False, device=device)

        # Step the scheduler
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        # Report intermediate results: for the pruner
        trial.report(val_mae, epoch)

        if trial.should_prune():    # Handle pruning
            raise optuna.exceptions.TrialPruned()

    return best_val_mae

def hyperparameter_tuning_optuna(args, train_dataloader, valid_dataloader, device):
    """
    Performs hyperparameter tuning for the number of pieces model using Optuna.
    """
    print(">> Starting Hyperparameter Tuning with Optuna")

    epochs_per_trial = 15   # The number of epochs to train for each trial
    n_trials = 50   # The number of different hyperparameter combinations to test

    # this pruner was too aggressive (pruning too much)
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # only discard if a step is in the worst 25% (less aggressive pruner)
    pruner = optuna.pruners.PercentilePruner(percentile=75.0, n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.optimize(
        lambda trial: objective(trial, train_dataloader, valid_dataloader, device, epochs=epochs_per_trial),
        n_trials=n_trials
    )

    print("-" * 50)
    print("Hyperparameter Tuning Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (MAE): {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df = study.trials_dataframe()
    save_filename = "optuna_tuning_results_run4.csv"
    df.to_csv(save_filename)
    print(f"\nFull tuning results saved to '{save_filename}'")

def main_num_pieces(args, train_dataloader, valid_dataloader, test_dataloader, device):
    save_dir = get_model_results_save_dir(args.model_name)
    model_save_path = os.path.join(save_dir, args.model_name + ".pth")

    model = NumPiecesPredictor.create_efficient_net().to(device)
    loss_fn = nn.L1Loss()
    if args.mode == "train":
        os.makedirs(save_dir, exist_ok=False)
        epochs = 30

        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Best performing model (from hyperparameter tuning)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005255282016307133, weight_decay=0.03255916377235661)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model = train_model_num_pieces(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            loss_fn,
            scheduler,
            device,
            epochs=epochs,
            save_dir=save_dir,
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    elif args.mode.startswith("infer"):
        if args.mode == "infer-test":
            filename = "test"
            dataloader = test_dataloader
        elif args.mode == "infer-valid":
            filename = "valid"
            dataloader = valid_dataloader
        else:
            raise ValueError("Invalid args mode found in main_num_pieces")

        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        test_loss, test_mae, all_preds, all_labels = epoch_iter_num_pieces(
            model,
            dataloader,
            loss_fn=loss_fn,
            is_training=False,
            device=device
        )
        save_model_results(filename=os.path.join(save_dir, filename), preds=all_preds.numpy(), true=all_labels.numpy())
        print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

        visualise_preds(
            model,
            dataloader,
            device,
            viz_sample_fn=visualise_num_pieces_sample,
            viz_dir="num_pieces_visualisations",
            get_targets_fn=get_num_pieces_predictor_targets,
            get_preds_fn=None,
        )

def main():
    parser = argparse.ArgumentParser()
    # TODO: add something like is-delivery (e.g. mode "delivery")
    parser.add_argument("--mode", type=str, choices=["train", "infer-valid", "infer-test", "tune"], default="infer-test",
                        help="Whether to train a new model, run inference or perform hyperparameter tuning")
    parser.add_argument("--type", type=str, choices=["chessboard", "num-pieces"], default="num-pieces",
                        help="The type of the model to use/train")
    parser.add_argument("--model-name", type=str, default="best_model",
                        help="Name of the save model weights for inference (e.g. best_model)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="The batch size for training or inference")
    args = parser.parse_args()

    root_dir = "complete_dataset"
    images_dir = os.path.join(root_dir, "chessred")

    # For no augmentations during training, replace data_aug with data_in
    train_dataset = ChessDataset(root_dir, images_dir, 'train', data_aug)
    valid_dataset = ChessDataset(root_dir, images_dir, 'valid', data_in)
    test_dataset = ChessDataset(root_dir, images_dir, 'test', data_in)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # defaut: 16
    batch_size = args.batch_size    # as large as possible (depends on image resizes used)
    num_workers = 12

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    experiment(train_dataloader)  # uncomment to visualise train dataloader (mostly for visualizing augmentations)
    exit(0)

    if args.type == "chessboard":
        main_chessboard(args, train_dataloader, valid_dataloader, test_dataloader, device)
    elif args.type == "num-pieces":
        if args.mode == "tune":
            hyperparameter_tuning_optuna(args, train_dataloader, valid_dataloader, device)
        else:
            main_num_pieces(args, train_dataloader, valid_dataloader, test_dataloader, device)
if __name__ == "__main__":
    main()
