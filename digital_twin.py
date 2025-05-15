import matplotlib.pyplot as plt, numpy as np, os, torch, random, cv2, json
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2 as transforms
from sklearn.metrics import accuracy_score, f1_score
from torchsummary import summary

random.seed(42)

# Normalize images
data_aug = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_in = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def chesspos2number(chesspos):
    col = ord(chesspos[0])-ord('a')
    row = int(chesspos[1])-1
    return row, col

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
        self.boards = torch.zeros((len(self.file_names), 8, 8))

        for piece in self.anns['annotations']['pieces']:
            idx = np.where(self.ids == piece['image_id'])[0][0]
            row, col = chesspos2number(piece['chessboard_position'])
            self.occupancy_boards[idx][row][col] = 1
            self.boards[idx][row][col] = piece["category_id"] + 1   # 0 means empty, [1,12] means a specific piece

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
        self.num_pieces = torch.sum(self.occupancy_boards.view(len(self.occupancy_boards), 64), axis=-1)
        self.num_pieces = F.one_hot(self.num_pieces.long()-1, 32)
        self.ids = self.ids[self.split_ids]

        self.transform = transform
        print(f"Number of {partition} images: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        image = cv2.imread(os.path.join(self.images_dir, self.file_names[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        num_pieces = self.num_pieces[i]
        board = self.boards[i]
        # occupancy_board = self.occupancy_boards[i]

        return image, num_pieces.float(), board

class ChessboardPredictor(nn.Module):
    def __init__(self, backbone, in_channels):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
            nn.Conv2d(512, 13, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# class ChessboardPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Load ResNet backbone
#         backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

#         # Use layers up to the last conv block
#         self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Output: (B, 2048, 7, 7)

#         # Custom head: Upsample to 8x8 and map to 13 channels
#         self.head = nn.Sequential(
#             nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # (B, 512, 7, 7)
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),

#             nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),  # (B, 512, 8, 8)

#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 13, kernel_size=1),  # (B, 13, 8, 8)
#         )

    # def forward(self, x):
    #     features = self.backbone(x)   # -> (B, 2048, 7, 7)
    #     out = self.head(features)     # -> (B, 13, 8, 8)
    #     return out

def epoch_iter(model, dataloader, optimizer=None, is_training=True, device='cuda'):
    model.train() if is_training else model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    for images, _, boards in dataloader:
        images = images.to(device)               # (B, 3, 224, 224)
        targets = boards.long().to(device)       # (B, 8, 8)

        if is_training:
            optimizer.zero_grad()

        outputs = model(images)                  # (B, 13, 8, 8)
        loss = F.cross_entropy(outputs, targets)

        if is_training:
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

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = epoch_iter(model, train_loader, optimizer=optimizer, is_training=True, device=device)

        # Validate
        val_loss, val_acc = epoch_iter(model, valid_loader, is_training=False, device=device)

        print(f"Epoch {epoch+1}/{epochs} | "
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


root_dir = "complete_dataset"
images_dir = os.path.join(root_dir, "chessred2k")

train_dataset = ChessDataset(root_dir, images_dir, 'train', data_aug)
valid_dataset = ChessDataset(root_dir, images_dir, 'valid', data_in)
test_dataset = ChessDataset(root_dir, images_dir, 'test', data_in)

# get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# now we need to define a Dataloader, which allows us to automatically batch our inputs, do sampling and multiprocess data loading
batch_size = 16
num_workers = 2 # how many processes are used to load the data

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

model = models.mobilenet_v3_small(weights="DEFAULT")
net = ChessboardPredictor(model.features, in_channels=576).to(device)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

criterion = nn.CrossEntropyLoss()

model = train_model(net, train_dataloader, valid_dataloader, optimizer, scheduler, device, epochs=20)
save_path = "first_model" + ".pth"
torch.save(model.state_dict(), save_path)
