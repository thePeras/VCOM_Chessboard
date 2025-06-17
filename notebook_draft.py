import torch
import torch.nn as nn
from torchvision import models, transforms


"""
The main goals of task 3 are to find the bounding boxes of chess pieces, as well as to extract the digital twin of the chessboard.
"""

# Example digital twin image
# Motivation for the task: what we want to do here
import chess
import chess.svg
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import SVG, display
import io
import cairosvg

fen_string = "r2q1rk1/ppp1bppp/2p1b3/3n4/2N5/2NP3P/PPPB1PP1/R2Q2KR"
board = chess.Board(fen_string)
fen_svg = chess.svg.board(board=board)
png_data = cairosvg.svg2png(bytestring=fen_svg)
fen_image = Image.open(io.BytesIO(png_data))
actual_image = Image.open("dataset/images/56/G056_IMG023.jpg")
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(actual_image)
axs[0].set_title("Actual Image")
axs[0].axis("off")
axs[1].imshow(fen_image)
axs[1].set_title("FEN Board")
axs[1].axis("off")
plt.tight_layout()
plt.show()


"""
So, its clear the need of object detection and classification to identify the chess pieces.
For this task, we rely on YOLO architecture and some different variants (Yolov8n, Yolov8x, Yolo11s).

The easiest way to use YOLO is through the Ultralytics package, which provides a simple interface for training and inference.
With this library, we can train a YOLO model on our dataset by just prompting some parameters.
"""

from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Training the model
model.train(data='path/to/dataset.yaml', epochs=100, imgsz=640)

# Predicting on an image
results = model('path/to/image.jpg')

"""
However the dataset should be structured in a way that ultralytics can understand, typically with images, corresponding label files (txt), splits files and .yaml file defining the dataset structure.
We easily achive this by creating a python script to structure the dataset.
(...TODO: improve)
"""


"""
With the dataset ready, we train various YOLO models and compare their performance. All the models were trained with 100 epochs.
Weirdly, in all the models, we found a breakthrough epoch (around 90) where the model abruptly decrease its loss value.
"""

"""
Yolov8n

The smallest and fastest model in the YOLOv8 family, designed for real-time applications with lower computational resources.
We trained this model freezing the backbone (10 initial layers), using batch size of 64 and an image size of 940 pixels.
A big image size is important to capture the chess pieces details, since they are small in the image.

"""

model = YOLO('yolov8n.pt')
model.train(
    data='/kaggle/input/chessboard-cv/data.yaml',
    epochs=100,
    imgsz=940,
    batch=64,
    freeze=10,
    name='myYolov8n'
)


"""
Yolov8x

Is larger and more powerful model in the YOLOv8 family, designed for high accuracy tasks.
Because of its size, we reduce the batch size to 16 and image size to 640. For this model, we did not freeze any layers, allowing the model adapt its weights to the dataset.
"""

model = YOLO('yolov8x.pt')
model.train(
    data='/kaggle/input/chessboard-cv/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='myYolov8x'
)


"""
Yolo11s

Version 11 of YOLO is more recent and includes significant improvements in architecture and performance.
We trained this model with an image size of 940 pixels and batch size of 16, because v11 is has a bigger architecture than the v8 and high batch size would lead to out of memory errors.
"""

model = YOLO('yolo11s.pt')
model.train(
    data='/kaggle/input/chessboard-cv/data.yaml',
    epochs=100,
    imgsz=940,
    batch=16,
    name='myYolo11n'
)


"""
Comparative Analysis

We evaluate the models using mAP50 and mAP50-95 for bounding boxes, as well a confusion matrix for the classes.
"""

"""
Comparative Analysis - Conclusions

(...TODO IMAGES)

Based on the results, yolov8x model outperforms the others in mAP50-95 metric, so its the choice for the next steps.
"""


"""
## Digital Twin Pipeline
First, for this task, we followed the approach in the ChessReD paper of using end-to-end board recognition.
We trained with cross-entropy loss, with classification of each board square (fully connected layer).
Initially, we also attempted to do a kind of semantic segmentation with the board squares, but it failed to work with a decent performance.
This approach, however, the results were poor, with decent performance only in common chess positions.

The loss curve and a few examples that demonstrate this performance are shown here:

# TODO: add plots and eaxmples

Because of this lacking performance, we decided to leverage the YOLO model for the predictions, since it had great performance in finding piece bounding boxes.
"""

"""
With bounding box detection and piece classification in place, we can construct a digital twin of the chessboard.
This involves mapping detected pieces to their corresponding positions on the chess grid and converting this layout into Forsyth-Edwards Notation (FEN).
This process is based in a key assumption: the central pixel of the bounding box's bottom half will always be mapped to the correct square.
"""

def map_pieces_to_board(yolo_result, intersections, warp_matrix, fen_map):
    board_matrix = [["*" for _ in range(8)] for _ in range(8)]
    if not intersections:
        print("No intersections provided, cannot map pieces.")
        return board_matrix

    piece_boxes = yolo_result.boxes.xyxy.cpu().numpy()
    piece_classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_result.names

    grid_side_len = int(math.sqrt(len(intersections)))

    for i in range(len(piece_boxes)):
        box = piece_boxes[i]
        x1, y1, x2, y2 = box
        
        xb = (x1 + x2) / 2
        yb = y1 * 0.1 + y2 * 0.9  

        point_to_transform = np.array([[[xb, yb]]], dtype=np.float32)
        
        transformed_point = cv2.perspectiveTransform(point_to_transform, warp_matrix)
        tx, ty = transformed_point[0][0]
        

        found_square = False
        for r in range(8):
            for c in range(8):
                top_left_idx = r * grid_side_len + c
                top_right_idx = top_left_idx + 1
                bottom_left_idx = (r + 1) * grid_side_len + c
                
                if (intersections[top_left_idx][0] < tx < intersections[top_right_idx][0] and
                    intersections[top_left_idx][1] < ty < intersections[bottom_left_idx][1]):
                    
                    class_name = class_names[piece_classes[i]]
                    fen_char = fen_map.get(class_name)
                    
                    if fen_char:
                        board_matrix[r][c] = fen_char
                    
                    found_square = True
                    break
            if found_square:
                break
    
    return board_matrix


"""
To evaluate the accuracy of the generated FEN, we compare it to a pre-annotated ground truth FEN using the Levenshtein edit distance. 
As shown in the process_fen function, both FEN strings are first expanded into 64-character representations, where each character corresponds to a square on the board. 
The edit distance between these strings provides a count of incorrect, missing, or misplaced pieces.
"""
from editdistance import eval as edit_distance

FEN_MAP = {
    "white-pawn": "P", "white-rock": "R", "white-knight": "N", "white-bishop": "B", "white-queen": "Q", "white-king": "K", 
    "black-pawn": "p", "black-rock": "r", "black-knight": "n", "black-bishop": "b", "black-queen": "q", "black-king": "k",
}

def matrix_to_fen(board_matrix):
    fen_rows = []
    for row in board_matrix:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == "*":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def get_ground_truth_fen(filepath):
    with open('annotations_fen.json', 'r') as f:
        data = json.load(f)
    return data.get(str(filepath.split("/")[-1]), "")

def process_fen(board_matrix, filename):
    pred_fen = matrix_to_fen(board_matrix)
    exp_fen = get_ground_truth_fen(filename).split(' ')[0] 
    pred_string = "".join("".join(row) for row in board_matrix)
    
    # Create comparable string from expected FEN
    exp_string_full = ""
    for char in exp_fen.replace('/', ''):
        if char.isdigit():
            exp_string_full += '*' * int(char)
        else:
            exp_string_full += char
    
    edit_dist = edit_distance(pred_string, exp_string_full)
    return {"pred_fen": pred_fen, "exp_fen": exp_fen, "edit_dist": edit_dist}


"""
### 3. Approaches to Board Recognition

#TODO: global image enumration here

#### Approach A: Traditional(Canny Edge + Polygon Approx + Feature Matching) + YOLO for piece detection
(TODO: image a)

Reusing code from the fist task, we detect both corners and board orientation using traditional computer vision techniques.
"""

"""
1. Corners
The board corners are detected by classic segmentation techinques and line detection algorithms.
More information about this can be found in the first task report.

2. Orientation
For the orientation, the horse is used as a reference point. 
We try to find it by matching features between the 4 corners of the chessboard image and a template image of a horse.
The corner with the highest number of matches is considered the horse corner, and the board is rotated accordingly.

Later, we found out this approach is not robust enough, as it fails when the horse is occluded by pieces.
"""

def find_orientation_by_horse(image):
    horse_path = "figures/horse.png"
    horse_img = cv2.imread(horse_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    corner_size = min(width, height) // 4
    
    corners = {
        "top_left": (image[:corner_size, :corner_size], cv2.rotate(horse_img, cv2.ROTATE_90_CLOCKWISE)),
        "top_right": (image[:corner_size, width-corner_size:], cv2.rotate(horse_img, cv2.ROTATE_180)),
        "bottom_left": (image[height-corner_size:, :corner_size], horse_img),
        "bottom_right": (image[height-corner_size:, width-corner_size:], cv2.rotate(horse_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    }
    
    best_score = -1
    best_match_loc = None
    best_rotation = None
    
    for corner_name, (corner_img, horse_template) in corners.items():
        target_size = corner_size // 4
        resized_template = cv2.resize(horse_template, (target_size, target_size))
        
        if corner_img.shape[0] < resized_template.shape[0] or corner_img.shape[1] < resized_template.shape[1]:
            continue
            
        result = cv2.matchTemplate(corner_img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            
            if corner_name == "top_left":
                match_x, match_y = max_loc
                best_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif corner_name == "top_right":
                match_x, match_y = width - corner_size + max_loc[0], max_loc[1]
                best_rotation = cv2.ROTATE_180
            elif corner_name == "bottom_left":
                match_x, match_y = max_loc[0], height - corner_size + max_loc[1]
                best_rotation = None
            elif corner_name == "bottom_right":
                match_x, match_y = width - corner_size + max_loc[0], height - corner_size + max_loc[1]
                best_rotation = cv2.ROTATE_90_CLOCKWISE
                
            best_match_loc = (match_x, match_y)

    return best_rotation, best_match_loc

"""
#### Approach B: Yolo Keypoint Detection + YOLO for piece detection
(TODO: image b)

In this approach, we tried to use YOLO keypoint detection to find the corners of the chessboard.
We trained a YOLO model to detect the four corners of the chessboard as keypoints with labels "top_left", "top_right", "bottom_left" and "bottom_right".
This way, not only the model give us the points of the corners, but also the orientation of the chessboard.

This approach was not successful, as the model struggled to learn the keypoints, leading to poor corner detection.
"""

model = YOLO("yolo11s-pose.pt")
model.train(
    data='/kaggle/input/chessboard-corner-detect/data_corners_kp.yaml',
    epochs=100,
    imgsz=640,
    batch=64,
)

"""
#### Approach C: CNN (Regression) + YOLO for piece detection
Since our first approach had lacking results, we moved to a simpler strategy:
using a pre-trained CNN to regress the four corner coordinates of the chessboard.

For our first attempt, we froze the backbone of a ResNet50, and added a few layers to predict output eight values: (x1, y1, x2, y2, x3, y3, x4, y4).
To train the model, we normalized the labels to be within [0, 1] (given the image size).

We trained using MSE as our loss function, a sigmoid activation on the output layer and the Adam optimizer.
After 50 epochs, the corner predictions were still unreliable (quite similar results to the pose estimation setup).
The mean distance from the corners to the predictions obtained were TODO.

TODO: image
"""

"""
We improved this basic approach by choosing a better backbone, EfficientNetV2-S (like in Task 2).
We fine-tuned of all its layers.

More crucially, we added manual data augmentations to improve generalization, since the dataset's size is limited in regards to corner labels.

Introducing augmentations requires transforming each keypoint's coordinates according to the augmentations used.
Note that we can only use manual augmentations and not automatic ones, since they don't support transforming labels.

We used torchvision 0.23, which supports this feature, even though it is not yet a stable version.

The complete code is available in the main.py file, runnable by specifying the argument `--type corners`
"""
def load_image(image_file: str):
    # Load with about 750x750 (3000 / 4)
    image = cv2.imread(image_file, cv2.IMREAD_REDUCED_COLOR_4)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class ChessCornersDataset(Dataset):
    # A different dataset class because augmentations for corners require special care here
    # We use torchvision 0.23 (currently not stable) to be able to use tv_tensors.KeyPoints
    def __init__(self, root_dir, images_dir, partition, transform, use_2k_dataset=False):
        self.anns = json.load(open(os.path.join(root_dir, 'annotations.json')))
        self.categories = [c['name'] for c in self.anns['categories']]
        self.root = root_dir
        self.images_dir = images_dir
        self.ids = []
        self.file_names = []
        self.original_widths = []
        self.original_heights = []

        for x in self.anns['images']:
            self.file_names.append(x['path'])
            self.ids.append(x['id'])
            self.original_widths.append(x['width'])
            self.original_heights.append(x['height'])

        self.file_names = np.asarray(self.file_names)
        self.original_heights = np.asarray(self.original_heights)
        self.original_widths = np.asarray(self.original_widths)
        self.ids = np.asarray(self.ids)

        self.corners = torch.zeros((len(self.file_names), 4, 2), dtype=torch.float32)
        for corner_ann in self.anns['annotations']['corners']:
            corner_data = corner_ann['corners']
            for i, (corner_label, xyvalues) in enumerate(sorted(corner_data.items())):
                for coord in range(2):
                    self.corners[corner_ann['image_id']][i][coord] = xyvalues[coord]

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
        self.corners = self.corners[self.split_ids]
        self.original_heights = self.original_heights[self.split_ids]
        self.original_widths = self.original_widths[self.split_ids]
        self.ids = self.ids[self.split_ids]

        self.transform = transform
        print(f"Number of {partition} images: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        image = load_image(os.path.join(self.images_dir, self.file_names[i]))

        original_width = self.original_widths[i]
        original_height = self.original_heights[i]
        h, w, _ = image.shape
        sx = w / original_width
        sy = h / original_height

        corners_scaled = self.corners[i].clone()
        corners_scaled[:, 0] *= sx
        corners_scaled[:, 1] *= sy

        image_chw = image.transpose((2, 0, 1))
        img_tv = tv_tensors.Image(image_chw)
        kps_tv = tv_tensors.KeyPoints(corners_scaled.unsqueeze(0), canvas_size=(h, w))  # requires torchvision 0.23

        img_transformed, kps_transformed = self.transform(img_tv, kps_tv)

        # Normalize to [0-1]
        _, final_h, final_w = img_transformed.shape
        kps_transformed[:,:,0] /= final_w
        kps_transformed[:,:,1] /= final_h

        final_corners = kps_transformed.squeeze(0)

        return img_transformed, final_corners

def calculate_corners_metric(all_preds, all_labels):
    # Mean Euclidean Distance between predicted and true corners
    # all_preds and all_labels are (N, 8) where N is batch_size, 8 is (x1,y1,x2,y2,x3,y3,x4,y4)
    # Reshape to (N, 4, 2)
    all_preds_reshaped = all_preds.view(-1, 4, 2)
    all_labels_reshaped = all_labels.view(-1, 4, 2)

    # Calculate Euclidean distance for each corner and then average
    distances = torch.sqrt(torch.sum((all_preds_reshaped - all_labels_reshaped)**2, dim=-1))
    mean_distance_per_corner = torch.mean(distances)
    return mean_distance_per_corner.item()

class CornersPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # should predict: 4 corners * 2 coordinates (x, y)

    @staticmethod
    def create_efficient_net(pretrained: bool = True):
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        conv_model = models.efficientnet_v2_s(weights=weights)
        conv_model.classifier[1] = nn.Linear(conv_model.classifier[1].in_features, out_features=8)
        return CornersPredictor(conv_model)

    def forward(self, x):
        corners = self.model(x)
        corners = torch.sigmoid(corners)
        return corners

def train_model_corners(
    # ...
):
    # Classic training function
    # ...
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"

## Training process (inside a train function)
model = CornersPredictor.create_efficient_net(pretrained=True).to(device)
epochs = 200

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

train_dataloader = ...
valid_dataloader = ...
model = train_model_corners(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    loss_fn,
    scheduler,
    device,
    epochs=epochs,
)
"""

The results improved with a mean distance of the predicted points to the corners of \fixme.
# TODO: image

#### Approach D: UNet (Segmentation) + ResNet(Classication) + Yolo for piece detection
(TODO: image d)

In this approach we try to use segmentation as the way to extract the board corners. 
We devoped the necessary dataset and trained a U-Net model to segment the chessboard.
The mask is then used to approximate to a 4 corners polygon and extract the corners.
"""

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        return torch.sigmoid(self.conv_last(d1))


"""
Because segmentation give us no clue about the orientation of the chessboard, we use a ResNet base model to classify the board orientation.
The orientation was treated as a classification problem with 4 classes: 0, 90, 180 and 270 degrees, mapped as 0, 1, 2 and 3.

Note: The annotations corners give us the internal board corners, which have no information about the horse position.
Because we believe this is good information for the model, we give him the image with a sufficient margin around the chessboard, so the horse is always present in the image.
"""

cl_model = models.resnet18(pretrained=True)
cl_model.fc = nn.Linear(cl_model.fc.in_features, 4)

# TODO: Imagem de uma chessboard sem cavalo e outra com cavalo


"""
#### Approach E: UNet (Heatmap) + Yolo for piece detection

After struggling with the reliability of the other corner detection methods, we decided to try a different approach.

The key idea was to predict heatmaps rather than directly predicting corner coordinates. 
A heatmap represents the probability distribution of where each corner might be located, brighter pixels indicate higher confidence that a corner exists at that position. 
This approach is more robust because instead of forcing the model to output exact coordinates, it can express uncertainty and handle ambiguous cases.

We trained a U-Net model to output four separate heatmaps, one for each inner corner of the chessboard. 
The encoder uses transfer learning from a pretrained ResNet34 (trained on ImageNet), while the decoder was trained from scratch. 
The brightest point in each heatmap gives us the predicted corner location.
(TODO: image E)

##### Model Architecture
"""
class UNetWithResnetEncoder(nn.Module):
    """A U-Net model using a pretrained ResNet34 as the encoder for corner detection."""
    def __init__(self, n_class=4, pretrained=False): 
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x); enc2 = self.encoder2(enc1); enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3); enc5 = self.encoder5(enc4)
        up4 = self.upconv4(enc5); dec4 = self.decoder4(torch.cat([up4, enc4], 1))
        up3 = self.upconv3(dec4); dec3 = self.decoder3(torch.cat([up3, enc3], 1))
        up2 = self.upconv2(dec3); dec2 = self.decoder2(torch.cat([up2, enc2], 1))
        return self.final_conv(dec2)
        
"""

The UNet-based heatmap approach demonstrated significantly improved corner detection accuracy compared to our previous methods. 
The model successfully learned to generate the heatmaps, with peak activations consistently aligning with true corner locations.
However, we observed some limitations in the current implementation:

- Corner Label Confusion: While the heatmaps themselves were generally accurate, the model occasionally struggled with corner identity assignment (e.g., confusing top-left with bottom-right corners).

(TODO: Show the image with the corner confusion)

- Sample Complexity: Our relatively small dataset of 2,000 annotated images may have limited the model's ability to generalize across diverse chessboard orientations and lighting conditions.

## Future Improvements

In the future we could try several improvements to address these limitations such as:

- Data augmentation to improve sample efficiency
- Orientation detection: Incorporating explicit orientation estimation or feature matching algorithms to resolve corner labeling ambiguities
"""

"""
Conclusions

The chosen metrics for the evaluation were essential to create achieve more robust digital twin extraction, since it allows us to identify clear issues in the pipeline.
For example,
- edit distance = 1: Most of the times this indicates that the YOLO model made a classification error and a unique piece was wrong classified. For example, pieces that are hidden or occluded can lead to such errors. This observation means the pipeline is always limited by the performance of the YOLO model, which is not a big issue since its performance is already quite good as shown in the comparative analysis.
- edit distance > 40: This tells us a brutal discrepancy between the ground truth and the predicted board, which is usually caused early on the pipeline, such as the detection of corners. By wrongly detecting the corners of the chessboard, the mapping of the bounding boxes to the chess grid is affected, leading to a completely wrong FEN string.
- edit distance > 10: This is usually caused by the incorrect orientation of the chessboard. Because the FEN string expects the chessboard to be oriented in a specific way, any wrong rotation will lead to a significant difference in the FEN string. So, it is important not only to detect the pieces and the chessboard corners, but also the orientation of the chessboard.

"""