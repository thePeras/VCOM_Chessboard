# Chessboard Computer Vision

## Task 1

#### Input:
- Image with chessboard

#### Output:
- Total number of black/white pieces on the board
- Position of the pieces on the image (bounding boxes)
- Position of the pieces on the board (8x8 matrix with 0/1 values representing absence/presence of piece - any board orientation is acceptable) 

#### Dataset:
- 50 images randomly chosen from a public dataset
https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2
- The results will be tested in 10 undisclosed images

#### Deliverables:
- Short report (2 pages max) presenting the methodology and some results
- Python script (only one file)

#### Evaluation:
- Accounts for 30% of the overall project grade
- Elements being considered: methodology, report and quality of the results

#### Important remarks
- Follow strictly the JSON structure for the input and output files
- It is okay to use AI tools while developing your work, but it is not okay to use them without acknowledging it
- All members of the group are expected to understand the methodology and the submitted code

## Task 2
Count the number of pieces on a chessboard using a CNN-based architecture
*Extra*: Quantitative comparison (with adequate metrics) different architectures

#### Input:
- Image containing a game of chess

#### Output:
- Total number of pieces within he chess board

#### Solutions:

1. Regression-based approach
2. 

## Task 3

### Task 3.1
Chess Pieces Detection:
- At least one model, e.g. YOLO, Faster R-CNN
*Extra*: Quantitative comparison (with adequate metrics) of different architectures

#### Solutions:
- YOLO, different variants
- Faster R-CNN

#### Adequate Metrics:
- mAP (mean Average Precision)
- different tresholds?

### Task 3.2
Board "Digital Twin":
Identify the board status, i.e. where each piece (incl. colour and type) is in the board
- One model, e.g. detection + traditional methods from task 1
Qualitative evaluation with some (good and bad) results are enough

Forsyth-Edwards Notation (FEN)

#### Solutions:
- Use bounding boxes to detect pieces location (using homographies calculated in task1)
- 13x64 classes, image tagging and classification
- YOLO inspired 8x8 grid

#### Adequate Metrics:
- occupancy grid
- F1 score?

Warning:
Save all the configuration / information to don't run the model again