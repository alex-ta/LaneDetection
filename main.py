import grid.grid as grid
import LaneDetector as detect
import cv2 as cv

img = cv.imread("arial_0.png")
grid = grid.Grid(img)
grid.getGridCount(25,25)
detector = detect.LaneDetector("model.py","model.h5")
img = detector.predictLane(grid);
cv.imshow(img)