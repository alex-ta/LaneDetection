import grid.grid as grid
import LaneDetector as detect
import cv2 as cv

img = cv.imread("img.jpg")
grid = grid.Grid(img)
grid.getGridShape(60,60)
detector = detect.LaneDetector("save_data/model.json","save_data/model.h5")
img = detector.predictLane(grid);
cv.imshow("image",img)
cv.waitKey(0);