import grid
import cv2 as cv
import numpy as np

img = cv.imread("index.jpg");
grid = grid.Grid(img)
#cv.imshow(grid.getGrid()[0])
print(type(grid.getGridCount(5,5)[3][3]))
cv.imshow("img",grid.getGridCount(5,5)[0][0])
cv.imshow("img2",img)
grid.markField(0,0)
cv.imshow("img4",grid.getMarkedGrid()[0][0])
cv.imshow("img6",grid.getMarkedImage())
cv.waitKey()
