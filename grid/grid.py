import numpy as np
import cv2 as cv

class Grid :
	def __init__(self, img):
		self.img = img;
		self.height, self.width, self.channels = img.shape
		self.xWidth = self.width
		self.yHeight = self.height
		self.xCount = self.yCount = 1
		self.grid = []
		
	def getGridCount(self,xCount,yCount):
		self.xWidth = self.width/xCount
		self.yHeight = self.height/yCount
		self.xCount = xCount;
		self.yCount = yCount;
		return self.getGrid();

	def getGridShape(self,xWidth,yHeight):
		self.xWidth = xWidth
		self.yHeight = yHeight
		self.xCount = self.width/xWidth
		self.yCount = self.height/yHeight
		return self.getGrid();

	def getGridPercent(self,xpercent,ypercent):
		self.xWidth = self.width*xpercent
		self.yHeight = self.height*ypercent
		self.xCount = self.width / self.xWidth
		self.yCount = self.height / self.yHeight
		return self.getGrid();

	def getGrid(self):
		'''
		x1y1x2y1x3y1

		x1y2x2y2x3y2

		x1y3x2y3x3y3

		'''
		self.grid = []
		for y in range(self.yCount):
			row = []
			for x in range(self.xCount):
				yH = y*self.yHeight
				xW = x*self.xWidth
				row.append(self.img[int(yH):int(yH+self.yHeight), int(xW):int(xW+self.xWidth)])
			self.grid.append(row)
		return self.grid

	def markField(self, xPos, yPos, r=100, g=0, b=0):
		mask = np.zeros((int(self.yHeight), int(self.xWidth), 3), dtype="uint8")
		mask[:,:] = (b,g,r)
		img = self.grid[xPos][yPos];
		self.grid[xPos][yPos] = cv.addWeighted(img,0.7,mask,0.3,0)

	def getMarkedGrid(self):
		return self.grid

	def getMarkedImage(self):
		parts = []
		for row in self.grid:
			print(len(row))
			parts.append(np.concatenate(row, axis=1))
		print(len(parts))
		return np.concatenate(parts, axis=0)
		
		
		
		