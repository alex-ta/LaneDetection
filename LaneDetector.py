import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
import numpy as np
import cv2 as cv
from sklearn import preprocessing

class LaneDetector:
	def __init__(self, model_file, weight_file, lane=1, printOut = 0):
		self.lane = lane
		#load model
		json_file = open(model_file, 'r')
		loaded_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_json)
		# load weights
		model.load_weights(weight_file)
		# compile model
		model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
		if printOut:
			print("Loaded model as " + model_name +" & " + weights_name)
		self.model = model

	def _predict(self, imgs, one_hot = 1):
		# calculate predictions
		prediction = model.predict(imgs)
		return prediction
		
		
	def predictLane(self, grid):
		images = grid.grid;
		xCount = -1
		for row in images:
			xCount = xCount + 1 
			yCount = -1
			for img in row:
				#cv.imshow("img"+str(xCount*yCount),np.array(img))
				#print(len(images))
				#print(len(images[0]))
				#print(len(row))
				#print(len(row[0]))
				#print(len(img))
				#print(len(img[0]))
				print(grid.xCount)
				print(grid.yCount)
				yCount = yCount + 1
				p = self.model.predict(np.array([img]))
				print(xCount)
				print(yCount)
				print("x, y")
				print(len(p))
				if p == 1:
					grid.markField(xCount, yCount)
		return grid.getMarkedImage()
				