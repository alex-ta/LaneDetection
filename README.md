This Project was a waste product on a study of lanedetection and autonomous driving.
The Idea was to split an image into different parts and check over a neural network if it contains a street or not. (it sucks)
A better approach is a neural network with color segmentation of an incomming image like [Segnet](http://mi.eng.cam.ac.uk/projects/segnet/) there is also an python implenentation on all types of classifying image objects [Object detection](https://medium.com/weightsandbiases/car-image-segmentation-using-convolutional-neural-nets-7642448028f6). A small implementation of SegNet (less layers) did a great result for my project. If you want an easy way to detect the lines of a street you could use following Code:

```python
import  cv2
import  numpy as np
from  matplotlib  import  pyplot  as plt

sigma = 0.45
#load  img  from  file
img = cv2.imread(’1 _1920x1080.png ’)
height , width , channels = img.shape
hheight = round(height /2)

#remove  top  half
img = img[hheight:height]

#remove  not  white  color (lanemarking)
low = np.array ([240 ,240 ,240] ,  dtype = "uint16 ")
up = np.array ([255 ,255 ,255] ,  dtype = "uint16 ")
grey = cv2.inRange(img , low , up)
#canny
v = np.median(grey)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255,  (1.0 + sigma) * v))
edged = cv2.Canny(grey , lower , upper)
#average
kernel = np.ones ((5 ,5),np.float32)/25
avg = cv2.filter2D(edged ,-1,kernel)
#hough
lined = np.zeros ((hheight , width , 3), np.uint8)
minLineLength = 100
maxLineGap = 8
lines = cv2.HoughLinesP(avg ,1,np.pi/180,100, minLineLength ,maxLineGap)
for  line in  lines:
for x1 ,y1 ,x2 ,y2 in line:
cv2.line(lined ,(x1 ,y1),(x2 ,y2) ,(255 ,255 ,255) ,5)
plt.subplot (221),plt.imshow(img)
plt.title(’Original  Image ’), plt.xticks ([]), plt.yticks ([])
plt.subplot (222),plt.imshow(edged ,cmap = ’gray ’)
plt.title(’Edged  Image ’), plt.xticks ([]), plt.yticks ([])
```

![Example Image](https://github.com/alex-ta/LaneDetection/new/master/workingsample.png)
