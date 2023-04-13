# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import the necessary libraries and read the original image and save it as a image variable.

### Step 2:
Translate the image.

### Step 3:
Scale the image.

### Step 4:
Shear the image.

### Step 5:
Reflect of image.

### Step 6:
Rotate the image.

## Program:
```
Developed By: M VIDYA NEELA
Register Number: 212221230120
```

i)Image Translation:

```
#Image Translation

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread('puppy.jpeg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])
translated_image = cv2.warpPerspective (input_image, M, (cols, rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show(
)
```

ii) Image Scaling:

  ```
  #Image Scaling

S=np.float32([[4,0,0],[0,4,0],[0,0,4]])
scaled_image=cv2.warpPerspective(input_image,S,(cols*4,rows*4))
plt.axis('off')
plt.imshow(scaled_image)
plt.show()
```


iii)Image shearing:

```
#Image shearing

M_x = np.float32([[1, 0.5, 0],
                  [0, 1 ,0],
                  [0, 0 , 1]])
M_y= np.float32([[1, 0, 0],
                 [0.5, 1, 0],
                 [0, 0, 1]])
sheared_img_xaxis = cv2.warpPerspective (input_image,M_x,(int(cols*1.5), int(rows *1.5))) 
sheared_img_yaxis = cv2.warpPerspective (input_image,M_y, (int(cols*1.5), int(rows *1.5)))
plt.axis('off')
plt.imshow(sheared_img_xaxis)
plt.show()

plt.axis('off')
plt.imshow(sheared_img_yaxis)
plt.show()
```


iv)Image Reflection:

```
#Image Reflection

M_x=np.float32([[1,0,0],
               [0,-1,rows],
               [0,0,1]])
M_y=np.float32([[-1,0,cols],
               [0,1,0],
               [0,0,1]])
reflected_img_xaxis=cv2.warpPerspective(input_image,M_x,(cols,rows))
reflected_img_yaxis=cv2.warpPerspective(input_image,M_y,(cols,rows))
plt.axis('off')
plt.imshow(reflected_img_xaxis)
plt.show()
plt.axis('off')
plt.imshow(reflected_img_yaxis)
plt.show()
```


v)Image Rotation:

```
#Image Rotation

angle=np.radians(80)
M=np.float32([[np.sin(angle),-(np.cos(angle)),0],
               [np.cos(angle),np.sin(angle),0],
               [0,0,1]])
rotated_img=cv2.warpPerspective(input_image,M,(cols,rows))
plt.axis('off')
plt.imshow(rotated_img)
plt.show()
```


vi)Image Cropping:

```
#Image Cropping
cropped_img=input_image[15:150,14:150]
plt.axis('off')
plt.imshow(cropped_img)
plt.show()
```


## Output:
### i)Image Translation

![image](https://user-images.githubusercontent.com/94169318/231787694-d257e602-b927-4627-aea9-a503c9073735.png)


### ii) Image Scaling

![image](https://user-images.githubusercontent.com/94169318/231787797-aa4f2ea1-b589-4c77-a40b-8dff6ca960ba.png)


### iii)Image shearing

![image](https://user-images.githubusercontent.com/94169318/231787907-f41e66b0-4b05-4538-b104-8b3678c6a527.png)


### iv)Image Reflection

![image](https://user-images.githubusercontent.com/94169318/231788033-3bfb740a-efdf-46d1-85c8-425094aa7c14.png)



### v)Image Rotation

![image](https://user-images.githubusercontent.com/94169318/231788450-5b321442-c794-4fe4-aeaf-72556a294318.png)


### vi)Image Cropping

![image](https://user-images.githubusercontent.com/94169318/231788561-faf8e39e-bebc-46fc-aa38-64e89522d28e.png)





## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
