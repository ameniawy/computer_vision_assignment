import cv2
import numpy as np 

# woman
image_1 = cv2.imread('./A1/Q1I1.png')
image_1_row, image_1_col, channels_1 = image_1.shape

alpha = 0
beta = 0

added_value = np.full((image_1.shape[0], image_1.shape[1]), 25)

brighter_image_1 = cv2.addWeighted(image_1, 1.0, image_1, 0.8, 0)

# output first image
cv2.imwrite('brighter_image_1.png', brighter_image_1)


# batman
image_2 = cv2.imread('./A1/Q1I2.jpg')

image_2_inverted = cv2.imread('./A1/Q1I2.jpg')

image_2_row, image_2_col, channels_2 = image_2.shape

# print(image_2_row, image_2_col)

# invert image
for i in range(image_2_row):
    for j in range(image_2_col):
        image_2_inverted[i, image_2_col - j - 1] = image_2[i, j]



# resized_image = cv2.resize(image_2_inverted, (image_1_row, image_2_col)) 
cv2.imshow('image_2_inverted', image_2_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()