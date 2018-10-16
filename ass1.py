import cv2
import numpy as np 

def color_correction():
	# woman
	woman_image = cv2.imread('./A1/Q1I1.png')

	b = 70 # brightness
	c = 80  # contrast


	image_hsv = cv2.cvtColor(woman_image, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(image_hsv)
	v += 25
	final_hsv = cv2.merge((h, s, v))

	brighter_woman_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

	# brighter_woman_image = cv2.addWeighted(woman_image, 1.0, woman_image, 0.8, 0)
	cv2.imwrite('./output/woman_brighter.png',brighter_woman_image)


	# batman
	image_2 = cv2.imread('./A1/Q1I2.jpg')

	image_2_inverted = cv2.imread('./A1/Q1I2.jpg')
	image_2 = cv2.resize(image_2, (brighter_woman_image.shape[1], brighter_woman_image.shape[0]))
	image_2_inverted = cv2.resize(image_2_inverted, (brighter_woman_image.shape[1], brighter_woman_image.shape[0]))



	image_2_row, image_2_col, channels_2 = image_2.shape

	# flip image
	for i in range(image_2_row):
		for j in range(image_2_col):
			image_2_inverted[i, image_2_col - j - 1] = image_2[i, j]

	# translate batman a bit to the right
	translationMatrix = np.float32([[1,0,250],[0,1,0]])
	image_2_inverted = cv2.warpAffine(image_2_inverted, translationMatrix , (image_2_inverted.shape[1], image_2_inverted.shape[0]))


	# merge batman and woman
	image_4 = cv2.addWeighted(brighter_woman_image,1.0,image_2_inverted,0.5,0)

	cv2.imwrite('./output/image_4.png',image_4)

	# return merged


def fitting_frames():
	# read images
	sherlock = cv2.imread('./A1/Q2I1.jpg')
	frame_fig_6 = cv2.imread('./A1/Q2I2.jpg')
	frame_fig_7 = cv2.imread('./A1/Q2I3.jpg')
	frame_fig_10 = cv2.imread('./A1/Q3I1.jpg')
	# print(sherlock.shape)

	# get corners
	# hardcoded from image editor
	sherlock_corners = np.float32([[0, 0], [sherlock.shape[1], 0], [sherlock.shape[1], sherlock.shape[0]]])
	sherlock_corners_four = np.float32([[0,0], [sherlock.shape[1],0], [sherlock.shape[1], sherlock.shape[0]], [0, sherlock.shape[0]]])
	frame_three_corners = np.float32([[1219, 378], [1310, 380], [1310, 517]])
	frame_four_corners = np.float32([[1219, 378], [1310, 378], [1310, 515], [1219, 515]])

	affine = cv2.getAffineTransform(sherlock_corners, frame_three_corners)
	sherlock_new_1 = cv2.warpAffine(sherlock, affine, (frame_fig_6.shape[1], frame_fig_6.shape[0]))
	cv2.fillConvexPoly(frame_fig_6, frame_four_corners.astype(int), 0, 16)

	image_7 = cv2.addWeighted(frame_fig_6, 1.0, sherlock_new_1, 1.0, 0)

	# cv2.imshow('image_2_inverted', image_8)
	cv2.imwrite('./output/image_7.png',image_7)


	# Q2
	frame_three_corners_2 = np.float32([[372, 96],[707, 131],[664, 559]])
	frame_four_corners_2 = np.float32([[372, 96],[707, 131],[664, 559], [329, 526]])

	affine = cv2.getAffineTransform(sherlock_corners, frame_three_corners_2)
	sherlock_new_2 = cv2.warpAffine(sherlock, affine, (frame_fig_7.shape[1], frame_fig_7.shape[0]))
	cv2.fillConvexPoly(frame_fig_7, frame_four_corners_2.astype(int), 0, 16)

	image_9 = cv2.addWeighted(frame_fig_7, 1.0, sherlock_new_2, 1.0, 0)

	# cv2.imshow('image_2_inverted', image_10)
	cv2.imwrite('./output/image_9.png',image_9)

	# Q3

	frame_four_corners_3 = np.float32([[165, 38],[469, 71],[463, 353],[161, 388]])

	perspective = cv2.getPerspectiveTransform(sherlock_corners_four, frame_four_corners_3)
	sherlock_new_3 = cv2.warpPerspective(sherlock, perspective, (frame_fig_10.shape[1], frame_fig_10.shape[0]))

	cv2.fillConvexPoly(frame_fig_10, frame_four_corners_3.astype(int), 0, 16)
	image_10 = cv2.addWeighted(frame_fig_10, 1.0, sherlock_new_3, 1.0, 0)
	cv2.imwrite('./output/image_10.png',image_10)








def main():
	color_correction()
	fitting_frames()
	# cv2.imshow('image_2_inverted', output)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


if __name__ == '__main__':
	main()