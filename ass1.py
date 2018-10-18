import cv2
import numpy as np


def gamma_correction(image, gamma=2):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])

    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def addWeights(image, brightness, contrast):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    return cv2.addWeighted(image, 1. + contrast/127., image, 0, brightness-contrast)


def color_correction():
    # reading images
    woman_image = cv2.imread('./A1/Q1I1.png')
    batman_image = cv2.imread('./A1/Q1I2.jpg')
    batman_image_inverted = cv2.imread('./A1/Q1I2.jpg')

    brighter_woman_image = gamma_correction(woman_image)
    contrast_woman_image = addWeights(brighter_woman_image, 40, 120)

    cv2.imwrite('./output/woman_contrast.png', contrast_woman_image)

    # batman_image = cv2.resize(batman_image,None,fx=0.7, fy=1, interpolation = cv2.INTER_CUBIC)
    batman_image = cv2.resize(
        batman_image, (contrast_woman_image.shape[1], contrast_woman_image.shape[0]))

    batman_image_inverted = cv2.resize(
        batman_image_inverted, (contrast_woman_image.shape[1], contrast_woman_image.shape[0]))


    image_2_row, image_2_col, channels_2 = batman_image.shape

    # flip image
    for i in range(image_2_row):
        for j in range(image_2_col):
            batman_image_inverted[i, image_2_col - j - 1] = batman_image[i, j]

    # translate batman a bit to the right
    translationMatrix = np.float32([[1, 0, 250], [0, 1, 0]])
    batman_image_inverted = cv2.warpAffine(batman_image_inverted, translationMatrix, (
        batman_image_inverted.shape[1], batman_image_inverted.shape[0]))

    # merge batman and woman
    image_4 = cv2.addWeighted(contrast_woman_image,
                              1.0, batman_image_inverted, 0.5, 0)

    cv2.imwrite('./output/image_4.png', image_4)

    # return merged


def fitting_frames():
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    # read images
    sherlock = cv2.imread('./A1/Q2I1.jpg')
    frame_fig_6 = cv2.imread('./A1/Q2I2.jpg')
    frame_fig_7 = cv2.imread('./A1/Q2I3.jpg')
    frame_fig_10 = cv2.imread('./A1/Q3I1.jpg')

    # get corners
    # hardcoded from image editor
    sherlock_corners = np.float32([[0, 0], [sherlock.shape[1], 0], [
                                  sherlock.shape[1], sherlock.shape[0]]])
    sherlock_corners_four = np.float32([[0, 0], [sherlock.shape[1], 0], [
                                       sherlock.shape[1], sherlock.shape[0]], [0, sherlock.shape[0]]])
    frame_three_corners = np.float32([[1219, 378], [1310, 380], [1310, 517]])
    frame_four_corners = np.float32(
        [[1219, 378], [1310, 378], [1310, 515], [1219, 515]])

    # transformation matrix
    affine = cv2.getAffineTransform(sherlock_corners, frame_three_corners)
    sherlock_new_1 = cv2.warpAffine(
        sherlock, affine, (frame_fig_6.shape[1], frame_fig_6.shape[0]))
    cv2.fillConvexPoly(frame_fig_6, frame_four_corners.astype(int), 0, 16)

    image_7 = cv2.addWeighted(frame_fig_6, 1.0, sherlock_new_1, 1.0, 0)

    cv2.imwrite('./output/image_7.png', image_7)

    # Q2
    frame_three_corners_2 = np.float32([[372, 96], [707, 131], [664, 559]])
    frame_four_corners_2 = np.float32(
        [[372, 96], [707, 131], [664, 559], [329, 526]])

    affine = cv2.getAffineTransform(sherlock_corners, frame_three_corners_2)
    sherlock_new_2 = cv2.warpAffine(
        sherlock, affine, (frame_fig_7.shape[1], frame_fig_7.shape[0]))
    cv2.fillConvexPoly(frame_fig_7, frame_four_corners_2.astype(int), 0, 16)

    image_9 = cv2.addWeighted(frame_fig_7, 1.0, sherlock_new_2, 1.0, 0)

    cv2.imwrite('./output/image_9.png', image_9)

    # Q3

    frame_four_corners_3 = np.float32(
        [[164, 36], [469, 70], [464, 353], [158, 389]])

    perspective = cv2.getPerspectiveTransform(
        sherlock_corners_four, frame_four_corners_3)
    sherlock_new_3 = cv2.warpPerspective(
        sherlock, perspective, (frame_fig_10.shape[1], frame_fig_10.shape[0]))

    cv2.fillConvexPoly(frame_fig_10, frame_four_corners_3.astype(int), 0, 16)
    image_10 = cv2.addWeighted(frame_fig_10, 1.0, sherlock_new_3, 1.0, 0)
    cv2.imwrite('./output/image_10.png', image_10)


def main():
    color_correction()
    fitting_frames()


if __name__ == '__main__':
    main()
