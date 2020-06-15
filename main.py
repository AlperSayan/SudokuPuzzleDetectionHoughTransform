import cv2 as cv
import glob
import numpy as np
import operator
import os
import matplotlib.pyplot as plt

# Mustafa Alper Sayan S015674 Department of Computer Science

def main(images):
    i = 1

    cwd = os.getcwd()
    if not os.path.exists(cwd + "\output"):
        os.mkdir(cwd + "\output")

    for image in images:

        image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        preprocess = preprocessor(image_to_gray)
        corners_of_sudoku = get_largest_corners(preprocess)
        cropped_image = crop_sudoku_puzzle(image, corners_of_sudoku)

        cropped_image_to_gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
        preprocess2 = preprocessor2(cropped_image_to_gray)

        canny_image = cv.Canny(preprocess2, 30, 60, apertureSize=3)
        hough_image = hough_transform(cropped_image, canny_image)

        titles = ["gray image", "preprocess", " crop to gray", "canny image", "houghImage"]
        images = [image_to_gray, preprocess, cropped_image_to_gray, canny_image, hough_image]
        mat_plot_lib_show_images(images, titles)

        output_path = "output/image" + str(i) + ".jpg"
        cv.imwrite(output_path, hough_image)
        i += 1


def mat_plot_lib_show_images(images, titles, cols = 2):

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title, fontsize=40)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def hough_transform(image, hough_image):
    global line_flags
    lines = cv.HoughLines(hough_image, 1, np.pi / 180, 150)
    if lines is None:
        return image

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            # if we already disregarded the ith element in the ordered list then we don't care
            if not line_flags[indices[i]]:
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return image


def preprocessor(image):

    preprocess = cv.GaussianBlur(image, (9, 9), 0)
    preprocess = cv.adaptiveThreshold(preprocess, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    preprocess = cv.bitwise_not(preprocess, preprocess)
    kernel = np.ones((5, 5), np.uint8)
    preprocess = cv.dilate(preprocess, kernel)
    return preprocess


def preprocessor2(image):
    image = cv.GaussianBlur(image, (9, 9), 0)
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    image = cv.bitwise_not(image, image)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((6, 6), np.uint8)
    edges = cv.erode(edges, kernel, iterations=1)
    return edges


def get_largest_corners(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def crop_sudoku_puzzle(image, crop_rectangle):
    img = image
    crop_rect = crop_rectangle

    def distance_between(a, b):

        return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

    def crop_img():
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

        source_rect = np.array(np.array([top_left, bottom_left, bottom_right, top_right], dtype='float32'))

        sides = max([
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right)
        ])

        dest_square = np.array([[0, 0], [sides - 1, 0], [sides - 1, sides - 1], [0, sides - 1]], dtype='float32')

        modified = cv.getPerspectiveTransform(source_rect, dest_square)

        return cv.warpPerspective(img, modified, (int(sides), int(sides)))

    return crop_img()


if __name__ == '__main__':
    path = glob.glob("images/*.jpg")
    images = [cv.imread(file, -1) for file in path]
    main(images)
