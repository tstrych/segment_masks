import click
import cv2 as cv
import glob
import math
import numpy as np
import os


def extract_n_bounding_boxes_from_image(image, image_path, output_dir):
    # convert image from 3 channels to 1 channel
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # different data may require different thresholding
    # threshold block size was tested iteratively on 'RS_HOMEWORK_BB.png'
    # with aim to maximize the average of all contours content and keep the number of counter to 3
    # such approach is timely ineffective
    # other possibility is to skip counters with content lower than e.g. 200
    # I decided not to use cv.Canny as I didn't find it beneficial
    threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 223, 0)

    # Only looking for the external bounds,
    # CHAIN_APPROX_SIMPLE is more time effective contour approximation method
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # find rotated rectangle
        rect = cv.minAreaRect(contour)
        # get box points in order to
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # draws rectangle around contours, used for debugging purposes
        # cv.drawContours(image, [box], 0, (0, 0, 255), 2)

        width = math.ceil(rect[1][0])
        height = math.ceil(rect[1][1])
        angle = rect[2]

        xs = [i[0] for i in box]
        ys = [i[1] for i in box]
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # rotate the whole image based on the contour center
        matrix = cv.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        rotated = cv.warpAffine(image, matrix, gray.shape)
        # crop the image
        cropped = cv.getRectSubPix(rotated, (width, height), center)

        # rotate based on assignment
        if width > height:
            cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)

        # save
        output_fn = f'{(".".join(image_path.split(".")[:-1]))}_{i}.{image_path.split(".")[-1]}'
        output_path = os.path.join(output_dir, output_fn)
        if not os.path.exists(output_path):
            cv.imwrite(output_path, cropped)
            print(f'output {output_path} was created')
        else:
            print(f'output {output_path} already exists')

        # for debugging purposes
        # from matplotlib import pyplot as plt
        # plt.imshow(cropped)
        # plt.show()


@click.command()
@click.option(
    '--input-dir',
    default='testing_images',
    type=click.Path(exists=True),
    help='specify path to input directory that contains images',
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='output_images',
    help='specify path to output directory',
)
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # create output folder
    for file in glob.glob(os.path.join(input_dir, "*.png")):
        image = cv.imread(file)
        fn = os.path.split(file)[-1]
        extract_n_bounding_boxes_from_image(image, fn, output_dir)  # extract and save images


if __name__ == "__main__":
    main()
