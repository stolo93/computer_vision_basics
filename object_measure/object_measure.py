import argparse
import cv2
import imutils
import numpy as np
from imutils import contours
from imutils import perspective
from math import dist


def midpoint(pt_a: tuple, pt_b: tuple):
    return int((pt_a[0] + pt_b[0]) * 0.5), int((pt_a[1] + pt_b[1]) * 0.5)


def main():
    parser = argparse.ArgumentParser(
        prog='Object measure',
        description='Measure objects in the image provided\nmeasure is calibrated by the leftmost object in the image'
    )
    parser.add_argument('-i', '--image', required=True)  # Image path
    parser.add_argument('-w', '--width', required=True)  # Width of the reference 'leftmost' object

    args = parser.parse_args()
    try:
        referral_width = float(args.width)
    except ValueError as e:
        print(e)
        exit(1)

    image = cv2.imread(args.image)
    assert image is not None, f'Invalid image path {args.image}'

    desired_max_width = 640

    # Resize the image if needed
    if image.shape[1] > desired_max_width:
        scale_factor = float(desired_max_width) / float(image.shape[1])
        dim = (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor))
        image = cv2.resize(image, dim)

    # Convert to grayscale image before edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 100, 200)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts, _ = contours.sort_contours(cnts)  # Sort contours from left to right

    pixels_per_metric = None
    for c in cnts:
        # If contour area is not big enough ignore it
        if cv2.contourArea(c) < 100:
            continue

        # Rotated bounding rect
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype='int')

        # Order the points such that they appear to be:
        # top-left, top-right, bottom-left, bottom-right
        # then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)

        for x, y in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), cv2.FILLED)

        # Unpack box and compute the midpoint between
        # top left <-> top right
        # bottom left <-> bottom right
        # and so on...
        tl, tr, br, bl = box
        tltr_x, tltr_y = midpoint(tl, tr)
        blbr_x, blbr_y = midpoint(bl, br)

        tlbl_x, tlbl_y = midpoint(tl, bl)
        trbr_x, trbr_y = midpoint(tr, br)

        cv2.circle(image, (tltr_x, tltr_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (blbr_x, blbr_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (tlbl_x, tlbl_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (trbr_x, trbr_y), 5, (255, 0, 0), cv2.FILLED)

        cv2.line(image, (tltr_x, tltr_y), (blbr_x, blbr_y), (255, 0, 255), 1)
        cv2.line(image, (tlbl_x, tlbl_y), (trbr_x, trbr_y), (255, 0, 255), 1)

        # Compute size in pixels
        distance_a = dist((tlbl_x, tlbl_y), (trbr_x, trbr_y))
        distance_b = dist((tltr_x, tltr_y), (blbr_x, blbr_y))

        # Compute pixels per metric ration based on the dimension provided
        if pixels_per_metric is None:
            pixels_per_metric = referral_width / distance_a

        width = distance_a * pixels_per_metric
        height = distance_b * pixels_per_metric

        cv2.putText(image, f'{height:.2f}', (tltr_x - 10, tltr_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(image, f'{width:.2f}', (tlbl_x + 10, tlbl_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()