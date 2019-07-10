import os
import enum
import argparse
import cv2
import numpy as np

from config import dataset_config

GREEN_COLOR = (0, 255, 0)

INPUT_FOLDER = 'input'
IMAGES_FOLDER = 'img'
OUTPUT_FOLDER = 'output'

# Setup the termination criteria, either 1000 iterations or move by at least 1 pt
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)
PYRAMIDAL_MAX_LEVEL = 8

HISTOGRAM_SIZE = [180]
# Disable scaling
SCALE = 1
# Histogram bin boundaries
VIDEO_FPS = 30

# Params for Shi-Tomasi corner detection
FEATURE_PARAMS = dict(
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=5
)

# Lucas Kanade params
LK_PARAMS = dict(
    winSize=(20, 20),
    maxLevel=0,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


class METHODS(enum.Enum):
    LK = 'lucaskanade'
    LKPYRAMIDAL = 'lucaskanadepyramid'


def get_images(dataset):
    # Sort images in ascending order
    dataset_path = os.path.join(INPUT_FOLDER, dataset, IMAGES_FOLDER)
    names = sorted(os.listdir(dataset_path), key=lambda name: int(name.split('.')[0]))
    paths = (os.path.join(dataset_path, name) for name in names)
    return [cv2.imread(path) for path in paths]


def write_video(images, dataset, method):
    print('Generating video')
    # Save video as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, layers = images[0].shape
    video_path = os.path.join(OUTPUT_FOLDER, method, dataset + '.mp4')
    video = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    print('Done')


def lucas_kanade(images, args):
    x, y, width, height = args.roi

    # Convert all images to gray
    images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    roi_image = images_gray[0][y:y + height, x:x + width]

    # Get points for tracking optical flow only from roi
    roi_points = cv2.goodFeaturesToTrack(roi_image, mask=None, **FEATURE_PARAMS)
    points = np.array(
        [[(point_x + x, point_y + y,)] for (point_y, point_x) in roi_points[:, 0]],
        dtype=np.float32)

    if args.method == METHODS.LKPYRAMIDAL.value:
        LK_PARAMS.update({'maxLevel': PYRAMIDAL_MAX_LEVEL})

    print('Start tracking')
    for idx, frame in enumerate(images[:-1]):
        gray_frame = images_gray[idx].copy()
        next_gray_frame = images_gray[idx + 1].copy()

        points, status, error = cv2.calcOpticalFlowPyrLK(gray_frame, next_gray_frame, points, None, **LK_PARAMS)

        for point in points:
            cv2.circle(frame, (point[0][0], point[0][1]), 2, GREEN_COLOR, -1)
        cv2.imshow("Frame", frame)

        # Show each frame for 40 ms
        key = cv2.waitKey(40)
        # Break on esc key
        if key == 27:
            break

    cv2.destroyAllWindows()
    write_video(images, args.dataset, args.method)


def main(args):
    if args.dataset not in dataset_config:
        print('Dataset not found')
        return

    if not args.roi:
        # Use default roi from config
        args.roi = dataset_config[args.dataset]['roi']

    images = get_images(args.dataset)
    lucas_kanade(images, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lucas Kanade detector')
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--roi', type=int, nargs=4, help='Region of interest (x, y, width, height)')
    parser.add_argument('--method', type=str, help='lucaskanade or lucaskanadepyramid method', default=METHODS.LK.value)

    args = parser.parse_args()
    main(args)
