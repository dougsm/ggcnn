import datetime
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp


DATASET_NAME = 'dataset'
OUTPUT_DIR = 'data/datasets'
RAW_DATA_DIR = 'data/cornell'
OUTPUT_IMG_SIZE = (300, 300)
RANDOM_ROTATIONS = 10
RANDOM_ZOOM = True

TRAIN_SPLIT = 0.8
# OR specify which images are in the test set.
TEST_IMAGES = None
VISUALISE_ONLY = False

# File name patterns for the different file types.  _ % '<image_id>'
_rgb_pattern = os.path.join(RAW_DATA_DIR, 'pcd%sr.png')
_pcd_pattern = os.path.join(RAW_DATA_DIR, 'pcd%s.txt')
_pos_grasp_pattern = os.path.join(RAW_DATA_DIR, 'pcd%scpos.txt')
_neg_grasp_pattern = os.path.join(RAW_DATA_DIR, 'pcd%scneg.txt')


def get_image_ids():
    # Get all the input files, extract the numbers.
    rgb_images = glob.glob(_rgb_pattern % '*')
    rgb_images.sort()
    return [r[-9:-5] for r in rgb_images]


if __name__ == '__main__':
    # Create the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Label the output file with the date/time it was created
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    outfile_name = os.path.join(OUTPUT_DIR, '%s_%s.hdf5' % (DATASET_NAME, dt))

    fields = [
        'img_id',
        'rgb',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width'
    ]

    # Empty datatset.
    dataset = {
        'test':  dict([(f, []) for f in fields]),
        'train': dict([(f, []) for f in fields])
    }

    for img_id in get_image_ids():
        print('Processing: %s' % img_id)

        # Decide whether this is train or test.
        ds_output = 'train'
        if TEST_IMAGES:
            if int(img_id) in TEST_IMAGES:
                print("This image is in TEST_IMAGES")
                ds_output = 'test'
        elif np.random.rand() > TRAIN_SPLIT:
            ds_output = 'test'
        ds = dataset[ds_output]

        # Load the image
        rgb_img_base = Image(io.imread(_rgb_pattern % img_id))
        depth_img_base = DepthImage.from_pcd(_pcd_pattern % img_id, (480, 640))
        depth_img_base.inpaint()

        # Load Grasps.
        bounding_boxes_base = grasp.BoundingBoxes.load_from_file(_pos_grasp_pattern % img_id)
        center = bounding_boxes_base.center

        for i in range(RANDOM_ROTATIONS):
            angle = np.random.random() * 2 * np.pi - np.pi
            rgb = rgb_img_base.rotated(angle, center)
            depth = depth_img_base.rotated(angle, center)

            bbs = bounding_boxes_base.copy()
            bbs.rotate(angle, center)

            left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
            right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

            top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
            bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

            rgb.crop((top, left), (bottom, right))
            depth.crop((top, left), (bottom, right))
            bbs.offset((-top, -left))

            if RANDOM_ZOOM:
                zoom_factor = np.random.uniform(0.4, 1.0)
                rgb.zoom(zoom_factor)
                depth.zoom(zoom_factor)
                bbs.zoom(zoom_factor, (OUTPUT_IMG_SIZE[0]//2, OUTPUT_IMG_SIZE[1]//2))

            depth.normalise()

            pos_img, ang_img, width_img = bbs.draw(depth.shape)

            if VISUALISE_ONLY:
                f = plt.figure()
                ax = f.add_subplot(1, 5, 1)
                rgb.show(ax)
                bbs.show(ax)
                ax = f.add_subplot(1, 5, 2)
                depth.show(ax)
                bbs.show(ax)

                ax = f.add_subplot(1, 5, 3)
                ax.imshow(pos_img)

                ax = f.add_subplot(1, 5, 4)
                ax.imshow(ang_img)

                ax = f.add_subplot(1, 5, 5)
                ax.imshow(width_img)

                plt.show()
                continue

            ds['img_id'].append(int(img_id))
            ds['rgb'].append(rgb.img)
            ds['depth_inpainted'].append(depth.img)
            ds['bounding_boxes'].append(bbs.to_array(pad_to=25))
            ds['grasp_points_img'].append(pos_img)
            ds['angle_img'].append(ang_img)
            ds['grasp_width'].append(width_img)

    # Save the output.
    if not VISUALISE_ONLY:
        with h5py.File(outfile_name, 'w') as f:
            for tt_name in dataset:
                for ds_name in dataset[tt_name]:
                    f.create_dataset('%s/%s' % (tt_name, ds_name), data=np.array(dataset[tt_name][ds_name]))