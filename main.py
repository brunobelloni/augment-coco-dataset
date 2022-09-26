import glob
import json
from itertools import chain
import random

import albumentations as A
import cv2
import numpy as np

# random.seed(1232132121)  # necessary

DATASET = 'ddr-seg-bbox'
DATASET_OUT = 'ddr-seg-bbox-crop-4tilesn'

annotation_id = 0

example = {
    'info': {},
    'licenses': [],
    'images': [],  # insert here
    'annotations': [],  # insert here
    'categories': [
        {'supercategory': 'EX', 'id': 1, 'name': 'EX'},
        {'supercategory': 'HE', 'id': 2, 'name': 'HE'},
        {'supercategory': 'SE', 'id': 3, 'name': 'SE'},
        {'supercategory': 'MA', 'id': 4, 'name': 'MA'},
    ],
}


def coco2albumentations(segmentation):
    return [(segmentation[i], segmentation[i + 1], 0, 0) for i in range(0, len(segmentation), 2)]


def albumentations2coco(keypoints):
    return list(chain.from_iterable([keypoint[:2] for keypoint in keypoints]))


def apply_albumentations(image, annotations, image_id):
    global annotation_id

    transform = A.Compose(
        transforms=[
            # A.RandomResizedCrop(height=height, width=width, always_apply=True),
            A.HorizontalFlip(always_apply=True),
        ],
        additional_targets={f"keypoints_{index}": 'keypoints' for index, anno in enumerate(annotations)},
    )

    keypoint_kwargs = {
        f"keypoints_{index}": coco2albumentations(anno['segmentation'][0]) for index, anno in enumerate(annotations)
    }

    transformed = transform(image=image, **keypoint_kwargs)

    new_annos = []
    for anno, keypoint_name in zip(annotations, keypoint_kwargs):
        # print(transformed[keypoint_name])
        keypoints = albumentations2coco(transformed[keypoint_name])
        parsed_keypoints = np.array(
            [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)],
            dtype='int',
        )
        annotation_id += 1
        new_annos.append({
            'id': annotation_id,
            'image_id': image_id,
            'iscrowd': anno['iscrowd'],
            'segmentation': [keypoints],
            'category_id': anno['category_id'],
            'area': cv2.contourArea(parsed_keypoints),
            'bbox': list(cv2.boundingRect(parsed_keypoints)),
        })

    return transformed['image'], new_annos


if __name__ == '__main__':

    folder_path = 'ddr-seg-bbox/**/*.coco.json'

    for annotation_path in glob.iglob(folder_path):
        annotation_type = annotation_path.split('/')[1]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

            for image in annotation['images'][5:]:
                image_path = f'{DATASET}/{annotation_type}/{image["file_name"]}'

                cv2_image = cv2.imread(filename=image_path)
                img_annotations = list(filter(lambda x: x['image_id'] == image['id'], annotation['annotations']))

                # start preprocessing
                new_image, new_annotations = apply_albumentations(
                    image=cv2_image,
                    image_id=image['id'],
                    annotations=img_annotations,
                )

                cv2.imwrite(filename=image["file_name"], img=new_image)

                example['images'].append(image)
                example['annotations'].extend(new_annotations)

                with open('data.json', 'w') as f:
                    json.dump(example, f)

                exit(0)

