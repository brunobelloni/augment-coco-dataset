import asyncio
import glob
import json
import random
from itertools import chain

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

random.seed(42)

DATASET = 'dataset'
DATASET_OUT = 'output'
FOLDER_PATH = '{DATASET}/{DATASET_TYPE}/*.json'

pbar = tqdm()


def coco2albumentations(segmentation):
    return [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]


def albumentations2coco(keypoints):
    return list(chain.from_iterable(keypoints))


def coco2cv2(keypoints):
    return np.array([(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)], dtype='int')


def apply_albumentations(image, annotations, image_id, annotation_id, transforms):
    keypoint_kwargs = {
        f"keypoints_{index}": coco2albumentations(anno['segmentation'][0]) for index, anno in enumerate(annotations)
    }

    transform = A.Compose(
        transforms=transforms,
        keypoint_params=A.KeypointParams(format='xy'),
        additional_targets={f"keypoints_{index}": 'keypoints' for index, anno in enumerate(annotations)},
    )
    transformed = transform(image=image, keypoints=[], **keypoint_kwargs)

    new_annos = []
    for anno, keypoint_name in zip(annotations, keypoint_kwargs):
        annotation_id += 1
        keypoints = albumentations2coco(transformed[keypoint_name])
        cv2_keypoints = coco2cv2(keypoints)
        if not keypoints or len(keypoints) <= 4:
            continue
        new_annos.append({
            'id': annotation_id,
            'image_id': image_id,
            'iscrowd': anno['iscrowd'],
            'segmentation': [keypoints],
            'category_id': anno['category_id'],
            'area': cv2.contourArea(cv2_keypoints),
            'bbox': list(cv2.boundingRect(cv2_keypoints)),
        })

    return transformed['image'], new_annos


def process(image, dataset_type, anno, anno_id):
    input_image_path = f'{DATASET}/{dataset_type}/{image["file_name"]}'
    output_image_path = f'{DATASET_OUT}/{dataset_type}/{image["file_name"]}'

    cv2_image = cv2.imread(filename=input_image_path)
    img_annotations = list(filter(lambda x: x['image_id'] == image['id'], anno['annotations']))

    height, width, _ = cv2_image.shape
    transforms = [
        # change to your own transforms
        A.RandomResizedCrop(height=height, width=width, always_apply=True),
        A.HorizontalFlip(always_apply=True),
    ]

    # start preprocessing
    transformed_image, transformed_keypoints = apply_albumentations(
        image=cv2_image,
        image_id=image['id'],
        transforms=transforms,
        annotation_id=anno_id,
        annotations=img_annotations,
    )

    cv2.imwrite(filename=output_image_path, img=transformed_image)

    if pbar.desc != dataset_type:
        pbar.reset()
        pbar.total = len(anno['images'])
        pbar.desc = dataset_type

    pbar.update(1)

    return transformed_keypoints


async def main():
    anno_id = 0

    for dataset_type in ['train', 'valid', 'test']:
        for anno_path in glob.iglob(FOLDER_PATH.format(DATASET=DATASET, DATASET_TYPE=dataset_type)):
            with open(anno_path, 'r') as f_in:
                anno = json.load(f_in)

                tasks = [
                    asyncio.to_thread(process, image=image, dataset_type=dataset_type, anno=anno, anno_id=anno_id)
                    for image in anno['images']
                ]
                results = await asyncio.gather(*tasks)
                new_annos = list(chain.from_iterable(results))

                with open(anno_path.replace(DATASET, DATASET_OUT), 'w') as f_out:
                    anno['annotations'] = new_annos
                    json.dump(anno, f_out)


if __name__ == '__main__':
    asyncio.run(main())
