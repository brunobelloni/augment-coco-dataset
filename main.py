import glob
import json
from itertools import chain
from tqdm import tqdm
import asyncio
import albumentations as A
import cv2
import numpy as np

DATASET = 'dataset'
DATASET_OUT = 'output'
FOLDER_PATH = '{DATASET}/{DATASET_TYPE}/*.json'


def coco2albumentations(segmentation):
    return [(segmentation[i], segmentation[i + 1], 0, 0) for i in range(0, len(segmentation), 2)]


def albumentations2coco(keypoints):
    return list(chain.from_iterable([keypoint[:2] for keypoint in keypoints]))


def apply_albumentations(image, annotations, image_id, annotation_id):
    height, width, _ = image.shape

    transform = A.Compose(
        transforms=[
            # change to your own transforms
            A.RandomResizedCrop(height=height, width=width, always_apply=True),
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
        annotation_id += 1
        keypoints = albumentations2coco(transformed[keypoint_name])
        cv2_keypoints = np.array([(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)], dtype='int')
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

    # start preprocessing
    transformed_image, transformed_keypoints = apply_albumentations(
        image=cv2_image,
        image_id=image['id'],
        annotation_id=anno_id,
        annotations=img_annotations,
    )

    cv2.imwrite(filename=output_image_path, img=transformed_image)

    return transformed_keypoints


async def main():
    anno_id = 0

    for dataset_type in ['train', 'valid', 'test']:
        for anno_path in glob.iglob(FOLDER_PATH.format(DATASET=DATASET, DATASET_TYPE=dataset_type)):
            with open(anno_path, 'r') as f_in:
                anno = json.load(f_in)

                tasks = [
                    asyncio.to_thread(process, image=image, dataset_type=dataset_type, anno=anno, anno_id=anno_id)
                    for image in tqdm(anno['images'], desc=dataset_type)
                ]
                results = await asyncio.gather(*tasks)
                new_annos = list(chain.from_iterable(results))

                with open(anno_path.replace(DATASET, DATASET_OUT), 'w') as f_out:
                    anno['annotations'] = new_annos
                    json.dump(anno, f_out)


if __name__ == '__main__':
    asyncio.run(main())
