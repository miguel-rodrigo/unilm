import numpy as np
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

with open("output/inference/coco_instances_results.json", "r") as file:
    data = json.load(file)

print(data[0]["bbox"])

img = mpimg.imread("data/publaynet-mini-sample/val/PMC5447509_00002.jpg")
# plt.imshow(img)
# plt.show()

def blend(
    top_color: np.array, bottom_color: np.array, opacity: float = 0.5
) -> np.array:
    return np.round(top_color * (1-opacity) + bottom_color * opacity).astype("uint8")


annotation_color = np.array([255, 0, 255])

# TODO: color based on category_id
# TODO: print the order so that I can tell which are the bad ones (and maybe plot them in batches of 3-4 only)

def annotate_region(img, bbox, color):
    left, top, w, h = np.round(bbox).astype(int)
    img = img.copy()
    img[top : (top + h), left : (left + w), :] = blend(
        img[top : (top + h), left : (left + w), :], color, 0.3
    )

    return img

plt.imshow(annotate_region(img, data[0]["bbox"], annotation_color))
plt.show()

annotated_img = img.copy()
for region in data:
    if region['score'] < 0.98:
        continue

    annotated_img = annotate_region(annotated_img, region['bbox'], annotation_color)

plt.imshow(annotated_img)
plt.show()