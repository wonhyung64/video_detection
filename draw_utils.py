#%%
import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont

#%%
def draw_outputs(img, outputs, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.int32)
    boxes, objectness, classes = outputs
    boxes, objectness, classes = boxes[0], objectness[0], classes[0] 

    # img = tf.squeeze(img, axis=0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(font = "C:/Users/USER/Documents/GitHub/video_detection/fonts/futur.ttf",
    #                         size=(img.size[0] + img.size[1]) // 100)

    for index, bbox in enumerate(boxes):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis = -1)

        final_labels_ = tf.reshape(classes, shape=(classes.shape[0],))
        final_scores_ = tf.reshape(objectness, shape=(objectness.shape[0],))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index])
        label_text = "{0} {1:0.3f}".format(class_names[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    rgb_img = img.convert("RGB")
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img



