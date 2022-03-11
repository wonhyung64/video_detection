#%%
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
import get_yolov3, postprocessing_utils, draw_utils

#%%
flags.DEFINE_string("classes", "C:/Users/USER/Documents/GitHub/video_detection/labels/coco.names", "path to classes file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string("video", "./data/video/paris.mp4",
                    "path to video file or number for webcam")
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_format", "XVID", "codec used in VideoWriter when saving video to file")
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")

#%%
def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = get_yolov3.yolo_v3()
    logging.info("weights loaded")

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]#
    logging.info("classes loaded")

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    fps = 0.0
    count = 0

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = tf.image.resize(img_in, (FLAGS.size, FLAGS.size)) / 255.

        t1 = time.time()
        yolo_outputs = model(img_in)
        bboxes, labels, scores = postprocessing_utils.Decode(yolo_outputs)
        fps = (fps + (1. / (time.time() - t1))) / 2

        img = draw_utils.draw_outputs(img, [bboxes, scores, labels], class_names)
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if FLAGS.output:
            out.write(img)
        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
        











