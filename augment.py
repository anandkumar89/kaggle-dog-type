import tensorflow as tf
from imgaug import augmenters as iaa
import cv2

# paths
train_dir = "train"
test_dir  = "test"

# preprocessing using cv2
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

sess = tf.Session()

# saver = tf.train.import_meta_graph("resnet_v2_101/res101v2.ckpt")
# 
# saver.restore(sess, tf.train.latest_checkpoint('./'))

# module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1")
# height, width = hub.get_expected_image_size(module)
# images = ...  # A batch of images with shape [batch_size, height, width, 3].and
# features = module(images)  # Features with shape [batch_size, num_features].

