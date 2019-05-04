from configuration import visualization_utils as vis_util
from configuration import label_map_util
import numpy as np
import sys
import os
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from PIL import Image
from configuration import ops as utils_ops
flags = tf.app.flags
flags.DEFINE_string('test_dataset', None,
                    'test-dataset path ')
flags.DEFINE_string('prediction_file', None,
                    'prediction file '
                    'file.')
flags.DEFINE_string('model', None,
                    'model path ')
FLAGS = flags.FLAGS
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height,im_width,3)).astype(np.uint8)

def image_detect(infile,sou_dir,detection_graph,category_index,sess):
    Time_str= time.strftime("%H:%M:%S", time.localtime(time.time()))
    print(Time_str)
    filename = os.path.join(sou_dir, infile)
    image = Image.open(filename)
    #image=image1.convert("RGB")
    detect_box=[]
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
    _,detect_box,class_star=vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),#正序排列
          detect_box,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
    Time_str = time.strftime("%H:%M:%S", time.localtime(time.time()))
    box_len=len(detect_box)
    detect_box=np.array(detect_box)
    infile=infile.split('_')[0]
    # print(infile,detect_box[0,0],detect_box[0,1],class_star)
    with open(FLAGS.prediction_file,'a',newline='') as fw:
      #fw.writelines(['id',',','x1',',','y1',',','x2',',','y2',',','x3',',','y3',',','havestar','\n'])
      if box_len==3:
        fw.writelines([infile,',',str(detect_box[0,0]),',',str(detect_box[0,1]),',',str(detect_box[1,0]),',',str(detect_box[1,1]),',',
        str(detect_box[2,0]),',',str(detect_box[2,1]),',',str(class_star),'\n'])
      elif box_len==2:
        fw.writelines([infile,',',str(detect_box[0,0]),',',str(ndetect_box[0,1]),',',str(detect_box[1,0]),',',str(detect_box[1,1]),',',
        str(detect_box[1,0]),',',str(detect_box[1,1]),',',str(class_star),'\n'])
      else:
        fw.writelines([infile,',',str(detect_box[0,0]),',',str(detect_box[0,1]),',',str(detect_box[1,0]),',',str(detect_box[1,1]),',',
        str(detect_box[0,0]),',',str(detect_box[0,1]),',',str(class_star),'\n'])
    #plt.imsave(des_dir+infile, image_np)
def main(_):
  sys.path.append("..")
  # if tf.__version__ < '1.4.0':
  #   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  PATH_TO_CKPT = FLAGS.model
  PATH_TO_LABELS = 'configuration/newstar_label_map.pbtxt'
  NUM_CLASSES = 2 
  tf.reset_default_graph()    
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')       
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, 
    max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  #print(category_index)
  sou_dir=FLAGS.test_dataset
  des_dir ='abc'
  test_file=os.listdir(sou_dir)
  # if not os.path.exists(des_dir):
  #     os.makedirs(des_dir)
  detection_graph = tf.get_default_graph()  
  with tf.Session(graph=detection_graph) as sess:
    for img in test_file:
      print(img)

      image_detect(img,sou_dir,detection_graph,category_index,sess)
if __name__ == '__main__':
  tf.app.run()