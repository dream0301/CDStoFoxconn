# object_detection
## 说明
该代码来源于tensorflow object_detection api，有兴趣得可以参考[原码](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)
## 环境配置
### Dependencies
Tensorflow Object Detection API depends on the following libraries:
* Protobuf 3.0.0（我的protoc-3.4.0-win32.zip，protoc-3.5.0-linux-x86_64.zip，下载地址在[这里](https://github.com/google/protobuf/releases)）  
* Python-tk  
* Pillow 1.0  
* lxml  
* Jupyter notebook  
* Matplotlib  
* Tensorflow (>=1.12.0,低于1.12.0会报错)  
* Cython  
* contextlib2  
* cocoapi

\# For CPU<br>
pip install tensorflow<br> 
\# For GPU<br>
pip install tensorflow-gpu<br> 
### COCO API installation
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    cp -r pycocotools /tf_models/ 
### Protobuf Compilation
\# From tf_models/<br> 
    
    ./bin/protoc object_detection/protos/*.proto --python_out=.<br> 
如果出错建议采用绝对路径<br> 
### Add Libraries to PYTHONPATH
#### Linux
\# From tf_models/<br> 

    export PYTHONPATH="$PYTHONPATH:`pwd`:`pwd`/slim"<br> 
'pwd'为tf_models绝对路径<br> 
#### windows
在 ‘此电脑’-‘属性’- ‘高级系统设置’ -‘环境变量’-‘系统变量’ 中新建名为‘PYTHONPATH’的变量，将tf_models及 tf_models/slim 两个文件夹的完整目录添加
进去<br> 
## Testing the Installation
    python object_detection/builders/model_builder_test.py<br> 
如果不出错，输出OK即为配置成功<br> 
## 训练集管道配置
1、新建xxx.pbtxt文件，添加如下内容，注意id从1开始，0留作背景使用，id数为类别数
\# From tf_models/object_detection/data
  
        item {
          id: 1
          name: 'aeroplane'
        }
        item {
          id: 2
          name: 'bicycle'
        }
2、 新建xxx.config文件,可从tf_models/object_detection/samples/configs中找到需要的配置文件进行修改
\# From tf_models/object_detection/XXX
主要修改如下内容

    num_classes: 2 \#为自己类别数量
    
    fine_tune_checkpoint: "object_detection/CHECKPOINT_PATH/model.ckpt"
    
    train_input_reader: {
      tf_record_input_reader {
        input_path: "object_detection/data/train.record"
      }
      label_map_path: "object_deteciton/data/xxx.pbtxt"
    }

    eval_config: {
      metrics_set: "coco_detection_metrics"
      num_examples: 1101
    }

    eval_input_reader: {
      tf_record_input_reader {
        input_path: "object_detection/data/val.record"
      }
      label_map_path: "object_deteciton/data/xxx.pbtxt"
      shuffle: false
      num_readers: 1
    }
CHECKPOINT_PATH为自己的预训练节点路径,num_examples为验证集数量，metrics_set为评价指标  
## 数据预处理
1、利用CSV生成tfrecords  
（1）、运行check_imgs.py检查图片与CSNV信息准确  
（2）、运行generate_tfrecord.py 生成tfrecords,命令为：<br>

    python generate_tfrecord.py --csv_input=${CSV_PATH}  --image_dir=${IMGGES_DIR}  --output_path=${OUTPUT_PATH}
${CSV_PATH}为CSV文件路径，${IMGGES_DIR}为图像路径，${OUTPUT_PATH}为生成得tfrecord文件<br>
2、利用图像与XML文件生成tfrecord，命令为：  
    
    python dataset_tools/create_disk_tf_record.py --data_dir=${DATA_DIR} --images_dir=${IMGGES_DIR}<br>
    --annotations_dir=${ANNOTATIONS_DIR} --output_path=${OUTPUT_PATH}
${DATA_DIR}为数据根目录，${IMGGES_DIR}为图像路径，${ANNOTATIONS_DIR}为XML文件路径，都位于${DATA_DIR}内，${OUTPUT_PATH}为生成得tfrecord文件<br>
## 训练
\# From tf_models/

    python object_detection/model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} \
        --num_train_steps=${NUM_TRAIN_STEPS} \  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
        --alsologtostderr
${PIPELINE_CONFIG_PATH}为xxx.config文件路径，${MODEL_DIR}为检查点保存路径可与xxx.config设置到一起，${NUM_TRAIN_STEPS} 为训练的steps数量  
$SAMPLE_1_OF_N_EVAL_EXAMPLES一般设置为1，即每个验证集数据都评估<br>
 `trick-限制GPU数`
 （1）将CUDA_VISIBLE_DEVICES=1,置于训练命令前用于选择GPU，1指训练所用的GPU号，或者将如下代码置于model_main.py前  
 
     import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
 （2）定量设置显存，0.7为所占显存比
 
     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
将第1行更改为tf.GPUOptions(allow_growth=True)即是按需设置显存
 `trick-进程转入后台` 
 >output.out 2>&1 &可置于训练命令末尾，在服务器中可用于将训练进程转入后台
