# object_detection
##说明
-
该代码来源于tensorflow object_detection api，有兴趣得可以参考[原码](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)
##环境配置
###Dependencies
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
###COCO API installation
-
git clone https://github.com/cocodataset/cocoapi.git<br> 
cd cocoapi/PythonAPI<br> 
make<br> 
cp -r pycocotools /tf_models/<br> 
###Protobuf Compilation
-
\# From tf_models/<br> 
./bin/protoc object_detection/protos/*.proto --python_out=.<br> 
如果出错建议采用绝对路径<br> 
###Add Libraries to PYTHONPATH
###Linux
\# From tf_models/<br> 
export PYTHONPATH="$PYTHONPATH:`pwd`:`pwd`/slim"<br> 
'pwd'为tf_models绝对路径<br> 
###windows
在 ‘此电脑’-‘属性’- ‘高级系统设置’ -‘环境变量’-‘系统变量’ 中新建名为‘PYTHONPATH’的变量，将tf_models及 tf_models/slim 两个文件夹的完整目录添加
进去<br> 
##Testing the Installation
python object_detection/builders/model_builder_test.py<br> 
如果不出错，输出OK即为配置成功<br> 
