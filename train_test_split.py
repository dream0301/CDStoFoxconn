# -*- coding: utf-8 -*-
#author: GHW
#数据集拆分为训练集，验证集，测试集
import re,os
import random
import numpy as np
from PIL import Image
import shutil
def read_write(csv_filepath):
    csv_file = os.path.join(csv_filepath, 'list.csv')
    with open(csv_file,'r') as fr:
        lines = fr.readlines()
        del lines[0]
        data=[]
        for row in lines:
            b = re.split(',|\n', row)
            row = [i for i in b if i]
            data.append(row)
    positivename = ['newtarget', 'isstar', 'known', 'isnova', 'asteroid']
    negativename = ['noise', 'ghost', 'pity']
    dataset=np.array(data)
    positive=[]
    #negative=[]
    for x in data:
        if str(x[3])  in positivename:
            x[3]='1'
            positive.append(x)
            print(x)
        if x[3] in negativename:
            print(x[3])
            x[3]='0'
            positive.append(x)
    # positive_len=len(positive)
    # data_num = random.sample(range(len(negative)),positive_len)
    # for x in data_num:
    #     positive.append(negative[x])


    csv_file = os.path.join(csv_filepath, 'result.csv')
    with open(csv_file,'a',newline='') as fw:
        # fw.writelines(['filename',',', 'width',',', 'height',',',\
        #                'xmin',',', 'ymin',',', 'xmax',',', 'ymax',',', 'class'])
        for x in positive:
            img = Image.open(csv_filepath+'supernova_dataset_RGB\\'+x[0]+'_a.jpg')
            width=img.size[0]
            height=img.size[1]
            xmin=max(2,int(x[1])-15)
            ymin=max(2,int(x[2])-15)
            xmax=min(width-1,int(x[1])+15)
            ymax=min(height-1,int(x[2])+15)
            fw.writelines([str(x[0]),',',str(width),',',str(height),',',str(xmin),',',str(ymin),',',\
                                    str(xmax),',',str(ymax),',',x[3],'\n'])

def train_test_split(csv_filepath):
    csv_file = os.path.join(csv_filepath, 'result.csv')
    with open(csv_file,'r') as fr:
        lines = fr.readlines()
        del lines[0]
        dataset=[]
        for row in lines:
            b = re.split(',', row)
            row = [i for i in b if i]
            dataset.append(row)
    data_arr=np.array(dataset)
    sample_len=range(len(dataset))
    trainval=random.sample(sample_len,int(len(dataset)*0.84) )
    #train=random.sample(trainval,int(len(dataset)*0.85) )
    train_file=os.path.join(csv_filepath, 'train.csv')
    val_file=os.path.join(csv_filepath,'val.csv')
    test_file=os.path.join(csv_filepath,'test.csv')
    with open(train_file, 'a', newline='') as fw:
        fw.writelines(['filename',',', 'width',',', 'height',',',\
                       'xmin',',', 'ymin',',', 'xmax',',', 'ymax',',', 'class','\n'])
        for x in trainval:
            fw.writelines([str(data_arr[x,0]) , ',',data_arr[x,1], ',',data_arr[x,2], ',', \
                           data_arr[x,3], ',', data_arr[x,4], ',', data_arr[x,5], ',', data_arr[x,6], ',', data_arr[x,7]])
    # with open(val_file, 'a', newline='') as fw:
    #     fw.writelines(['filename',',', 'width',',', 'height',',',\
    #                    'xmin',',', 'ymin',',', 'xmax',',', 'ymax',',', 'class','\n'])
    #     for x in trainval:
    #         if x not in train:
    #             fw.writelines([str(data_arr[x, 0]), ',', data_arr[x, 1], ',', data_arr[x, 2], ',', \
    #                            data_arr[x, 3], ',', data_arr[x, 4], ',', data_arr[x, 5], ',', data_arr[x, 6], ',',
    #                            data_arr[x, 7]])

    with open(test_file, 'a', newline='') as fw:
        fw.writelines(['filename', ',', 'width', ',', 'height', ',', \
                       'xmin', ',', 'ymin', ',', 'xmax', ',', 'ymax', ',', 'class','\n'])
        for x in sample_len:
            if x not in trainval:
                fw.writelines([str(data_arr[x, 0]), ','
                                  , data_arr[x, 1], ',', data_arr[x, 2], ',', \
                               data_arr[x, 3], ',', data_arr[x, 4], ',', data_arr[x, 5], ',', data_arr[x, 6],
                               ',', data_arr[x, 7]])
                oldfile=os.path.join(csv_filepath+'supernova_dataset_RGB\\',str(data_arr[x, 0])+'_a.jpg')
                newfile=os.path.join(csv_filepath+'supernova_testset_RGB\\',str(data_arr[x, 0])+'_a.jpg')
                shutil.move(oldfile,newfile)
                oldfile=os.path.join(csv_filepath+'supernova_dataset_RGB\\',str(data_arr[x, 0])+'_b.jpg')
                newfile=os.path.join(csv_filepath+'supernova_testset_RGB\\',str(data_arr[x, 0])+'_b.jpg')
                shutil.move(oldfile,newfile)
                oldfile=os.path.join(csv_filepath+'supernova_dataset_RGB\\',str(data_arr[x, 0])+'_c.jpg')
                newfile=os.path.join(csv_filepath+'supernova_testset_RGB\\',str(data_arr[x, 0])+'_c.jpg')
                shutil.move(oldfile,newfile)


if __name__ == '__main__':
    #csv_filepath = 'E:\\Tofind_newstar\\af2019-cv-training-20190312\\'
    new_filwpath='E:\\tf_models_hw\\supernova_dataset\\'
    #read_write(new_filwpath)
    train_test_split(new_filwpath)




