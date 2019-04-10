import numpy as np
import struct
import tensorflow as tf
import matplotlib.pyplot as plt


class DP:
    '''
    @param train_img_num
    @param train_label_num
    @param test_img_num
    @param test_label_num

    @param train_img_list
    @param train_label_list
    @param test_img_list
    @param test_label_list


    '''
    #获得一个60000 * 784的数组
    def read_train_image(self,filename):
        index = 0
        binfile = open(filename,'rb')
        buf = binfile.read()
        magic, self.train_img_num, self.numRows,self.numColums = struct.unpack_from('>IIII',buf,index)
        self.train_img_list = np.zeros((self.train_img_num, 28 * 28))
        index += struct.calcsize('>IIII')
        #print (magic, ' ', self.train_img_num, ' ', self.numRows, ' ', self.numColums)

        for i in range(self.train_img_num):
            #每次读784b，即为一张图的大小，返回为一个元组
            im = struct.unpack_from('>784B',buf,index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            # print(im)
            im = im/255 #归一
            im = im.reshape(1,28*28)
            self.train_img_list[i,:] = im
            # im = im.reshape([28, 28])
            # plt.imshow(im,cmap='binary')
            # plt.show()
        print("reading train image finished")


    # 获得一个60000 * 10的数组
    def read_train_lable(self,filename):
        index = 0
        binfile = open(filename,'rb')
        buf = binfile.read()
        magic, self.train_label_num = struct.unpack_from('>II',buf,index)
        self.train_label_list = np.zeros((self.train_label_num, 10))
        index += struct.calcsize('>II')
        #print(magic,' ', self.train_label_num)
        for i in range(self.train_label_num):
            lblTemp = np.zeros(10)
            lbl = struct.unpack_from('>1B',buf,index)
            index += struct.calcsize('>1B')
            lbl = np.array(lbl)
            lblTemp[lbl[0]] = 1
            self.train_label_list[i,:] = lblTemp
            # print(lblTemp)
        print("reading train label finished")

    def next_batch_image(self):#1-59999抽一个数
        rnd = np.random.randint(1,60000)
        batchCount = 50
        return self.train_img_list[rnd:rnd+batchCount],self.train_label_list[rnd:rnd+batchCount]

    def read_test_image(self,filename):
        index = 0
        binfile = open(filename,'rb')
        buf = binfile.read()
        magic, self.test_img_num, self.numRows,self.numColums = struct.unpack_from('>IIII',buf,index)
        self.test_img_list = np.zeros((self.test_img_num, 28 * 28))
        index += struct.calcsize('>IIII')
        #print (magic, ' ', self.test_img_num, ' ', self.numRows, ' ', self.numColums)

        for i in range(self.test_img_num):
            im = struct.unpack_from('>784B',buf,index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            im = im/255
            im = im.reshape(1,28*28)
            self.test_img_list[i,:] = im

        print("reading test image finished")

    def read_test_lable(self,filename):
        index = 0
        binfile = open(filename,'rb')
        buf = binfile.read()
        magic, self.test_label_num = struct.unpack_from('>II',buf,index)
        self.test_label_list = np.zeros((self.test_label_num, 10))
        index += struct.calcsize('>II')
        #print(magic,' ', self.test_label_num)
        for i in range(self.test_label_num):
            lblTemp = np.zeros(10)
            lbl = struct.unpack_from('>1B',buf,index)
            index += struct.calcsize('>1B')
            lbl = np.array(lbl)
            lblTemp[lbl[0]] = 1
            self.test_label_list[i,:] = lblTemp
            # print(lblTemp)
        print("reading test label finished")

    def test_image_label(self):
        return self.test_img_list[0:9999], self.test_label_list[0:9999]

filename_t_image  = "E:\\study\\DL\\HJFaceRecognition\\minist\\Set\\train-images.idx3-ubyte"
filename_t_label  = "E:\\study\\DL\\HJFaceRecognition\\minist\\Set\\train-labels.idx1-ubyte"
filename_test_image  = "E:\\study\\DL\\HJFaceRecognition\\minist\\Set\\t10k-images.idx3-ubyte"
filename_test_label  = "E:\\study\\DL\\HJFaceRecognition\\minist\\Set\\t10k-labels.idx1-ubyte"
