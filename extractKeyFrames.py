import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
#from getFrames import getBatchFramesFromSingleVideo
from twoSteps import processVideoKeyFrames


def exponential_smoothing(alpha, s):
    '''
    Primary exponential smoothing
    :param alpha:  Smoothing factor,num
    :param s:      List of data,list
    :return:       List of data after smoothing,list
    '''
    s_temp = [s[0]]
    print(s_temp)
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i - 1] + (1 - alpha) * s_temp[i - 1])
    return s_temp


def process_image(image):
    '''
    Graying and GaussianBlur
    :param image: The image matrix,np.array
    :return: The processed image matrix,np.array
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # 高斯滤波
    return gray_image


def abs_diff(pre_image, curr_image):
    '''
    Calculate absolute difference between pre_image and curr_image
    :param pre_image:The image in past frame,np.array
    :param curr_image:The image in current frame,np.array
    :return:
    '''
    gray_pre_image = process_image(pre_image)
    gray_curr_image = process_image(curr_image)
    diff = cv2.absdiff(gray_pre_image, gray_curr_image)
    res, diff = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #  fixme：这里先写成简单加和的形式
    cnt_diff = np.sum(np.sum(diff))
    return cnt_diff


class KeyFrameGetter:
    '''
    Get the key frame
    '''

    def __init__(self, video_path, img_path, window=25):
        '''
        Define the param in model.
        :param video_path: The path points to the movie data,str
        :param img_path: The path we save the image,str
        :param window: The comparing domain which decide how wide the peek serve.
        '''
        self.window = window  # 关键帧数量
        self.video_path = video_path  # 视频路径
        self.img_path = img_path  # 图像存放路径
        self.diff = []
        self.idx = []

    def load_diff_between_frm(self, smooth=False, alpha=0.07):
        '''
        Calculate and get the model param
        :param smooth: Decide if you want to smooth the difference.
        :param alpha: Difference factor
        :return:
        '''
        print("load_diff_between_frm")
        cap = cv2.VideoCapture(self.video_path)  # 打开视频文件
        diff = []
        frm = 0
        pre_image = np.array([])
        curr_image = np.array([])

        while True:

            # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
            # cv2.imwrite('./img/' + str(num) + ".jpg", data)
            success, data = cap.read()
            if not success:
                break
            #  这里是主体
            #  记录每次的数据
            if frm == 0:
                pre_image = data
                curr_image = data
            else:
                pre_image = curr_image
                curr_image = data
            #  进行差分
            diff.append(abs_diff(pre_image, curr_image))

            frm = frm + 1
            #  结尾Loop
            if frm % 1000 == 0:
                print('Detect Frame:', str(frm))
        cap.release()

        if smooth:
            diff = exponential_smoothing(alpha, diff)
        #  标准化数据
        self.diff = np.array(diff)
        mean = np.mean(self.diff)
        dev = np.std(self.diff)
        self.diff = (self.diff - mean) / dev

        #  在内部完成
        self.pick_idx()
        return

    def pick_idx(self):
        '''
        Get the index which accord to the frame we want(peek in the window)
        :return:
        '''
        print("pick_idx")
        for i, d in enumerate(self.diff):
            ub = len(self.diff) - 1
            lb = 0
            if not i - self.window // 2 < lb:
                lb = i - self.window // 2
            if not i + self.window // 2 > ub:
                ub = i + self.window // 2

            comp_window = self.diff[lb:ub]
            if len(comp_window) and d >= max(comp_window):
                self.idx.append(i)

        tmp = np.array(self.idx)
        tmp = tmp + 1  # to make up the gap when diff
        self.idx = tmp.tolist()
        print("Extract the Frame Index:" + str(self.idx))

    def save_key_frame(self):
        '''
        Save the key frame image
        :return:
        '''
        print("save_key_frame")
        cap = cv2.VideoCapture(self.video_path)  # 打开视频文件
        frm = 0
        idx = set(self.idx)
        success, data = cap.read()
        while success:
            frm = frm + 1
            if frm in idx:
                # print('Extracting idx:'+str(frm))
                #[ret] = v8model(success, save_crop=True)
                #box = ret.boxes[0]
                # print(self.img_path + '/' + str(frm) + ".jpg",
                #      '; 帧：', frm, ', 获取的预测牛数量', len(ret.boxes))
                cv2.imwrite(self.img_path + '/' + str(frm) + ".jpg", data)
                idx.remove(frm)
            if not idx:
                print('Done！')
                break
            success, data = cap.read()

    def plot_diff_time(self):
        '''
        Plot the distribution of the difference along to the frame increasing.
        :return:
        '''
        myfig = plt.gcf()
        plt.plot(self.diff, '-b')
        plt.plot(np.array(self.idx) - 1,
                 [self.diff[i - 1] for i in self.idx], 'or')
        plt.xlabel('Frame Pair Index')
        plt.ylabel('Difference')
        plt.legend(['Each Frame', 'Extract Frame'])
        plt.title("The Difference for Each Pair of Frame")
        plt.plot()
        plt.show()
        file_path = os.path.splitext(self.video_path)[0]
        file_name = file_path.split('/')[-1]
        name = self.img_path + '/' + file_name + '_diff.png'
        myfig.savefig(name)


def getCorpedFrames(videoDir, framesDir):
    framesArr = []

    for root, dirs, files in os.walk(videoDir):
        for file in sorted(files):

            source_path = os.path.join(root, file)
            # dir_path = framesDir + file.strip('.avi').strip('.mp4')  # + '/'
            dir_path = framesDir + os.path.splitext(file)[0]  # + '/'

            print("源文件目录为", root)
            print("源文件路径为", source_path)
            print("目的文件路径为", dir_path)

            if os.path.exists(dir_path):
                continue
                #os.rename(dir_path, dir_path + '.bak.' + str(time.time()))
            os.makedirs(dir_path)

            kfg = KeyFrameGetter(source_path, dir_path, 100)
            a = time.time()
            kfg.load_diff_between_frm(alpha=0.07)  # 获取模型参数
            print(kfg.idx)

            frameNum = processVideoKeyFrames(source_path, kfg.idx, dir_path)
            framesArr.append([file, frameNum])
            #b = time.time()
            #sa.append(b - a)
            # 耗时统计
            # print(sa)
            # kfg.save_key_frame()  # 存下index列表里对应图片

            # 此语句是打印每一个图片的差分信息
            kfg.plot_diff_time()
    cols = ['file', 'count']
    df = pd.DataFrame(framesArr, columns=cols)
    df.to_csv(
        '/content/drive/MyDrive/bi-seq-202302/standing2lying/results-final/framesStatistics.csv')


def oneStepProcess(videoDir, framesDir):
    framesArr = []

    for root, dirs, files in os.walk(videoDir):
        for file in sorted(files):
            source_path = os.path.join(root, file)
            dir_path = framesDir + os.path.splitext(file)[0]  # + '/'

            print("源文件目录为", root)
            print("源文件路径为", source_path)
            print("目的文件路径为", dir_path)

            if os.path.exists(dir_path):
                continue
                #os.rename(dir_path, dir_path + '.bak.' + str(time.time()))
            os.makedirs(dir_path)

            kfg = KeyFrameGetter(source_path, dir_path, 100)
            a = time.time()
            kfg.load_diff_between_frm(alpha=0.07)  # 获取模型参数
            print(kfg.idx)

            frameNum = processVideoKeyFrames(source_path, kfg.idx, dir_path)
            framesArr.append([file, frameNum])

            kfg.plot_diff_time()
    cols = ['file', 'count']
    df = pd.DataFrame(framesArr, columns=cols)
    df.to_csv(
        '/content/drive/MyDrive/bi-seq-202302/standing2lying/results-final/framesStatistics.csv')


if __name__ == '__main__':

    videoDir = sys.argv[1]
    # 需要确定的地址
    #videoDir = '/content/drive/MyDrive/bi-seq-202302/videos/316videos/me'
    #videoDir = '/content/video'
    framesDir = '/content/drive/MyDrive/bi-seq-202302/standing2lying/img-final/'
    print('framesDir: ', framesDir)
    getCorpedFrames(videoDir, framesDir)
