#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
from keras.models import load_model


# def getV8(path='./weight/best317v8.pt'):
def getV8(path='./weight/best.pt'):
    print('yolo model weight路径', path)
    return YOLO(path)


def loadResnet(path='/content/drive/MyDrive/bi-seq-202302/standing2lying/weight/ResNet50_best.h5'):
    print('resnet model weight路径', path)
    return load_model(path)


if __name__ == '__main__':
    res = getV8()
    print(res)
