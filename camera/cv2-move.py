# coding:utf-8

import cv2
import numpy as np
import io
from PIL import Image
import socket

# 调用摄像头
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
s.bind(("0.0.0.0", 9090))


kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    # 读入摄像头的帧
    data, IP = s.recvfrom(100000)
    bytes_stream = io.BytesIO(data)
    image = Image.open(bytes_stream)
    img = np.asarray(image)
    # 把第一帧作为背景
    if background is None:
        background = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue
    # 读入帧
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯平滑 模糊处理 减小光照 震动等原因产生的噪声影响
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # 检测背景和帧的区别
    diff = cv2.absdiff(background, gray_frame)
    # 将区别转为二值
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    # 定义结构元素
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    # 膨胀运算
    diff = cv2.dilate(diff, es, iterations=2)
    # 搜索轮廓
    cnts, hierarcchy = cv2.findContours(diff.copy(),
                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # 轮廓太小忽略 有可能是斑点噪声
        if cv2.contourArea(c) < 1500:
            continue
        # 将轮廓画出来
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("contours", img)
    cv2.imshow("diff", diff)
    if cv2.waitKey(5) & 0xff == ord("q"):
        break

