#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

img1 = cv2.imread(r'your original image PATH')
img2 = cv2.imread(r'your comparion image PATH')

res_img1 = cv2.resize(img1, (256,256))
res_img2 = cv2.resize(img2, (256,256))

ssim = tf.image.ssim(res_img1, res_img2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

plt.imshow(img1,cmap="gray"), plt.axis("off")
plt.show()

plt.imshow(img2,cmap="gray"), plt.axis("off")
plt.show()

print(type(ssim.numpy()))
print("ACCCRACY SSIM %.4f" % ssim.numpy())
