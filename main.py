#!/usr/bin/env python3
from pattern_recognition.signal_processing import SimplePitchDetector
from pattern_recognition.image_processing import normalized_cut, sift_descriptor, sobel_filter
from pattern_recognition.markov_models import gaussiam_hmm, lda
import matplotlib.pyplot as plt

"""
#####################
# signal processing #
#####################

spd = SimplePitchDetector(file_path='datasets/audio_data/simple_piano.wav')
_, pitches = spd.detect_pitch(window_size=5, lag_interval=(1,10))

#plt.acorr(spd.timeseries, maxlags=10)
plt.plot(pitches, 'bo')
plt.xlabel('Tau')
plt.ylabel('Pitch')
plt.show()

####################
# image processing #
####################

from skimage.io import imshow
fig = normalized_cut('datasets/msrc_data/1_1_s.bmp')
plt.imshow(fig)

kp, sift_desc = sift_descriptor('datasets/msrc_data/1_1_s.bmp')
plt.imshow(sift_desc)

x, y = sobel_filter('datasets/msrc_data/6_7_s.bmp')
plt.imshow(x)
plt.imshow(y)

#################
# markov models #
#################

pred_a, pred_l = gaussiam_hmm()
plt.plot(pred_a)
plt.plot(pred_l)
plt.show()
"""
lda_model = lda()
