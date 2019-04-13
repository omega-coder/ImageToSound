#!/usr/bin/env python

import math
import struct
import wave

import numpy as np
import scipy.ndimage
from PIL import Image

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)
# for resampling using nearest neighbour


def im_load(file, size, contrast=True):
    img = Image.open(file)
    img = img.convert("L")

    im_array = np.array(img)
    im_array = np.flip(im_array, axis=0)

    if contrast:
        im_array = 1/(im_array+10**15.2)
    else:
        im_array = 1 - im_array


    im_array -= np.min(im_array)
    im_array = im_array / np.max(im_array)

    # remove low pixel values
    # remove_low_pix = np.vectorize(lambda x: x if x > 0.5 else 0, otypes=[np.float])
    # im_array = remove_low_pix(im_array)

    if size[0] == 0:
        size = im_array.shape[0], size[1]
    if size[1] == 0:
        size = size[0], im_array.shape[1]


    res_factor = size[0] / im_array.shape[0], size[1] / im_array.shape[1]
    if res_factor[0] == 0:
        res_factor = 1, res_factor[1]
    if res_factor[1] == 0:
        res_factor = res_factor[0], 1

    im_array = scipy.ndimage.zoom(im_array, res_factor, order=0)

    # order=0 : nearest neighbour

    return im_array

def make_sound(file, out="out.wav", duration=6.9, sample_rate=44100.0, min_freq=0, max_freq=22000, intensity_fact=1, contrast=True):
    wave_f__ = wave.open(out, "w") 
    wave_f__.setnchannels(1)
    wave_f__.setsampwidth(2)
    wave_f__.setframerate(sample_rate)

    max_frames = int(duration * sample_rate)
    max_intens = 32767

    step_size = 400
    spec_step = int( (max_freq - min_freq) / step_size )

    im_array = im_load(file=file, size=(spec_step, max_frames), contrast=contrast)

    im_array *= intensity_fact
    im_array *= max_intens

    for frame in range(max_frames):
        if frame % 100 == 0:
            print("Total Progress = {:.2%}".format((frame/max_frames)), end="\r") 
        sig_val, count = 0, 0
        for step in range(spec_step):
            intensity = im_array[step, frame]
            if intensity < 0.1 * intensity_fact:
                continue
            current_freq = (step * step_size) + min_freq
            next_freq    = ((step+1) * step_size ) + min_freq

            if next_freq - min_freq > max_freq:
                next_freq = max_freq

            for freq in range(current_freq, next_freq, 1000):
                sig_val += intensity*math.cos(freq * 2 * math.pi * float(frame) / float(sample_rate))
                count += 1
        if count == 0:
            count = 1
        sig_val /= count

        data = struct.pack('<h', int(sig_val))
        wave_f__.writeframesraw(data)
    wave_f__.writeframesraw("".encode())
    wave_f__.close()

if __name__ == "__main__":
    import sys
    argv = sys.argv
    make_sound(file=argv[1], out=argv[2])














