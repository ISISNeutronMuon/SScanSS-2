import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import os
import numpy as np
import time

filepath = r'D:\DummyDir\dataset_phantom_rebin122_150um'
start = time.perf_counter()
list_of_files = []
for file in os.listdir(filepath):
    if file.endswith(".tif"):
        a = tiff.imread(filepath + "\\" + str(file))
        list_of_files.append(file.title())

elapsed_time = time.perf_counter() - start
print("tifffile took: ", elapsed_time)


start = time.perf_counter()
list_of_files = []
for file in os.listdir(filepath):
    if file.endswith(".tif"):
        a = plt.imread(filepath + "\\" + str(file))
        list_of_files.append(file.title())

elapsed_time = time.perf_counter() - start
print("matplotlib took: ", elapsed_time)

start = time.perf_counter()
list_of_files = []
for file in os.listdir(filepath):
    if file.endswith(".tif"):
        a = Image.open(filepath + "\\" + str(file))
        a.load()
        list_of_files.append(file.title())

elapsed_time = time.perf_counter() - start
print("PIL took: ", elapsed_time)


#a = tiff.imread(filepath)
#print(a.nbytes)

# Getting % usage of virtual_memory ( 3rd field)
#print('RAM memory % used:', psutil.virtual_memory()[2])