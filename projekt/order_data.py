import os
import shutil

PATH = "C:/Users/zlomi/OneDrive/Radna površina/projektR/data/complex_npy_segments"

labels = ["AR", "AS", "GR", "GS", "PR", "PS"]

for file in os.listdir(PATH):
    shutil.copy(PATH+'/'+file, f"C:/Users/zlomi/OneDrive/Radna površina/projektR/data/ordered_data/{ labels.index(file.split('_')[1])}")
