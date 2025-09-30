import os
from os import listdir
import os
import cv2 
from PIL import Image
import imageio
def gifgen(chara):
    folder_dir = "./Dataset/"+chara+"/Original"
    for images in os.listdir(folder_dir):
        if (images.endswith(".jpg")):
            with imageio.get_writer("generated/"+str(chara)+'.gif') as writer:
                for filename in images:
                    image = imageio.imread(filename)
                    writer.append_data(image)
def process(text):
    
    for i in text:
        if i.isalpha():
            chara=i.upper()
            gifgen(chara)
            
process("apple")
            
    
