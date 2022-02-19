import os
import sys
sys.path.append(".")
import cv2
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime

ModelPath="./model/up2x-latest-no-denoise.pth"
Tile=4

def main(file_path):
    file_name = file_path.split("/")[-1]
    img_name = file_name.split(".")[0]
    file_type = file_name.split(".")[-1]
    
    upscaler = RealWaifuUpScaler(2, ModelPath, half=False, device="cpu")
    try:
        img = cv2.imread(file_path)[:, :, [2, 1, 0]]
        result = upscaler(img,tile_mode=2)
        cv2.imwrite(f"./{img_name}_cleaned.{file_type}",result[:, :, ::-1])
    except RuntimeError as e:
        print("Failed")
        print(e)
    else:
        print("Done")

if __name__ == "__main__":
    main(sys.argv[1])

