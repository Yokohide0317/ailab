import os
import sys
import cv2
from src.upcunet_v3 import RealWaifuUpScaler
from time import time as ttime

model_name = "up2x-latest-no-denoise.pth"
path_name = os.path.dirname(__file__)
model_path = os.path.join(path_name, "src/model", model_name)

def main(file_name):
    upscaler = RealWaifuUpScaler(2, model_path, half=False, device="cpu")
    Tile = 4
    Amplification = 2

    t0 = ttime()
    try:
        img = cv2.imread(file_name)[:, :, [2, 1, 0]]
        result = upscaler(img,tile_mode=2)
        output_name = ".".join(file_name.split(".")[:-1]) + "_cleaned." + file_name.split(".")[-1]
        cv2.imwrite(output_name, result[:, :, ::-1])
    except RuntimeError as e:
        print ("Failed...")
        print (e)
    else:
        print("Done")
    t1 = ttime()
    print("Compleated", t1 - t0)

if __name__ == "__main__":
    file_name = sys.argv[1]
    print(file_name)
    main(file_name)
