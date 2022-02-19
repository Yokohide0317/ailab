import os
import sys
#root_dir = "/Users/yokoda/workspace/yokoscripts/clean_img/"
#sys.path.append(root_dir+"Real-CUGAN")
sys.path.append("./src")
import cv2
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime


def main(file_name):
    upscaler = RealWaifuUpScaler(2, f"./src/model/up2x-latest-no-denoise.pth", half=False, device="cpu")
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
