import os
from tqdm import tqdm
from argparse import ArgumentParser
import imageio

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-d', "--imgdir", type=str)
    parser.add_argument('-o', "--output", type=str)
    parser.add_argument("--fps", default=60, type=int)
    args = parser.parse_args()
    
    imglist = sorted(os.listdir(args.imgdir))
    framelist = []
    img0 = imageio.v3.imread(os.path.join(args.imgdir, imglist[0]))
    print('resolution: ', img0.shape)
    for imgname in tqdm(imglist):
        framelist.append(imageio.v3.imread(os.path.join(args.imgdir, imgname)))
    imageio.mimwrite(f"{args.output}.mp4", framelist, fps=args.fps, quality=8)