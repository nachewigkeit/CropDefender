import os
import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=r"saved_models/crop/encoder.pth")
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=r"E:\dataset\Caltech-101\subset")
    parser.add_argument('--save_dir', type=str, default=r"E:\dataset\stegastamp_crop")
    parser.add_argument('--secret', type=str, default='MITPBL')
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = torch.load(args.model).cuda()

    width = 400
    height = 400

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0).cuda()

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        with torch.no_grad():
            for filename in tqdm(files_list):
                image = Image.open(filename).convert("RGB")
                image = np.array(ImageOps.fit(image, size), dtype=np.float32)
                image = image.transpose((2, 0, 1))
                image = image / 255.
                image = torch.from_numpy(image).float().unsqueeze(0).cuda()

                residual = encoder((secret, image))
                encoded = torch.nn.Sigmoid()(image + residual - 0.5)
                encoded = np.array(encoded.squeeze(0).cpu() * 255, dtype=np.uint8).transpose((1, 2, 0))

                save_name = os.path.basename(filename).split('.')[0]

                im = Image.fromarray(encoded)
                im.save(args.save_dir + '/' + save_name + '_hidden.png')


if __name__ == "__main__":
    main()
