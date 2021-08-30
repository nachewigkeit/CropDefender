import bchlib
from PIL import Image, ImageOps
import numpy as np
import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from model import StegaStampDecoder

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_bits(secret="MITPBL"):
    # 输入字符串，输出BCH码
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    return packet_binary


def get_model(model_path):
    # 输入模型参数路径, 输出模型
    decoder = torch.load(model_path).cuda()
    return decoder


def decode(image, model):
    # 输出模型与图片，输出预测结果（图片一定要是归一化到0-1区间的！）
    image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).cuda()
    secret = model(image)
    secret = np.array(secret[0].cpu())
    secret = np.round(secret)
    packet_binary = "".join([str(int(bit)) for bit in secret[:96]])

    return packet_binary


def get_acc(true, pred):
    # 输入预测二进制串与实际二进制串，输出准确率
    secret_size = len(true)
    count = 0
    for i in range(secret_size):
        if true[i] == pred[i]:
            count += 1
    acc = count / 96
    return acc


if __name__ == "__main__":

    dirPath = r"E:/dataset/stegastamp_crop"
    modelPath = r'saved_models/decoder.pth'
    file_list = glob.glob(dirPath + '/*.png')

    model = StegaStampDecoder().cuda()
    model.load_state_dict(torch.load(modelPath))

    model.eval()
    bitstring = get_bits()

    store = []
    with torch.no_grad():
        for file in tqdm(file_list):
            image = Image.open(file).convert("RGB")
            image = image.crop((50, 50, 350, 350))
            image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
            image /= 255.

            result = decode(image, model)
            store.append(get_acc(bitstring, result))

    plt.hist(store)
    plt.show()
    print(np.mean(store))
