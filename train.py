import os
import torch
import yaml
from easydict import EasyDict
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import lpips

import model
import utils
from dataset import StegaData
from time import time

import warnings

warnings.filterwarnings("ignore")

with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)


def main():
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    encoder = model.StegaStampEncoder()
    decoder = model.StegaStampDecoder(secret_size=args.secret_size)
    discriminator = model.Discriminator()
    loss_combine = model.LossCombine([0.1, 1, 1])
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)

    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        loss_combine = loss_combine.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()},
              {'params': loss_combine.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    dataset = StegaData(args.train_path, args.secret_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    height = 400
    width = 400
    global_step = 0

    start_time = time()
    while global_step < args.num_steps:
        for image_input, secret_input in dataloader:
            if image_input.shape[0] != args.batch_size:
                print("not enough batch")
                break

            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()

            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp,
                                   args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]

            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator,
                                                                            loss_combine, lpips_alex,
                                                                            secret_input, image_input,
                                                                            l2_edge_gain, Ms,
                                                                            loss_scales, yuv_scales,
                                                                            args, global_step, writer)

            optimize_loss.zero_grad()
            loss.backward()
            optimize_loss.step()
            if not args.no_gan:
                optimize_dis.zero_grad()
                D_loss.backward()
                optimize_dis.step()

            if global_step % 100 == 99:
                print("step: ", global_step + 1)
                print("speed: ", (time() - start_time) / 100)
                start_time = time()

    writer.close()
    torch.save(encoder, r"saved_models/encoder.pth")
    torch.save(decoder, r"saved_models/decoder.pth")


if __name__ == '__main__':
    main()
