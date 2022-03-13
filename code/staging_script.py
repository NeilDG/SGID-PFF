import os
import glob
import inference
import numpy as np
import torch
from optparse import OptionParser
import sys
import cv2
from model.dehaze_sgid_pff import DEHAZE_SGID_PFF
from torchvision.utils import save_image
from torchvision.transforms import transforms

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")

def numpy2tensor(input, rgb_range=1.):
    img = np.array(input).astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
    tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
    tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
    tensor = tensor.unsqueeze(0)
    return tensor


def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img

def main(argv):
    (opts, args) = parser.parse_args(argv)
    # HAZY_PATH = "E:/Hazy Dataset Benchmark/OTS_BETA/haze/"
    # hazy_list = glob.glob(HAZY_PATH + "*0.95_0.2.jpg")  # specify atmosphere intensity

    # HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
    # hazy_list = glob.glob(HAZY_PATH + "*.jpeg")

    HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    #
    # HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    # hazy_list = glob.glob(HAZY_PATH + "*.jpg")

    print(hazy_list)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    net = DEHAZE_SGID_PFF(img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device=device)
    net.load_state_dict(torch.load("D:/Documents/GithubProjects/SGID-PFF/pretrained_models/SOTS_outdoor.pt"), strict=False)
    net = net.to(device)
    print("Loaded model")

    RESULTS_PATH = "./results/"
    transform_op = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for i, hazy_path in enumerate(hazy_list, 0):
        name = hazy_path.split("\\")[-1].split(".jpg")[0]
        print(name)

        hazy_img = cv2.imread(hazy_path)
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        hazy_img = transform_op(hazy_img)
        hazy_img = torch.unsqueeze(hazy_img, 0)
        hazy_img = hazy_img.to(device)

        pre_est_J, dehaze_img, trans, air, mid_loss = net(hazy_img)
        save_image(dehaze_img, RESULTS_PATH + name + ".jpg")


if __name__ == "__main__":
    main(sys.argv)