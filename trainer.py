import os
import sys
import time
import logging
import argparse
import torch
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, L1Loss, BCELoss

from customDataset import RGB_NIR_Dataset
from defaultConfig import get_config
from generator import RGB_IR_Converter as GenSWIN
from discriminator import SwinTransformerSys as DisSWIN
import torch.nn as nn


BATCH_SIZE = 4
NUM_WORKERS = 2
#HALF_TRAINED_PATH = ""
HALF_TRAINED_PATH = "trainedModels\\trainedModel_402.pt"

# DATAPATH = r"C:\Users\anshu\JupyterNotebooks\Seismic-Inpainting\dataset"
# assert os.path.exists(DATAPATH), "Dataset path invalid: {}".format(DATAPATH)

def setupLogger():
    """
        Logger setup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", '%m-%d-%Y %H:%M:%S')
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)
    logging.debug("Setting up logger completed")

def loadArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--lossName', default="mse", choices=["mse", "adaptive"], help='LossName')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    allArgs = parser.parse_args()
    return allArgs


if __name__ == "__main__":
    # setup logger
    setupLogger()
    # Load arguments
    allArgs = loadArguments()
    #import pdb
    #pdb.set_trace()
    config = get_config(allArgs)
    genModel = GenSWIN(config, img_size=allArgs.img_size, num_classes=allArgs.num_classes).cuda()
    # genModel.load_from(config)
    if HALF_TRAINED_PATH:
        genModel.load_state_dict(torch.load(HALF_TRAINED_PATH))
    else:
        genModel.load_from(config)

    trainContainer = RGB_NIR_Dataset("dataset\\rgb", "dataset\\nir")
    trainGenerator = DataLoader(trainContainer,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS)
    # Training
    genModel.train()
    lossObj = MSELoss()
    base_lr = allArgs.base_lr
    genOptimizer = optim.AdamW(genModel.parameters(), lr=base_lr, weight_decay=0.00001)
    iter_num = 0
    max_epoch = allArgs.max_epochs
    lossVecE = []
    #import pdb
    #pdb.set_trace()
    for epochNum in range(max_epoch):
        eStart = time.time()
        lossVecB = []
        for batchIdx, batchData in enumerate(trainGenerator):
            if os.path.exists("pauseTraining.txt"):
                os.remove("pauseTraining.txt")
                #import pdb
                #pdb.set_trace()
            startTime = time.time()
            rgbImg, nirImg = batchData
            rgbImg = rgbImg.cuda()
            rgbImg = rgbImg.float()
            nirImg = nirImg.cuda()
            genOp = genModel(rgbImg)

            # import pdb
            # pdb.set_trace()

            genLoss = lossObj(nirImg.float(), genOp)

            lossVecB.append(genLoss.mean())
            genOptimizer.zero_grad()
            genLoss.backward()
            genOptimizer.step()
            endTime = time.time()
            # print("Batch Time: {}".format(endTime - startTime))
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
        eEnd = time.time()
        lossEMean = sum(lossVecB) / float(len(lossVecB))
        print("Epoch Time {}: {} [Loss: {}]".format(epochNum + 1, eEnd - eStart, lossEMean))
        lossVecE.append(lossEMean)
        torch.save(genModel.state_dict(), "trainedModels\\trainedModel_{}.pt".format(epochNum + 1))
        if epochNum>2:
            del_epochNum = epochNum-2
            del_name = "trainedModels\\trainedModel_"+str(del_epochNum)+".pt"
            os.remove(del_name)
    #import pdb
    #pdb.set_trace()
    print("HOOK")

    torch.save(genModel.state_dict(), "trainedModel.pt")
    print("HOOK 2")


#python trainer.py --dataset other --cfg modifyConfig.yaml --root_path "C:\\Users\\anshu\\Seismic-Inpainting\\dataset" --max_epochs 1 --output_dir output --img_size 7200 --base_lr 0.05 --batch_size 4