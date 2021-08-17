#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from albumentations.pytorch import ToTensorV2
from vector_cv_tools import datasets
from vector_cv_tools import transforms as T
from vector_cv_tools import utils
from vector_cv_tools import checkpointing as vckpts

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
parser.add_argument("--image_folder",
                    type=str,
                    default="data/artifacts/images",
                    help="path to dataset")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="size of each image batch")
parser.add_argument("--model_config_path",
                    type=str,
                    default="object_detector/config/yolov3.cfg",
                    help="path to model config file")
parser.add_argument("--data_config_path",
                    type=str,
                    default="object_detector/config/coco.data",
                    help="path to data config file")
parser.add_argument("--weights_path",
                    type=str,
                    default="object_detector/config/yolov3.weights",
                    help="path to weights file")
parser.add_argument("--conf_thres",
                    type=float,
                    default=0.8,
                    help="object confidence threshold")
parser.add_argument("--nms_thres",
                    type=float,
                    default=0.4,
                    help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu",
                    type=int,
                    default=0,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size",
                    type=int,
                    default=416,
                    help="size of each image dimension")
parser.add_argument("--checkpoint_interval",
                    type=int,
                    default=10,
                    help="interval between saving model weights")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default="checkpoints",
                    help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda",
                    type=bool,
                    default=True,
                    help="whether to use cuda if available")
parser.add_argument('--eval', action='store_true')
parser.add_argument('-thb', '--test_hardcoded_boxes', action='store_true')
parser.add_argument('--data_dir',
                    type=str,
                    default="/scratch/ssd002/datasets/cv_project/forms_dataset")
parser.add_argument('--data_split', type=str, default="train")

# use this to differentiate between the different runs, otherwise
# every run will be saved in the same directory
parser.add_argument('--run_name', type=str, required=__name__ == '__main__')

opt = parser.parse_args()


def create_or_restore_training_state(model,
                                     optimizer,
                                     dir_path="./checkpoints/baseline_model"):
    cur_epoch = vckpts.SavebleNumber(0)
    prev_epoch_loss_list = vckpts.SaveableList()

    checkpoint = vckpts.SavableCollection(
        model=model,
        optimizer=optimizer,
        cur_epoch=cur_epoch,
        prev_epoch_loss_list=prev_epoch_loss_list,
    )

    manager = vckpts.CheckpointManager(
        checkpoint=checkpoint,
        directory=dir_path,
        max_to_keep=3,
        checkpoint_interval=1200,
        keep_best="max",
    )

    # this call does nothing if there is no checkpoint loaded
    # otherwise it loads from the latest checkpoint in the directory
    # specified above
    manager.load_latest_checkpoint()

    if manager.latest_checkpoint is not None:
        min_loss = (checkpoint.prev_epoch_loss_list[-1]
                    if len(checkpoint.prev_epoch_loss_list) > 0 else None)
        print_str = (
            f"Checkpoint that has finished epoch {checkpoint.cur_epoch} with "
            f"val loss {min_loss} is loaded from {manager.latest_checkpoint}")

    else:
        print_str = (f"No checkpoints found under {manager.directory}, "
                     "starting from scratch")

    print(print_str)

    cur_epoch.add_(1)
    return cur_epoch, model, optimizer, manager, prev_epoch_loss_list


def train():
    cuda = torch.cuda.is_available() and opt.use_cuda

    checkpoint_dir = os.path.join("./checkpoints", opt.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    train_path = data_config["train"]

    # Get hyper parameters
    hyperparams = parse_model_config(opt.model_config_path)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(opt.model_config_path)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()))

    if cuda:
        model = model.cuda()

    # load either non initialized or loaded state
    cur_epoch, model, optimizer, manager, prev_epoch_loss_list = create_or_restore_training_state(
        model, optimizer, dir_path=checkpoint_dir)

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
    )

    # transform_data = partial(T.transform_data, img_shape=(opt.img_size, opt.img_size))
    # transform = [ToTensorV2()]    
    # transform = T.ComposeFUNSDTransform(transform, post_process_fn=transform_data)
    # dset = datasets.FUNSD(opt.data_dir,
    #                       split=opt.data_split,
    #                       transforms=transform,
    #                       load_qa_only=True,
    #                       load_linked_only=True,
    #                       linking_limit=1)

    # dataloader = DataLoader(dset,
    #                         num_workers=opt.n_cpu,
    #                         batch_size=opt.batch_size,
    #                         shuffle=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    while cur_epoch < opt.epochs + 1:

        epoch_loss_list = []

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    cur_epoch,
                    opt.epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                ))

            model.seen += imgs.size(0)

            epoch_loss_list.append(loss.item())

        if cur_epoch % opt.checkpoint_interval == 0:
            # predict('object_detector/images/82092117.png', model, cur_epoch)
            # predict('object_detector/images/BloodImage_00001_jpg.rf.3f768de1133398b7558743883a808be7.jpg', model, cur_epoch)
            predict('object_detector/images/results2.jpg', model, cur_epoch)
           
        epoch_loss = 0 if len(epoch_loss_list) == 0 else sum(
            epoch_loss_list) / len(epoch_loss_list)
        print("Loss over epoch {} is {}".format(cur_epoch, epoch_loss))

        prev_epoch_loss_list.append(epoch_loss)
        manager.save(global_step=cur_epoch, metric=epoch_loss, do_logging=True)

        cur_epoch.add_(1)

def predict(img_path, model, epoch):
    img = Image.open(img_path)
    img = np.array(img)
    if len(img.shape) != 3:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        img = Image.fromarray(img)
    else:
        img = Image.fromarray(img)

    detections = detect_image(img, opt.img_size, model)
    return plot_box(
                img, opt.img_size, detections, (1, 0, 0),
                img_path.replace(".jpg", "_{}.jpg".format(epoch)).replace(
                    ".png", "_{}.png".format(epoch)), "locations")
    
def detect_image(img, img_size, model):

    # scale and pad image
    conf_thres = 0.8
    nms_thres = 0.4
    Tensor = torch.cuda.FloatTensor
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)

    img_transforms = transforms.Compose([
        transforms.Resize((imh, imw)),
        transforms.Pad((max(int(
            (imh - imw) / 2), 0), max(int(
                (imw - imh) / 2), 0), max(int(
                    (imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                       (128, 128, 128)),
        transforms.ToTensor(),
    ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = image_tensor
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def test_hardcoded_boxes():
    img_size = opt.img_size
    # img_path = "object_detector/images/82092117.png"  # form
    img_path = "object_detector/images/results2.jpg" # test_form
    # img_path = 'images/000012.jpg' # car
    prev_time = time.time()
    img = Image.open(img_path)

    # detections = torch.tensor([[116, 352, 27, 15, 0.9, 0.9, 4]]) # form field 1
    # detections = torch.tensor([[188, 512, 171, 17, 0.9, 0.9, 4]])  # form field 2
    detections = torch.tensor([[0.419102*1916, 0.477005*935, 0.188935*1916, 0.044920*935, 0.9, 0.9, 4]]) # test_form
    # detections = torch.tensor([[253.5, 183.499999999983, 195.0, 173.00000000016, 0.9, 0.9, 4]]) # car

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    plot_box(img, img_size, detections, (1, 0, 0),
             img_path.replace(".jpg", "-det.jpg").replace(".png", "-det.png"),
             "width_height")


def evaluation():
    img_size = opt.img_size

    # Load model and weights
    model = Darknet(opt.model_config_path, img_size=img_size)
    model.load_state_dict(torch.load(opt.weights_path)["model"])

    model.cuda()
    model.eval()

    # load image and get detections
    img_path = "images/82092117.png"  # form
    # img_path = 'images/000012.jpg' # car
    prev_time = time.time()
    img = Image.open(img_path)
    detections = detect_image(img, img_size, model)

    # detections = torch.tensor([[0.5070, 0.5330, 0.3900, 0.3460, 0.9, 0.9, 3]])
    # detections = torch.tensor([[10.3952, 23.7931,  1.0586,  1.0345, 0.9, 0.9, 4]])
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print('Inference Time: %s' % (inference_time))

    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    plot_box(img, img_size, detections, (1, 0, 0),
             img_path.replace(".jpg", "-det.jpg").replace(".png", "-det.png"),
             "width_height")


def plot_box(img,
             img_size,
             detections,
             color,
             img_path,
             prediction_type='width_height'):
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    all_bbox = []
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if prediction_type == 'width_height':
                all_bbox.append([x1, y1, x2, y2])
                bbox = patches.Rectangle((x1, y1),
                                         x2,
                                         y2,
                                         linewidth=2,
                                         edgecolor=color,
                                         facecolor='none')
            else:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                all_bbox.append([x1, y1, box_w, box_h])
                bbox = patches.Rectangle((x1, y1),
                                         box_w,
                                         box_h,
                                         linewidth=2,
                                         edgecolor=color,
                                         facecolor='none')
            ax.add_patch(bbox)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0,  dpi=300)
    return all_bbox


if __name__ == "__main__":
    if opt.eval:
        evaluation()
    elif opt.test_hardcoded_boxes:
        test_hardcoded_boxes()
    else:
        train()
