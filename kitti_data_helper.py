# background의 label은 0!
import torchvision
import torch
import torchvision.transforms.functional as FT
from PIL import Image,ImageDraw
from torch.utils.data.sampler import SubsetRandomSampler
import utils
import matplotlib.pyplot as plt
import numpy as np

transform_size = [384, 1280]
# transform_size = [300, 300]

def get_True_label():
    True_label = ['Background', 'Car', 'Van', 'Truck', 'Cyclist', 'Pedestrian', 'Person_sitting', 'Tram']
    return True_label

def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)
    # new_image.show()

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def basic_transform(image, boxes, labels, difficulties):
    """
    Apply the transformations above.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(transform_size[0], transform_size[1]))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)


    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    # new_image = FT.normalize(new_image, mean=mean, std=std)
    new_image = new_image - 0.5

    return new_image, new_boxes, new_labels, new_difficulties

def val_collate_fn(batch):
    # 리턴용 리스트
    images = list()
    boxes = list()
    labels = list()
    difficulties = list()

    for b in batch:
        # print(b)
        tmp_img = b[0]
        tmp_box = []
        tmp_label = []
        tmp_difficult = []
        for target_elem in b[1]:
            if target_elem['type'] == 'DontCare' or target_elem['type'] == 'Misc':
                continue
            else:
                digit_label = get_True_label().index(target_elem['type'])
            tmp_label.append(digit_label)
            tmp_box.append(target_elem['bbox'])
            if target_elem['occluded'] != 2:
                tmp_difficult.append(0)
            else:
                tmp_difficult.append(1)

        # print("여기까지 txt 파일 가공")
        tmp_box = torch.tensor(tmp_box)
        tmp_label = torch.tensor(tmp_label)
        tmp_difficult = torch.tensor(tmp_difficult)
        tmp_difficult = tmp_difficult.type(torch.uint8)
        new_img, new_box, new_label, new_difficult = basic_transform(tmp_img, tmp_box, tmp_label, tmp_difficult)
        # print("전처리 적용")
        images.append(new_img)
        boxes.append(new_box)
        labels.append(new_label)
        difficulties.append(new_difficult)

    images = torch.stack(images, dim=0)
    return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors eac

def train_collate_fn(batch):
    # 리턴용 리스트
    images = list()
    boxes = list()
    labels = list()
    difficulties = list()

    for b in batch:
        # print(b)
        tmp_img = b[0]
        tmp_box = []
        tmp_label = []
        tmp_difficult = []
        for target_elem in b[1]:
            if target_elem['type'] == 'DontCare' or target_elem['type'] == 'Misc':
                continue
            else:
                digit_label = get_True_label().index(target_elem['type'])
            tmp_label.append(digit_label)
            tmp_box.append(target_elem['bbox'])
            if target_elem['occluded'] != 2:
                tmp_difficult.append(0)
            else:
                tmp_difficult.append(1)
        # print("여기까지 txt 파일 가공")
        new_img, new_box, new_label, new_difficult = utils.transform(tmp_img, torch.tensor(tmp_box), torch.tensor(tmp_label),
                                                                     torch.tensor(tmp_difficult).type(torch.uint8),transform_size)
        # print("전처리 적용")
        images.append(new_img)
        boxes.append(new_box)
        labels.append(new_label)
        difficulties.append(new_difficult)

    images = torch.stack(images, dim=0)
    return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors eac


def make_loader(train_batch_size, val_batch_size, workers):
    with open("./kitti_data/train.txt") as f:
        train_data_indices = f.readlines()
    with open("./kitti_data/val.txt") as f:
        val_data_indices = f.readlines()
    print(len(train_data_indices))
    print(len(val_data_indices))
    train_data_indices = list(map(int, train_data_indices)) # str -> int 변환하기
    val_data_indices = list(map(int, val_data_indices))

    train_dataset = torchvision.datasets.Kitti(root='./kitti_data', train=True, download=False, transform=None)
    train_sampler = SubsetRandomSampler(train_data_indices)
    val_sampler = SubsetRandomSampler(val_data_indices)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=train_collate_fn,
                                                   num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=val_batch_size, collate_fn=val_collate_fn,
                                                   num_workers=2, pin_memory=True, sampler=val_sampler)

    return train_dataloader, val_dataloader, len(train_dataset)



if __name__ == '__main__':
    train_dataloader,val_dataloader, _ = make_loader(1, 1, 1)
    for i, data in enumerate(train_dataloader):
        img = data[0][0]
        bbox = data[1][0][0]
        print(bbox)
        img += 0.5
        img = FT.to_pil_image(data[0][0])
        draw = ImageDraw.Draw(img)
        draw.rectangle(((bbox[0] * transform_size[1], bbox[1] * transform_size[0]), (bbox[2] * transform_size[1], bbox[3] * transform_size[0])), outline=255, width=1)
        draw.text((bbox[0]*transform_size[1], bbox[1]*transform_size[0]), True_label[data[2][0][0]], fill=(255, 255, 255, 0))
        plt.imshow(np.array(img))
        plt.figure(figsize=(10, 10))
        img.show()
