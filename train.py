import time
import datetime
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model_RESNET50_kitti import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import utils
from tqdm import tqdm
from pprint import PrettyPrinter
import numpy as np
import kitti_data_helper
pp = PrettyPrinter()

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(kitti_data_helper.get_True_label())  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
train_batch_size = 6  # batch size
valid_batch_size = 6
iterations = 60000  # number of iterations to train
valid_epoch_term = 20
workers = 6  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [40000, 50000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_loader, valid_loader, len_train_data_set = kitti_data_helper.make_loader(train_batch_size, valid_batch_size, workers)

    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len_train_data_set // 32)
    decay_lr_at = [it // (len_train_data_set // 32) for it in decay_lr_at]

    # Epochs
    one_epoch_processing_times = []
    start = 0
    min_valid_loss = 0
    print("총 에포치: ", epochs)
    for epoch in range(start_epoch, epochs):
        if start != 0:
            sec = time.time() - start
            sec = sec * (epochs - epoch)
            times = str(datetime.timedelta(seconds=sec)).split(".")
            times = times[0]
            print("예상 학습 완료시간: ", times)
        start = time.time()

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        if epoch % valid_epoch_term == valid_epoch_term - 1 or epoch == epochs-1:
            # valid(valid_loader=valid_loader, model=model, num_test_count=15)
            min_valid_loss = evaluate_valid_loss_for_save(valid_loader=valid_loader, model=model, current_loss=min_valid_loss,
                                                          criterion=criterion, optimizer=optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss


    # Batches
    for i, (images, boxes, labels, difficult) in enumerate(tqdm(train_loader)):

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        # print(torch.mean(images))
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    # Print status
    print('Epoch: [{0}][{1}/{2}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def valid(valid_loader, model, num_test_count):
    model.eval()
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(valid_loader, desc='Validation')):
            images = images.to(device)  # (N, 3, 300, 300)
            labels = [l.to(device) for l in labels]
            boxes = [b.to(device) for b in boxes]
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            if i > num_test_count - 1:
                break

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, kitti_data_helper.get_True_label())

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

def evaluate_valid_loss_for_save(valid_loader, model, current_loss, criterion, optimizer):
    model.eval()
    with torch.no_grad():
        losses = []
        for i, (images, boxes, labels, difficult) in enumerate(tqdm(valid_loader, desc='Validation_for_save')):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            losses.append(loss.item())
        print("현재 상황에서 valid_loss는?", np.mean(losses))
        print("그 전까지 최저 loss는?", current_loss)

        if current_loss == 0:
            current_loss = np.mean(losses)
        elif current_loss > np.mean(losses):
            current_loss = np.mean(losses)
            print("저장!")
            save_checkpoint(epoch, model, optimizer)

    return current_loss

if __name__ == '__main__':
    main()
