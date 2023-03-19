from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import kitti_data_helper
from PIL import Image,ImageDraw
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
import numpy as np
from model_RESNET50_kitti import SSD300, MultiBoxLoss

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 12
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

# Switch to eval mode
model.eval()

# Load test data
train_loader, valid_loader, len_train_data_set = kitti_data_helper.make_loader(batch_size, batch_size, workers)

def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
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
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
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

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, kitti_data_helper.get_True_label())

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

def show_detect_result(valid_loader, model):
    model.eval()
    # Lists to store detected and true boxes, labels, scores

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(valid_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            predicted_locs, predicted_scores = model(images)
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            transform_size = [384, 1280]
            img = images[0]
            bbox = det_boxes_batch[0]
            label = det_labels_batch[0]
            score = det_scores_batch[0]
            score_thresh_hold = 0.4
            detect_object_count = len(label)
            print(label)
            print(bbox)
            print(score)

            img += 0.5
            img = FT.to_pil_image(img)
            draw = ImageDraw.Draw(img)

            for i in range(detect_object_count):
                if score[i] < score_thresh_hold:
                    continue
                draw.rectangle(((bbox[i][0] * transform_size[1], bbox[i][1] * transform_size[0]),
                                (bbox[i][2] * transform_size[1], bbox[i][3] * transform_size[0])), outline=255, width=1)
                draw.text((bbox[i][0] * transform_size[1], bbox[i][1] * transform_size[0]), kitti_data_helper.get_True_label()[label[i]],
                          fill=(255, 255, 255, 0))
            plt.imshow(np.array(img))
            plt.figure(figsize=(10, 10))
            img.show()

def evaluate_valid_loss(valid_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        losses = []
        for i, (images, boxes, labels, difficult) in enumerate(tqdm(valid_loader, desc='Validation_for_loss')):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            losses.append(loss.item())
        print("현재 상황에서 valid_loss는?", np.mean(losses))


if __name__ == '__main__':
    # evaluate(valid_loader, model)
    #show_detect_result(valid_loader, model)
    evaluate_valid_loss(valid_loader=valid_loader, model=model,criterion=criterion)