#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.

from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os, random, math
from models import fc_model
from dataset import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------

def main(args):
    args.mode = "test"
    valid_loader, validset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=False, args=args)  #  test set
    
    # multi-class hinge loss
    label_crit = nn.CrossEntropyLoss()

    model = torch.load('model')
    model = model.to(device)
    model.eval()

    print("model loaded & starting testing ...\n\n")

    # Testing script
    valid_total_correct_preds = 0.0
    valid_total = 1e-10
    valid_loss = 0.0
    for step, (image_input, class_idxs) in enumerate(valid_loader):
        # move all data loaded from dataloader to gpu
        class_idxs = class_idxs.to(device)
        image_input = image_input.to(device)

        # feed-forward data in the model
        output = model(image_input)  # 32 * 150528 --> 32 * 11

        # compute losses
        state_loss = label_crit(output, class_idxs)  # --> 32 * 1

        # aggregate loss for logging
        valid_loss += state_loss.item()

        # accuracy computation
        _, pred_idx = torch.max(output, dim=1)
        valid_total_correct_preds += torch.sum(pred_idx == class_idxs).item()
        valid_total += output.size(0)

        # epoch accuracy & loss
        valid_accuracy = round(valid_total_correct_preds / valid_total, 4)
        valid_loss = round(valid_loss / valid_total, 4)
        # you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

    print('\rtestset: accuracy: {}, loss: {}\n'.format(valid_accuracy, valid_loss), end="")

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)

