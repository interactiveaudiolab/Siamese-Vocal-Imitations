import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utilities


def mrr_per_epoch(train_mrrs, val_mrrs, title="MRR vs. Epoch", xlabel='epoch'):
    plt.plot(train_mrrs, color='blue', label='training')
    plt.plot(val_mrrs, color='orange', label='validation')
    plt.legend()

    plt.ylabel('MRR')
    plt.yticks(np.arange(0, 1.1, .1))
    plt.ylim(0, 1)

    plt.xlabel(xlabel)

    plt.suptitle(title)
    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    filename = title_to_filename(title)
    plt.savefig(filename)
    plt.close()


def loss_per_epoch(train_loss, val_loss, title="Loss vs. Epoch"):
    plt.plot(train_loss, color='blue', label='training')
    plt.plot(val_loss, color='orange', label='validation')
    plt.legend()

    plt.ylabel('loss')
    plt.yticks(np.arange(0, 2.1, .1))
    plt.ylim(0, 2)

    plt.xlabel('epoch')
    plt.suptitle(title)
    plt.title("Trial #{0}".format(utilities.get_trial_number()))
    filename = title_to_filename(title)
    plt.savefig(filename)
    plt.close()


def title_to_filename(title):
    file = title
    file = file.replace(' ', '_')
    file = file.replace('.', '')
    file += '.png'
    file = file.lower()
    return os.path.join('./output', str(utilities.get_trial_number()), file)
