#
#
#   Utils
#
#

import torch
import markdown
import numpy as np

from bs4 import BeautifulSoup


def md_to_text(md):
    html = markdown.markdown(str(md))
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def calculate_pos_weights(class_counts, total):
    pos_weights = np.ones_like(class_counts)
    neg_counts = [total - pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return torch.as_tensor(pos_weights, dtype=torch.float)
