import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

class AttentionConsistency(nn.Module):
    def __init__(self, lambd=6e-2, T=1.0):
        super().__init__()
        self.name = "AttentionConsistency"
        self.T = T
        self.lambd = lambd

    def CAM_neg(self, c):
        result = c.reshape(c.size(0), c.size(1), -1)
        result = -nn.functional.log_softmax(result / self.T, dim=2) / result.size(2)
        result = result.sum(2)

        return result

    def CAM_pos(self, c):
        result = c.reshape(c.size(0), c.size(1), -1)
        result = nn.functional.softmax(result / self.T, dim=2)

        return result

    def forward(self, c, ci_list, y, segmentation_masks=None):
        """
        CAM (batch_size, num_classes, feature_map.shpae[0], feature_map.shpae[1]) based loss

        Argumens:
            :param c: (Torch.tensor) clean image's CAM
            :param ci_list: (Torch.tensor) list of augmented image's CAMs
            :param y: (Torch.tensor) ground truth labels
            :param segmentation_masks: (numpy.array)
        :return:
        """
        c1 = c.clone()
        c1 = Variable(c1)
        c0 = self.CAM_neg(c)

        # Top-k negative classes
        c1 = c1.sum(2).sum(2)
        index = torch.zeros(c1.size())
        c1[range(c0.size(0)), y] = - float("Inf")
        topk_ind = torch.topk(c1, 3, dim=1)[1]
        index[torch.tensor(range(c1.size(0))).unsqueeze(1), topk_ind] = 1
        index = index > 0.5

        # Negative CAM loss
        neg_loss = c0[index].sum() / c0.size(0)
        for ci in ci_list:
            ci = self.CAM_neg(ci)
            neg_loss += ci[index].sum() / ci.size(0)
        neg_loss /= len(ci_list) + 1

        # Positive CAM loss
        index = torch.zeros(c1.size())
        true_ind = [[i] for i in y]
        index[torch.tensor(range(c1.size(0))).unsqueeze(1), true_ind] = 1
        index = index > 0.5
        p0 = self.CAM_pos(c)[index]
        pi_list = [self.CAM_pos(ci)[index] for ci in ci_list]

        # Middle ground for Jensen-Shannon divergence
        p_count = 1 + len(pi_list)
        if segmentation_masks is None:
            p_mixture = p0.detach().clone()
            for pi in pi_list:
                p_mixture += pi
            p_mixture = torch.clamp(p_mixture / p_count, 1e-7, 1).log()

        else:
            mask = np.interp(segmentation_masks, (segmentation_masks.min(), segmentation_masks.max()), (0, 1))
            p_mixture = torch.from_numpy(mask).cuda()
            p_mixture = p_mixture.reshape(p_mixture.size(0), -1)
            p_mixture = torch.nn.functional.normalize(p_mixture, dim=1)

        pos_loss = nn.functional.kl_div(p_mixture, p0, reduction='batchmean')
        for pi in pi_list:
            pos_loss += nn.functional.kl_div(p_mixture, pi, reduction='batchmean')
        pos_loss /= p_count

        loss = pos_loss + neg_loss
        return self.lambd * loss