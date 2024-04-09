import torch


def triplet_loss(threshold=0.2):
    def _triplet_loss(feature, batch_size):
        # feature = 3batch_size * embedding_size
        anchor, pos, neg = (
            feature[: int(batch_size)],
            feature[int(batch_size) : int(2 * batch_size)],
            feature[int(2 * batch_size) :],
        )

        pos_dist = torch.sqrt(
            torch.sum(torch.pow(anchor - pos, 2), axis=-1)
        )  # 在embedding维度求欧式距离
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - neg, 2), axis=-1))

        keep = neg_dist - pos_dist < threshold
        pos_dist = pos_dist[keep]
        neg_dist = neg_dist[keep]

        loss_ = pos_dist - neg_dist + threshold
        loss = torch.sum(loss_) / torch.max(
            torch.tensor(1), torch.tensor(pos_dist.numel())
        )
        return loss

    return _triplet_loss
