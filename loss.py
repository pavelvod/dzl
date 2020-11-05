import torch
from torch import nn


class OnlineTripletLoss(nn.Module):
    """
    designed for tasks:
        - binary classification
        - multiclass classification
        - multilabel classification
        - undirected graphs

    Should be modified to use with directed graphs! (currently only upper triangle of the adjacency matrix is used)

    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def _pairwise_distances(self, embeddings, squared=False):
    
        """
        
        Taken from some github repo, not written by me
        
        Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        ## (Pavel) devide distances by sqrt or dimensions to normalize to 1
        distances = distances / torch.sqrt(torch.tensor(embeddings.shape[-1]).type(torch.float32))
        return distances


    def forward(self,
                embeddings: torch.tensor,
                labels: torch.tensor,
                weights: torch.tensor) -> torch.tensor:
        """

        :param embeddings:
                - shape [batch_size x embeddings_dim]
                - outputs of embedder module
        :param labels:
                - shape [batch_size x batch_size]
                - should be boolean 2d tensor, adjacency-matrix-like (may be passed as long tensor of 1's and 0's)
        :param weights:
                - shape [batch_size, 1]
                - weight of each sample
        :return:

        example:
            margin = 0.6

            1)
                positive link, dist = 0.4
                we want to reduce dist to 0, so loss=0.4

            2)
                negative link, dist = 0.4
                we want to push points in different directions to enlarge the distance to be more than 0.6
                so loss = max(0, 0.6 - 0.4) = max(0, 0.2) = 0.2

            3)
                negative_link, dist 0.7
                distance between 2 points greater than margin, loss = 0
                loss = max(0, 0.6 - 0.7) = max(0, -0.1) = 0

        """
        # create weight mask with the same dimensions as labels, each value is a max of instances
        w1 = weights.repeat(1, weights.shape[0])
        weight_mask = torch.max(w1, w1.t())

        # take upper triangle only, drop first diagonal of self correlations
        mask_anchor_positive = torch.triu(labels, diagonal=1).type(torch.bool)
        mask_anchor_negative = torch.triu(1 - labels).type(torch.bool)
        # calculate pairwise euclidean distances
        pairwise_dist = self._pairwise_distances(embeddings)

        positive_dist = pairwise_dist * mask_anchor_positive
        negative_dist = torch.relu((self.margin - pairwise_dist)) * mask_anchor_negative

        # find location of greatest positive and negative loss in each row and create a masks
        positive_argmax_mask = torch.nn.functional.one_hot(positive_dist.argmax(1),
                                                           num_classes=embeddings.shape[0]).type(torch.bool)
        negative_argmax_mask = torch.nn.functional.one_hot(negative_dist.argmax(1),
                                                           num_classes=embeddings.shape[0]).type(torch.bool)

        # using mask, take losses and weights
        positive_weights = weight_mask[positive_argmax_mask]
        negative_weights = weight_mask[negative_argmax_mask]
        positive_loss = positive_dist[positive_argmax_mask]
        negative_loss = negative_dist[negative_argmax_mask]

        # calculate weighted loss
        weighted_loss = (positive_loss * positive_weights).sum() + (negative_loss * negative_weights).sum()
        total_weights = (positive_weights + negative_weights).sum()
        loss = weighted_loss / total_weights
        return loss
