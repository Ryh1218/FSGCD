import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from project_utils.cluster_and_log_utils import logAccs


def print_info(train_dataset, unlabeled_train_examples_test, datasets, args):
    label_len = len(train_dataset.labelled_dataset)
    unlabeled_len = len(train_dataset.unlabelled_dataset)

    if args.dataset_name == "cifar10":
        print(
            "Known classes: {}".format(len(set(train_dataset.labelled_dataset.targets)))
        )
        print(
            "Total classes in unlabeled set: {}".format(
                len(set(unlabeled_train_examples_test.targets))
            )
        )
        print("Labeled sample count: {}".format(label_len))
        print("Unlabeled sample count: {}".format(unlabeled_len))
    elif args.dataset_name == "cifar100":
        print(
            "Known classes: {}".format(len(set(train_dataset.labelled_dataset.targets)))
        )
        print(
            "Total classes in unlabeled set: {}".format(
                len(set(unlabeled_train_examples_test.targets))
            )
        )
        print("Labeled sample count: {}".format(label_len))
        print("Unlabeled sample count: {}".format(unlabeled_len))
    elif args.dataset_name == "cub":
        print(
            "Known classes: {}".format(
                len(set(train_dataset.labelled_dataset.data["target"].values))
            )
        )
        print(
            "Total classes in unlabeled set: {}".format(
                len(set(unlabeled_train_examples_test.data["target"].values))
            )
        )
        print("Labeled sample count: {}".format(label_len))
        print("Unlabeled sample count: {}".format(unlabeled_len))
    elif args.dataset_name == "herbarium_19":
        print(f"Num Labelled Classes: {len(set(datasets['train_labelled'].targets))}")
        print(
            f"Num Unabelled Classes: {len(set(datasets['train_unlabelled'].targets))}"
        )
        print(f"Len labelled set: {len(datasets['train_labelled'])}")
        print(f"Len unlabelled set: {len(datasets['train_unlabelled'])}")
    elif args.dataset_name == "scars":
        print(f"Num Labelled Classes: {len(set(datasets['train_labelled'].target))}")
        print(f"Num Unabelled Classes: {len(set(datasets['train_unlabelled'].target))}")
        print(f"Len labelled set: {len(datasets['train_labelled'])}")
        print(f"Len unlabelled set: {len(datasets['train_unlabelled'])}")
    elif args.dataset_name == "imagenet_100":
        print(f"Num Labelled Classes: {len(set(datasets['train_labelled'].targets))}")
        print(
            f"Num Unabelled Classes: {len(set(datasets['train_unlabelled'].targets))}"
        )
        print(f"Len labelled set: {len(datasets['train_labelled'])}")
        print(f"Len unlabelled set: {len(datasets['train_unlabelled'])}")


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class AffinitySimilarityLoss(torch.nn.Module):
    def __init__(self, ncrops=2):
        super().__init__()
        self.ncrops = ncrops

    def forward(self, anchor, aff_nn):
        anchor = anchor.detach().chunk(2)
        aff_nn = aff_nn.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(aff_nn):
            for v in range(len(anchor)):
                if v == iq:
                    # Skip the same view
                    continue
                cos_sim = torch.nn.functional.cosine_similarity(anchor[v], q, dim=1)
                cos_loss = 1 - cos_sim
                total_loss += cos_loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return total_loss


def unlabel_aff(train_loader, device, model):
    model.eval()

    all_feats = []
    all_targets = np.array([])
    all_uq_ids = np.array([])

    for _, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs, _ = batch

        # Discard the second view
        images = images[0]

        class_labels = class_labels.to(device)
        images = images.cuda()

        features = model(images)
        features = torch.nn.functional.normalize(features, dim=-1)

        all_feats.append(features.detach().cpu().numpy())
        all_targets = np.append(all_targets, class_labels.cpu().numpy())
        all_uq_ids = np.append(all_uq_ids, uq_idxs.cpu().numpy())

    all_feats = np.concatenate(all_feats)

    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(all_feats)
    _, indices = knn.kneighbors(all_feats)

    un_dict = {}

    for i in range(len(indices)):
        un_id = int(all_uq_ids[i])
        nn_ids = all_uq_ids[indices[i]]
        for nn_idxs in range(len(nn_ids)):
            nn_id = int(nn_ids[nn_idxs])
            if int(nn_id) != un_id and un_id not in un_dict:
                un_dict[un_id] = int(nn_id)

    return un_dict


def label_aff(train_loader, device, model, args):
    model.eval()

    all_feats = []
    all_targets = np.array([])
    all_uq_ids = np.array([])
    mask_lab = np.array([])
    mask_cls = np.array([])

    for _, batch in enumerate(tqdm(train_loader)):
        img_lst, labels, uq_ids, mask_labs, _ = batch
        mask_labs = mask_labs[:, 0]

        # Discard the second view
        images = img_lst[0].to(device)

        features = model(images)
        feats = torch.nn.functional.normalize(features, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        all_targets = np.append(all_targets, labels.cpu().numpy())
        all_uq_ids = np.append(all_uq_ids, uq_ids.cpu().numpy())

        mask_cls = np.append(
            mask_cls,
            np.array(
                [
                    True if x.item() in range(len(args.train_classes)) else False
                    for x in labels
                ]
            ),
        )
        mask_lab = np.append(mask_lab, mask_labs.cpu().bool().numpy())

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    labeled_targets = all_targets[mask_lab]
    labeled_feats = all_feats[mask_lab]
    unlabeled_feats = all_feats[~mask_lab]
    unlabeled_ids = all_uq_ids[~mask_lab]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(unlabeled_feats)
    _, indices = knn.kneighbors(labeled_feats)

    label_aff_dict = {}

    for i in range(len(indices)):
        cls = labeled_targets[i]
        nn_idxs = indices[i]
        for nn_idx in nn_idxs:
            nn_id = unlabeled_ids[nn_idx]
            ful_target = int(cls)
            if nn_id not in label_aff_dict:
                label_aff_dict[nn_id] = ful_target

    return label_aff_dict


def info_nce_logits(features, device, args):
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def test_kmeans(model, test_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    for _, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(
            mask,
            np.array(
                [
                    True if x.item() in range(len(args.train_classes)) else False
                    for x in label
                ]
            ),
        )

    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(
        n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0
    ).fit(all_feats)
    preds = kmeans.labels_

    all_acc, old_acc, new_acc = logAccs(y_true=targets, y_pred=preds, mask=mask)

    return all_acc, old_acc, new_acc
