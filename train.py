import argparse
import copy

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import vision_adapter
from config import exp_root, pretrained_weights
from data.augmentations import get_transform
from data.data_utils import AffMergedDataset
from data.get_datasets import get_class_splits, get_datasets
from modules import (
    AffinitySimilarityLoss,
    ContrastiveLearningViewGenerator,
    SupConLoss,
    info_nce_logits,
    label_aff,
    print_info,
    test_kmeans,
    unlabel_aff,
)
from project_utils.general_utils import AverageMeter, init_experiment


def pre_train(projection_head, model, labelled_dataset, device, args):
    labelled_loader = DataLoader(
        labelled_dataset, num_workers=args.num_workers, batch_size=32, shuffle=True
    )
    pre_optimizer = SGD(
        list(projection_head.parameters()) + list(model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    pre_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        pre_optimizer,
        T_max=args.pretrain_epoch,
        eta_min=args.lr * 1e-3,
    )

    for epoch in range(args.pretrain_epoch):
        loss_record = AverageMeter()

        projection_head.train()
        model.train()

        for _, batch in enumerate(tqdm(labelled_loader)):
            images, class_labels, _ = batch

            class_labels = class_labels.to(device)
            class_labels_numpy = class_labels.cpu().numpy()
            images = images.to(device)

            features = model(images)
            features = projection_head(features)
            features = torch.nn.functional.normalize(features, dim=-1)

            triplet_lst = []

            for i in range(0, len(features)):
                current_anchor = features[i]
                current_label = class_labels_numpy[i]

                if (
                    len(np.where(class_labels_numpy == current_label)[0]) >= 1
                    and len(np.where(class_labels_numpy != current_label)[0]) >= 1
                ):
                    positive_idx = np.random.choice(
                        np.where(class_labels_numpy == current_label)[0]
                    )
                    positive = features[positive_idx]
                    negative_idx = np.random.choice(
                        np.where(class_labels_numpy != current_label)[0]
                    )
                    negative = features[negative_idx]

                    merged = torch.stack([current_anchor, positive, negative])
                    triplet_lst.append(merged)

            if len(triplet_lst) != 0:
                triplet_lst = torch.stack(triplet_lst)

                anchor, positive, negative = (
                    triplet_lst[:, 0],
                    triplet_lst[:, 1],
                    triplet_lst[:, 2],
                )
                loss = F.triplet_margin_loss(
                    anchor, positive, negative, margin=1.0, p=2
                )

                loss_record.update(loss.item(), class_labels.size(0))
                pre_optimizer.zero_grad()
                loss.backward()
                pre_optimizer.step()
            else:
                pass

        print("Pretrain Epoch: {} Avg Loss: {:.4f} ".format(epoch, loss_record.avg))
        pre_lr_scheduler.step()

    torch.save(model.state_dict(), args.model_path[:-3] + "_pretrain.pt")
    print("model saved to {}.".format(args.model_path[:-3] + "_pretrain.pt"))
    torch.save(projection_head.state_dict(), args.model_path[:-3] + "_pretrain_head.pt")
    print(
        "projection head saved to {}.".format(
            args.model_path[:-3] + "_pretrain_head.pt"
        )
    )


def train(projection_head, model, train_dataset, unlabeled_train_dataset, args):
    optimizer = SGD(
        list(projection_head.parameters()) + list(model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    best_new = 0
    sup_con_crit = SupConLoss()
    cluster_criterion = AffinitySimilarityLoss(ncrops=2)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        set_loader = DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        knn_dict = unlabel_aff(set_loader, device, model)

        aff_train_dataset = AffMergedDataset(
            merged_dataset=train_dataset, aff_dict=knn_dict
        )

        label_len = len(train_dataset.labelled_dataset)
        unlabelled_len = len(train_dataset.unlabelled_dataset)
        sample_weights = [
            1 if i < label_len else label_len / unlabelled_len
            for i in range(len(train_dataset))
        ]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset)
        )

        train_loader = DataLoader(
            aff_train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
        )
        unlabeled_train_loader = DataLoader(
            unlabeled_train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )

        label_aff_dict = label_aff(train_loader, device, model, args)

        projection_head.train()
        model.train()

        for _, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, mask_lab, aff_images = batch
            mask_lab = mask_lab[:, 0]

            # Peudo labels
            pseudo_labels = copy.deepcopy(class_labels)
            pseudo_mask = copy.deepcopy(mask_lab)

            for i in range(0, len(uq_idxs)):
                batch_id = int(uq_idxs[i])
                if batch_id in label_aff_dict:
                    pseudo_labels[i] = label_aff_dict[batch_id]
                    pseudo_mask[i] = 1

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            pseudo_labels, pseudo_mask = (
                pseudo_labels.to(device),
                pseudo_mask.to(device).bool(),
            )

            images = torch.cat(images, dim=0).to(device)
            aff_images = torch.cat(aff_images, dim=0).to(device)
            images = torch.cat([images, aff_images], dim=0)

            features = model(images)
            features = projection_head(features)
            features = torch.nn.functional.normalize(features, dim=-1)

            features, knn_features = features.chunk(2)

            contrastive_logits, contrastive_labels = info_nce_logits(
                features, device, args
            )
            contrastive_loss = torch.nn.CrossEntropyLoss()(
                contrastive_logits, contrastive_labels
            )

            # Supervised contrastive loss
            f1, f2 = [f[pseudo_mask] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = pseudo_labels[pseudo_mask]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # Affinity similarity loss
            f3, f4 = [f[~pseudo_mask] for f in features.chunk(2)]
            un_feats = torch.cat([f3, f4], dim=0)
            f5, f6 = [f[~pseudo_mask] for f in knn_features.chunk(2)]
            aff_feats = torch.cat([f5, f6], dim=0)

            similarity_loss = cluster_criterion(un_feats, aff_feats)

            # Knowledge transfer loss
            triplet_lst = []

            for i in range(0, len(un_feats)):
                current_anchor = un_feats[i]
                positive = aff_feats[i]

                # random choose a feature in un_feats
                negative_idx = np.random.choice(range(len(un_feats)))
                while negative_idx == i:
                    negative_idx = np.random.choice(range(len(un_feats)))
                negative = un_feats[negative_idx]

                merged = torch.stack([current_anchor, positive, negative])
                triplet_lst.append(merged)

            triplet_lst = torch.stack(triplet_lst)

            anchor, positive, negative = (
                triplet_lst[:, 0],
                triplet_lst[:, 1],
                triplet_lst[:, 2],
            )
            triplet_loss = F.triplet_margin_loss(
                anchor, positive, negative, margin=1.0, p=2
            )

            # Total loss
            loss = (
                (1 - args.sup_con_weight) * contrastive_loss
                + args.sup_con_weight * sup_con_loss
                + similarity_loss
                + triplet_loss
            )

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(loss.item(), pseudo_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            "Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} ".format(
                epoch, loss_record.avg, train_acc_record.avg
            )
        )

        with torch.no_grad():
            all_acc, old_acc, new_acc = test_kmeans(model, unlabeled_train_loader, args)

        print(
            "Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(
                all_acc, old_acc, new_acc
            )
        )

        # Step schedule
        exp_lr_scheduler.step()

        if new_acc > best_new:
            print(f"Best NEW ACC: {new_acc:.4f}...")
            best_new = new_acc
            torch.save(model.state_dict(), args.model_path)
            print("model saved to {}.".format(args.model_path))
            torch.save(
                projection_head.state_dict(), args.model_path[:-3] + "_proj_head.pt"
            )
            print(
                "projection head saved to {}.".format(
                    args.model_path[:-3] + "_proj_head.pt"
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="scars",
    )
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--base_model", type=str, default="vit_dino")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sup_con_weight", type=float, default=0.5)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--known_class", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default="outs")
    parser.add_argument("--pretrain_epoch", type=int, default=120)

    # Init Experiment
    args = parser.parse_args()

    # better device handling, not limit to cuda 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_class_splits(args)

    init_experiment(args)

    # Model Init
    args.interpolation = 3
    args.crop_pct = 0.875
    pretrain_path = pretrained_weights

    model = vision_adapter.__dict__["vit_base"]()

    state_dict = torch.load(pretrain_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.to(device)

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 256

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Init Datasets
    train_transform, test_transform = get_transform(
        image_size=args.image_size, args=args
    )
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )

    train_dataset, unlabeled_train_dataset, labeled_train_dataset, datasets = (
        get_datasets(args.dataset_name, train_transform, test_transform, args)
    )

    projection_head = vision_adapter.__dict__["DINOHead"](
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers
    )
    projection_head.to(device)

    print_info(train_dataset, unlabeled_train_dataset, datasets, args)

    pre_train(projection_head, model, labeled_train_dataset, device, args)
    model.load_state_dict(torch.load(args.model_path[:-3] + "_pretrain.pt"))
    projection_head.load_state_dict(
        torch.load(args.model_path[:-3] + "_pretrain_head.pt")
    )

    train(projection_head, model, train_dataset, unlabeled_train_dataset, args)
