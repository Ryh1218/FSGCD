from copy import deepcopy

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.cub import get_cub_datasets
from data.data_utils import MergedDataset
from data.herbarium_19 import get_herbarium_datasets
from data.imagenet import get_imagenet_100_datasets
from data.stanford_cars import get_scars_datasets

get_dataset_funcs = {
    "cifar10": get_cifar_10_datasets,
    "cifar100": get_cifar_100_datasets,
    "imagenet_100": get_imagenet_100_datasets,
    "herbarium_19": get_herbarium_datasets,
    "cub": get_cub_datasets,
    "scars": get_scars_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=args.train_classes,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
    )

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(
        labelled_dataset=deepcopy(datasets["train_labelled"]),
        unlabelled_dataset=deepcopy(datasets["train_unlabelled"]),
    )

    labeled_train_dataset = deepcopy(datasets["train_labelled"])
    labeled_train_dataset.transform = test_transform
    unlabeled_train_dataset = deepcopy(datasets["train_unlabelled"])
    unlabeled_train_dataset.transform = test_transform

    return train_dataset, unlabeled_train_dataset, labeled_train_dataset, datasets


def get_class_splits(args):
    if args.dataset_name == "cifar10":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 10)

    elif args.dataset_name == "cifar100":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 100)

    elif args.dataset_name == "herbarium_19":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 683)

    elif args.dataset_name == "imagenet_100":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 100)

    elif args.dataset_name == "scars":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 196)

    elif args.dataset_name == "cub":
        args.train_classes = range(args.known_class)
        args.unlabeled_classes = range(args.known_class, 200)

    else:
        raise NotImplementedError

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    return args
