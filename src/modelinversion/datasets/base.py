from typing import Any, Callable, Dict, List, Tuple, Optional
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import os


class LabelDatasetFolder(DatasetFolder):

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Tuple[str, ...] | None = None,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(
            root, loader, extensions, transform, target_transform, is_valid_file
        )

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            (
                entry.name
                for entry in os.scandir(directory)
                if entry.is_dir() and entry.name.isalnum()
            ),
            key=lambda x: int(x),
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class LabelImageFolder(LabelDatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/0/xxx.png
        root/0/xxy.png

        root/1/123.png
        root/1/nsdf3.png

    This class inherits from :class:`LabelDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
