from torchvision.transforms import Normalize, Resize


class rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]
        return {"image": Resize(self.output_size)(image), "label": label}


class normalize:
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]
        return {"image": Normalize(mean=self.mean, std=self.std)(image), "label": label}
