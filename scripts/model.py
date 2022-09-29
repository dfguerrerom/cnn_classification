from importlib import reload

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, lr_scheduler
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name, output_size, fixed_feature=False):
    """get corresponding pretrained model. Either fixed_feature or fine_tune

    output_size: is the number of the output classes, it has to match


    """

    if model_name in ["dpn92", "dpn131"]:
        model = torch.hub.load(
            "rwightman/pytorch-dpn-pretrained", model_name, pretrained=True
        )

    elif model_name in ["resnet34"]:
        model = models.resnet34(pretrained=True)

    # when the model is dpn, it will return a tuple
    model = next(iter(model)) if isinstance(model, (tuple, list)) else model

    # Get the last layer name
    fc_layer_name = list(model.named_modules())[-1][0]
    fc_layer = getattr(model, fc_layer_name)

    # Get the input number of features
    # in_features when using resnet, in_cahnels when using dpn
    num_ftrs = getattr(fc_layer, "in_features", None) or getattr(
        fc_layer, "in_channels"
    )

    if fixed_feature:
        # fixed feature extractor
        _ = [setattr(param, "require_grad", False) for param in model.parameters()]

    if "resnet" in model_name:
        last_layer = nn.Linear(num_ftrs, output_size)
    else:
        last_layer = nn.Conv2d(num_ftrs, output_size, kernel_size=1, bias=True)

    # Let's add the last layer to the model
    setattr(model, fc_layer_name, last_layer)

    return model


def get_optimizer(name, model, **kwargs):
    """Returns the optimizer based on the input name and specific args"""

    if name == "Adam":

        lr = kwargs.get("lr")
        weight_decay = kwargs.get("weight_decay")
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif name == "SDG":

        lr = kwargs.get("lr")
        momentum = kwargs.get("momentum")
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_scheduler(name, optimizer, **kwargs):
    """Returns the scheduler based on the input name and specific args"""

    if name == "LambdaLR":

        lr_lambda = kwargs.get("lr_lambda")
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif name == "StepLR":

        step_size = kwargs.get("step_size")
        gamma = kwargs.get("gamma")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


loss_fns = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "MSELoss": nn.MSELoss(),
}


def get_settings(test_name):
    """Return the variables to be used in the training step based on some
    predefined model settings"""

    import config.settings as ss

    ss = reload(ss)
    settings = ss.settings

    setting = settings[test_name]
    model_name = setting.model.name

    model = get_model(
        model_name=model_name,
        output_size=setting.model.out_features,
        fixed_feature=setting.model.fixed_feature,
    )

    model.to(device)

    optimizer = get_optimizer(
        name=setting.optimizer.name,
        model=model,
        lr=setting.optimizer.lr,
        weight_decay=getattr(setting.optimizer, "weight_decay", None),
        momentum=getattr(setting.optimizer, "momentum", None),
    )

    scheduler = get_scheduler(
        setting.scheduler.name,
        optimizer,
        step_size=setting.scheduler.get("step_size"),
        gamma=setting.scheduler.get("gamma"),
        lr_lambda=setting.scheduler.get("lr_lambda"),
    )

    loss_fn = loss_fns[setting.loss_fn.name]
    variable = setting.dataset.variable
    batch_size = setting.batch_size
    rescale_factor = setting.rescale_factor

    metadata = setting.to_json(default=lambda o: "<not serializable>")

    return (
        model,
        model_name,
        optimizer,
        loss_fn,
        scheduler,
        variable,
        batch_size,
        rescale_factor,
        metadata,
    )


def get_prediction_settings(test_name):
    """
    Will return the settings variables that corresponds to test_name and return:
    stride, width, height, batch_size
    """

    import config.settings as ss

    ss = reload(ss)
    settings = ss.settings

    setting = settings[test_name]

    stride = setting.prediction.stride
    width = setting.prediction.width
    height = setting.prediction.height
    batch_size = setting.prediction.batch_size

    metadata = setting.prediction.to_json(default=lambda o: "<not serializable>")

    return stride, width, height, batch_size, metadata
