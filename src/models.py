import torch.nn as nn
import torchvision.models as models


MODEL_REGISTRY = {
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'efficientnet_v2_l': models.efficientnet_v2_l,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32
}


def replace_classifier(model_name, model, num_classes):
    if model_name.startswith('efficientnet') or model_name.startswith('mobilenet'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name.startswith('resnet'):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)

    elif model_name.startswith('vit'):
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes, bias=True)

    else:
        raise ValueError(f"Unsupported model family: {model_name}")

    return model


def freeze_layers(config, model):
    model_name = config['model']['name']
    freeze_backbone = config['model']['freeze_backbone']
    freeze_classifier = config['model']['freeze_classifier']

    if model_name.startswith(('efficientnet', 'mobilenet')):
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False

        if freeze_classifier:
            for p in model.classifier.parameters():
                p.requires_grad = False

    elif model_name.startswith('resnet'):
        if freeze_backbone:
            for name, module in model.named_children():
                if name != 'fc':
                    for p in module.parameters():
                        p.requires_grad = False

        if freeze_classifier:
            for p in model.fc.parameters():
                p.requires_grad = False

    elif model_name.startswith('vit'):
        if freeze_backbone:
            for p in model.encoder.parameters():
                p.requires_grad = False

        if freeze_classifier:
            for p in model.heads.parameters():
                p.requires_grad = False

    else:
        raise ValueError(f"Unsupported model family: {model_name}")

    return model


def build_model(config):
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    num_classes = config['model']['num_classes']

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(weights='DEFAULT' if pretrained else None)

    model = replace_classifier(model_name, model, num_classes)
    model = freeze_layers(config, model)

    return model