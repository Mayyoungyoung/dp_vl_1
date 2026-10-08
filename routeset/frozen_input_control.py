"""Ordinary encoder-freezing ablation; no route or observation change."""
from .observed_training_audit import tensor_state_digest

PREFIXES = ('geometry.', 'head.feature_encoder.', 'head.state_encoder.')


def modules(model):
    return (model.geometry, model.head.feature_encoder, model.head.state_encoder)


def freeze(model):
    for module in modules(model):
        module.requires_grad_(False)
        module.eval()


def digest(model):
    return tensor_state_digest({k: v for k, v in model.state_dict().items() if k.startswith(PREFIXES)})


def audit(model, initial):
    final = digest(model)
    assert initial == final, 'Frozen input tensors changed'
    assert all(not p.requires_grad and p.grad is None for m in modules(model) for p in m.parameters())
    return dict(initial_sha256=initial, final_sha256=final, prefixes=list(PREFIXES),
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                grounding_loss_contract='Computed but constant with respect to remaining trainable parameters')
