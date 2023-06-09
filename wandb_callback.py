from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import wandb

    assert hasattr(wandb, '__version__')
except (ImportError, AssertionError):
    wandb = None

def class2dict(f):
    return dict(
        (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
    )


def on_pretrain_routine_start(trainer):    
    wandb.init(project="-".join(trainer.args.project.split("/")) or "YOLOv8", name=trainer.args.name, config=dict(
        trainer.args)) if not wandb.run else wandb.run


def on_fit_epoch_end(trainer):
    if trainer.metrics is not None:
        print("trainer.metrics", class2dict(trainer.metrics))
        wandb.log(class2dict(trainer.metrics), step=trainer.epoch + 1)
    if trainer.epoch == 0:
        model_info = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
            "model/speed(ms)": round(sum(trainer.validator.speed.values()), 3) if trainer.validator is not None else 0
        }
        wandb.log(model_info, step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    wandb.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wandb.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        wandb.log({f.stem: wandb.Image(str(f))
                       for f in trainer.save_dir.glob('train_batch*.jpg')},
                      step=trainer.epoch + 1)


def on_train_end(trainer):
    art = wandb.Artifact(type="model", name=f"run_{wandb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wandb.log_artifact(art)


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end} if wandb else {}
