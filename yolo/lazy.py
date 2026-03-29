import sys
from pathlib import Path

import torch
import hydra
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.deploy_utils import export_coreml
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    accelerator = getattr(cfg, "accelerator", "auto")
    use_mps = accelerator == "mps" or (accelerator == "auto" and torch.backends.mps.is_available())
    precision = "32-true" if use_mps else "16-mixed"

    trainer = Trainer(
        accelerator=accelerator,
        devices=cfg.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision=precision,
        callbacks=callbacks,
        sync_batchnorm=not use_mps,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model, ckpt_path=getattr(cfg, "resume", None))
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)
    if cfg.task.task == "export":
        export_coreml(
            model_cfg=cfg.model,
            weight_path=cfg.weight,
            class_num=cfg.dataset.class_num,
            image_size=tuple(cfg.task.image_size),
            output_path=cfg.task.output_path,
            class_list=cfg.dataset.class_list,
            pipeline=cfg.task.pipeline,
            iou_threshold=cfg.task.iou_threshold,
            confidence_threshold=cfg.task.confidence_threshold,
        )
        return


if __name__ == "__main__":
    main()
