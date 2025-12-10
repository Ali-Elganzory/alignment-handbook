import logging
import os
import shutil

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class SaveModelWeightsCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        logger.info(f"Copying lightweight model weights to {args.output_dir}")

        # The trainer just saved a checkpoint to this folder
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )

        # Define where you want to keep your permanent lightweight artifacts
        artifact_folder = os.path.join(
            args.output_dir, "artifacts", f"{state.global_step}"
        )
        os.makedirs(artifact_folder, exist_ok=True)

        # Copy ONLY the model weights and config (skip optimizer.pt / scheduler.pt)
        # Usually these are 'model.safetensors' (or pytorch_model.bin) and 'config.json'
        for filename in os.listdir(checkpoint_folder):
            if (
                filename.endswith(".safetensors")
                or filename.endswith(".bin")
                or filename.endswith(".json")
                or filename.endswith(".model")
                or filename.endswith(".bin")
                or filename.endswith(".jinja")
                or filename.endswith("latest")
            ):
                src = os.path.join(checkpoint_folder, filename)
                dst = os.path.join(artifact_folder, filename)
                shutil.copy2(src, dst)

        logger.info(f"Saved lightweight model artifact to {artifact_folder}")
