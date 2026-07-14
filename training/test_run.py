import argparse
import yaml
import torch
import os
from dataclasses import replace
from model.config import NeuralRadiosityConfig, TrainingConfig, SceneTransformerConfig
from training.train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config_46M.yaml', help='Path to config file')
    args = parser.parse_args()

    # We patch the trainer directly to run for just 2 epochs and avoid writing checkpoints or logs.
    print(f"Loading config {args.config} for test run...")
    config = NeuralRadiosityConfig.from_yaml(args.config)
    
    # Force settings for a fast, localized test
    log_dir = "tmp/test_run"
    config = replace(config,
        training=replace(config.training,
            num_epochs=10,
            save_interval=9999,
            checkpoint_interval=9999,
            package_model=False,
            data_dir="renderformer/datasets/processed_datasets/dataset_single_obj",
            log_dir=log_dir,
            run_name="test_multigpu_training",
        ),
    )

    # Save a temporary config file so Trainer can load it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    temp_config = f"{log_dir}/test_config_temp_rank{local_rank}.yaml"
    os.makedirs(log_dir, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
        
    try:
        trainer = Trainer(temp_config)
        if trainer.is_main_process:
            print(f"Starting {config.training.num_epochs}-epoch test run...")
        trainer.run()
        if trainer.is_main_process:
            print("Test run completed successfully!")
    finally:
        # Cleanup
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    main()
