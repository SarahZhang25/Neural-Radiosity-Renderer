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
    config = replace(config,
        training=replace(config.training,
            num_epochs=10,
            save_interval=9999,
            checkpoint_interval=9999,
            package_model=False,
            # data_dir="renderformer/datasets/processed_datasets/dataset_single_obj",
            # log_dir="tmp/test_run",
            run_name="LitePT_TEST",
        ),
    )

    # Save a temporary config file so Trainer can load it
    temp_config = "test_config_temp.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
        
    try:
        trainer = Trainer(temp_config)
        print(f"Starting {config.training.num_epochs}-epoch test run...")
        trainer.run()
        print("Test run completed successfully!")
    finally:
        # Cleanup
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    main()
