import argparse
import yaml
import torch
import os
# import shutil
from training.train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config_46M.yaml', help='Path to config file')
    args = parser.parse_args()

    # We patch the trainer directly to run for just 2 epochs and avoid writing checkpoints or logs.
    print(f"Loading config {args.config} for test run...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force settings for a fast, localized test
    config['training']['num_epochs'] = 2
    config['training']['save_interval'] = 9999
    config['training']['checkpoint_interval'] = 9999
    config['training']['package_model'] = False
    
    config['training']['data_dir'] = "data_generation/output_auto/datasets/attempt3_fixed_view_table"
    config['training']['log_dir'] = "tmp/test_run"
    config['training']['run_name'] = 'test_run'
    # Ensure num_register_tokens is tested
    config['decoder']['num_register_tokens'] = 2
    
    # Save a temporary config file
    temp_config = "test_config_temp.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
        
    try:
        trainer = Trainer(temp_config)
        print("Starting 2-epoch test run...")
        trainer.run()
        print("Test run completed successfully!")
    finally:
        # Cleanup
        if os.path.exists(temp_config):
            os.remove(temp_config)
        # if os.path.exists("tmp/test_run"):
        #     shutil.rmtree("test_run")

if __name__ == "__main__":
    main()
