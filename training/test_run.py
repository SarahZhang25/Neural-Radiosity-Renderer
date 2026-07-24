import argparse
import yaml
import torch
import os
from dataclasses import replace
from model.config import NeuralRadiosityConfig, TrainingConfig, SceneTransformerConfig
from training.train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config_46M_pointnet_h5.yaml', help='Path to config file')
    args = parser.parse_args()

    # We patch the trainer directly to run for just few epochs and avoid writing checkpoints or logs.
    print(f"Loading config {args.config} for test run...")
    config = NeuralRadiosityConfig.from_yaml(args.config)
    
    # Force settings for a fast, localized test
    log_dir = "tmp/test_run"
    config = replace(config,
        training=replace(config.training,
            global_batch_size=32, # 32 * 3 gpu
            num_steps=5000,
            warmup_steps=500,
            save_interval_steps=500,
            log_interval_steps=100,
            package_model=False,
            data_dir="tmp/dataset_test/nmr_dataset_chunk_0000.h5",
            log_dir="tmp/test_run",
            run_name="TEST_104M_pointnet_tokendim768_patch8_updated_spherical_cross_attn_rope",
            learning_rate=config.training.learning_rate * 2.5, # scale up as batch size inc. default is 1e-4
            # image_res=128,
            # lpips_loss_weighting=0.07
        ),
        # decoder=replace(config.decoder,
        #     use_obj_obj_attention_bias=False,
        #     obj_obj_bias_hidden_dim=128
        # )
    )

    # Save a temporary config file so Trainer can load it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    temp_config = f"{log_dir}/test_config_temp.yaml"
    os.makedirs(log_dir, exist_ok=True)
    if local_rank == 0:
        with open(temp_config, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
    else:
        import time
        while not os.path.exists(temp_config):
            time.sleep(0.1)
        
    try:
        trainer = Trainer(config_path=temp_config, original_config_path=args.config)
        if trainer.is_main_process:
            print(f"Starting test run...")
        trainer.run()
        if trainer.is_main_process:
            print("Test run completed successfully!")
    finally:
        # Cleanup
        if local_rank == 0 and os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    main()
