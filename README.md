# Neural-Radiosity-Renderer
`conda create -n neural_radiosity_renderer python=3.11`
-> new: use environment.yml and requirements.txt

`conda activate neural_radiosity_renderer`
`export PYTHONPATH=.`

### Data Generation
Simple shapes only, mitsuba rendering, example usage:

`python data_generation/generate_pure_mitsuba.py --shapes sphere --num_rotations 3 --scale_min 0.3 --scale_max 0.6 --pos_variation 0.15 --light_intensity 10 --spp 1024`

Training script:
`python training/train.py`
Training config: `training/train_config.yml`\

Launch tensorboard with `tensorboard --logdir=training/logs`