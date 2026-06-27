import torch
import torch.package
import yaml
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from model.global_illumination_model import GlobalIlluminationModel

try:
    with open("training/train_config_46M.yaml", 'r') as f:
        config = yaml.safe_load(f)
    model = GlobalIlluminationModel(config)
    pkg_path = "test_pkg2.pt"
    
    print("Exporting...")
    with torch.package.PackageExporter(pkg_path) as exp:
        exp.intern("model.**")
        exp.intern("utils.**")
        exp.intern("pos_encodings.**")
        exp.extern("**")
        exp.save_pickle("model", "model.pkl", model)
    print("Exported successfully to", pkg_path)
    
    print("Importing...")
    imp = torch.package.PackageImporter(pkg_path)
    model_loaded = imp.load_pickle("model", "model.pkl")
    print("Imported successfully!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
