# setup
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
# Create a virtual environment with uv
uv venv
source .venv/bin/activate
# so any changes you make to the code will be immediately reflected without needing to reinstall.
uv pip install -e .


# download sam checkpoints (ViT-H, ViT-L, ViT-B)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# or
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# or
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# configure dataset structure as follows:

# ```
# ├──dataset_path
# │   ├── img_dir
# │   │   ├── train
# │   │   │   ├── xxx{img_suffix}
# │   │   │   ├── yyy{img_suffix}
# │   │   │   ├── zzz{img_suffix}
# │   │   │   ├── ....
# │   │   ├── val
# │   │   │   ├── xxx{img_suffix}
# │   │   │   ├── yyy{img_suffix}
# │   │   │   ├── zzz{img_suffix} 
# │   │   ├── test
# │   │   │   ├── xxx{img_suffix}
# │   │   │   ├── yyy{img_suffix}
# │   │   │   ├── zzz{img_suffix}
# │   │   │   ├── ....
# ```

# run the code
python main.py --root_dir ../dataset_path --image_suffix .png --sam_checkpoint ../sam_vit_h_4b8939.pth --model_type vit_h --number_clusters 46 --vis True   