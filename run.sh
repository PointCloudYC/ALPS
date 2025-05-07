# setup
# git clone https://github.com/PointCloudYC/ALPS.git
# cd ALPS

# Create a virtual environment with uv
uv venv --python 3.8.5
source .venv/bin/activate


# install pytorch, torchvision, scikit-learn, etc.
# see https://pytorch.org/get-started/previous-versions/
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install scikit-learn==1.3.2

# install vanialla SAM w. edit mode
# Check if segment-anything folder exists
if [ ! -d "segment-anything" ]; then
    echo "Cloning segment-anything repository..."
    git clone https://github.com/facebookresearch/segment-anything.git
else
    echo "segment-anything repository already exists"
fi
# install SAM w. edit mode
uv pip install -e ./segment-anything

# download sam checkpoints (ViT-H, ViT-L, ViT-B) if they don't exist
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading ViT-H checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "ViT-H checkpoint already exists"
fi

if [ ! -f "sam_vit_l_0b3195.pth" ]; then
    echo "Downloading ViT-L checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
else
    echo "ViT-L checkpoint already exists"
fi

if [ ! -f "sam_vit_b_01ec64.pth" ]; then
    echo "Downloading ViT-B checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "ViT-B checkpoint already exists"
fi
echo "Completed downloading SAM checkpoints"

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
# python main.py --root_dir ../dataset_path --image_suffix .png --sam_checkpoint ../sam_vit_h_4b8939.pth --model_type vit_h --number_clusters 46 --vis True   
python main.py --root_dir ../iSAID --image_suffix .png --sam_checkpoint sam_vit_h_4b8939.pth --model_type vit_h --number_clusters 46 --vis True   