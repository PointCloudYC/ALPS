# setup
git clone https://github.com/PointCloudYC/ALPS.git
cd ALPS

# Create a virtual environment with uv
uv venv --python 3.8.5
source .venv/bin/activate

# install SAM w. edit mode
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
# install SAM w. edit mode
uv pip install -e ./segment-anything

# install pytorch, torchvision, scikit-learn, etc.
uv pip install torch torchvision torchaudio
uv pip install scikit-learn==1.3.2

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