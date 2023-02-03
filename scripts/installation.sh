python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -U opencv-python

pip install -r requirements.txt

# CUDA operators for MS Deformable Attn Decoder
cd vlformer/modeling/pixel_decoder/ops
sh make.sh
