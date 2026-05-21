# Environment

## 1. **clone the repo**
```bash
git clone https://github.com/cambrian-mllm/cambrian-p.git
conda create -n cambrian-p python=3.11 cmake=3.14.0
conda activate cambrian-p
```
## 2. **install vggt**
```bash
cd cambrian-p/vggt
pip install -e .
pip install hydra-core tensorboard iopath wcmatch fvcore
```

## 3. **setup model env**
```bash
cd ../cambrianp
pip install --upgrade pip  
pip install -e ".[train]"
```

## 4. **install flash-attn**
```bash
pip install torch=='2.4.1+cu121' torchvision=='0.19.1+cu121' torchaudio=='2.4.1+cu121' --index-url https://download.pytorch.org/whl/cu121
# pip install flash-attn --no-build-isolation
pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install accelerate==0.29.3 easydict matplotlib 
pip install roma evo imageio 
pip install OpenEXR
```

