wget https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip
unar updated_weights.zip
rm updated_weights.zip
mkdir -p real-CUGAN/model
mv updated_weights/up2x-latest-no-denoise.pth real-CUGAN/model
rm -r updated_weights
