# TensorRT deployment-2706

Reproducing and fixing TensorRT issue.

## Reproducibility

```
module load python/3.12.12

python -m venv ...

activate ...

pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
pip install torch-tensorrt   

git submodule add git@github.com:MIC-DKFZ/nnUNet.git external/nnUNet
cd external/nnUNet
git checkout 0e49508

cd nnUNet
pip install -e .

# if dataset not processed
nnUNetv2_plan_and_preprocess -d 027 -pl nnUNetPlannerResEncL

```

## Nsight kernel time stats
```bash
nsys profile \                                                                  
   --trace=cuda,nvtx,cudnn,cublas,osrt \                                         
   --pytorch=autograd-shapes-nvtx \                                              
   --python-backtrace=cuda \                                                     
   --python-sampling=true \                                                      
   --force-overwrite=true \                                                      
   -o ./profiling/nsys- \                                   
   python ....py 
   
nsys stats \                                                                    
  --report cuda_gpu_kern_sum \                                                  
  --force-export=true \                                                         
  "$OUTPUT_DIR/....nsys-rep" \                             
  > "$OUTPUT_DIR/..._kernels.txt"
```

chmod +x profile_experiment.sh   # already done, but just in case
./profile_experiment.sh 2d trt-solution -- \
    --dataset-name Dataset306_BONE_TUMOR_EXTENDED \
    --nnunet-preprocessed /home/maxrodri/Datasets/nnunet_preprocessed/Dataset306_BONE_TUMOR_EXTENDED \
    --plans-filename nnUNetPlans.json \
    --compiled-engines-dir ./Dataset306_BONE_TUMOR_EXTENDED

## Worth to try
- `use_fast_partioner` default is True, hence the fastest partion found might not be the optimal choice globally
- `num_avg_timing_iters` deafult is 1, increase for stability reasons?
- `max_aux_streams` default is None, max allowed TRT streams per engine, don't understand this one
- `truncate_double` default is Falsea
- `tiling_optimization_level` default is None,  [“none”, “fast”, “moderate”, “full”]
    - Tiling can substantially improve throughput for convolution-heavy models.
- `require_full_compilation` set to True correctness gate as nnUNet is currently fully TRT-compatible

## Profiling wiht trtExec
```bash
python compile_and_save.py \
    --dataset-name Dataset027_ACDC \
    --nnunet-preprocessed /lustre/uc3m_a0/dynamic/maxrodri/datasets/nnUNet/nnUNet_preprocessed/Dataset027_ACDC \
    --plans-filename nnUNetResEncUNetLPlans.json \
    --mode trt-solution-autocast --configurations 3d_fullres \
    --export-raw-engine

trtexec --loadEngine=Dataset027_ACDC/ \
        --iterations=50 --avgRuns=50 \
        --profilingVerbosity=detailed \
        --dumpProfile --exportProfile=a40_fullres_profile.json
```
