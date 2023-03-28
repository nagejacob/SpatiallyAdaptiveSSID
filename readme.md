# Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising
The source code for paper "[Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising](https://arxiv.org/pdf/2303.14934.pdf)" (CVPR 2023)

## Usage
### Datasets
Download [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) and [DND](https://noise.visinf.tu-darmstadt.de/) datasets, and modify `dataset_path` in `dataset/base.py` accordingly.
```
|- dataset_path
  |- SIDD
    |- SIDD_Medium_Srgb
      |- Data
        |- 0001_001_S6_00100_00060_3200_L
        |- 0002_001_S6_00100_00020_3200_N
        |- ...
    |- SIDD_Validation
      |- ValidationNoisyBlocksSrgb.mat
      |- ValidationGtBlocksSrgb.mat
    |- SIDD_Benchmark
      |- BenchmarkNoisyBlocksSrgb.mat
  |- DND
    |- info.mat
    |- images_srgb
```

### Validation
Validate on SIDD Validation dataset,
```
cd validate
python validate_SIDD.py
```

### Training
Training on SIDD Medium dataset,
```
sh train.sh
```

## Citation
If you find our work useful in your research or publication, please cite:
```
@inproceedings{li2023spatially,
  title={Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising},
  author={Li, Junyi and Zhang, Zhilu and Liu, Xiaoyu and Feng, Chaoyu and Wang, Xiaotao and Lei, Lei and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
