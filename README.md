# DELTAR: Depth Estimation from a Light-weight ToF Sensor And RGB Image
### [Project Page](https://zju3dv.github.io/deltar/) | [Paper](https://arxiv.org/pdf/2209.13362.pdf)
<br/>

> DELTAR: Depth Estimation from a Light-weight ToF Sensor And RGB Image  
> [Yijin Li](https://github.com/eugenelyj), [Xinyang Liu](https://github.com/QsingHuan), [Wenqi Dong](https://github.com/wqdong8), [Han Zhou](https://github.com/asdiuzd), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang), [Yinda Zhang](https://www.zhangyinda.com), [Zhaopeng Cui](https://zhpcui.github.io)  
> ECCV 2022

![Demo Video](https://github.com/eugenelyj/open_access_assets/raw/master/deltar/comp_realsense_short.gif)


## Download Link

We provide the download link [[google drive](https://drive.google.com/drive/folders/1ZGUdagrmFDr90Lm6qG1FkbZR_Tgpmr64?usp=share_link), [baidu](https://pan.baidu.com/s/13qoVoZejiRzmoJGkFJdh0w?pwd=1i11)(code: 1i11)] to
  - pretrained model trained on NYU.
  - ZJUL5 dataset.
  - demo data.


## Run DELTAR

### Installation
```bash
conda create --name deltar --file requirements.txt
```

### Prepare the data and pretrained model
Download from the above link, and place the data and model as below: 


```
deltar
├── data
│   ├── demo
│   └── ZJUL5
└── weights
    └── nyu.pt
```

### Evaluate on ZJUL5 dataset

```bash
python evaluate.py configs/test_zjuL5.txt
```

### Run the demo

```bash
python evaluate.py configs/test_demo.txt
python scripts/make_gif.py --data_folder data/demo/room --pred_folder tmp/room
```


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{deltar,
  title={DELTAR: Depth Estimation from a Light-weight ToF Sensor and RGB Image},
  author={Li Yijin and Liu Xinyang and Dong Wenqi and Zhou han and Bao Hujun and Zhang Guofeng and Zhang Yinda and Cui Zhaopeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Acknowledgements

We would like to thank the authors of [Adabins](https://github.com/shariqfarooq123/AdaBins), [LoFTR](https://github.com/zju3dv/LoFTR) and [Twins](https://github.com/Meituan-AutoML/Twins) for open-sourcing their projects.

