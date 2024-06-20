# 可通行区域边缘检测

[English](README.md)

### Python 环境
- tqdm
- NumPy
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)
- [TABKit](https://github.com/kaiopen/tab_kit)

### 预处理
根据配置文件中的配置项 `preprocess.dst`，预处理的结果将保存在对应的路径中。
```
python tools/preprocess.py --config=<configuration file> --split=<split>
```
例如
```shell
python tools/preprocess.py --config=TAB_Pillar256x512x20_UNet.yaml --split=train
```

### 训练
该训练为分布式并行训练。可以通过 `CUDA_VISIBLE_DEVICES` 限制或指定使用的显卡。结果将保存在 `./checkpoints/` 路径中。
```shell
torchrun --nproc_per_node=2 tools/train_ddp.py --config=TAB_Pillar256x512x20_UNet.yaml --split=train --batch_size=16 --num_worker=10 --end_epoch=200
```

### 评估
损失曲线将保存在 `./eval/` 路径中。你也可以通过修改 `tools/train_ddp.py` 实现边训练边评估。
```shell
python tools/eval.py --config=TAB_Pillar256x512x20_UNet.yaml --batch_size=64 --num_worker=10
```

### 可视化
可视化结果将保存在 `./vis` 路径中。
```shell
python tools/vis_tab.py --config=TAB_Pillar256x512x20_UNet.yaml --split=test --batch_size=64 --num_worker=10 --checkpoint=199
```

### 测试
评估训练好的模型。结果保存在 `./results/` 路径中。
```shell
python tools/test.py --config=TAB_Pillar256x512x20_UNet.yaml --split=test --batch_size=64 --num_worker=10 --checkpoint=199
```

### Models
<table border="1">
    <tr>
        <th rowspan="2" style="text-align: center;">Backbone</th>
        <th colspan="4" style="text-align: center;">Straight-going side</th>
        <th colspan="4" style="text-align: center;">Turning</th>
        <th colspan="4" style="text-align: center;">Ignoring semantics</th>
    </tr>
    <tr>
        <th style="text-align: center;">mF_p</th>
        <th style="text-align: center;">F_0.3</th>
        <th style="text-align: center;">F_0.5</th>
        <th style="text-align: center;">F_0.8</th>
        <th style="text-align: center;">mF_p</th>
        <th style="text-align: center;">F_0.3</th>
        <th style="text-align: center;">F_0.5</th>
        <th style="text-align: center;">F_0.8</th>
        <th style="text-align: center;">mF_p</th>
        <th style="text-align: center;">F_0.3</th>
        <th style="text-align: center;">F_0.5</th>
        <th style="text-align: center;">F_0.8</th>
    </tr>
    <tr>
        <td style="text-align: center;"><a href="https://github.com/kaiopen/tabdet/releases/download/UNet/140.pth">UNet</a></td>
        <td style="text-align: center;">0.55</td>
        <td style="text-align: center;">0.50</td>
        <td style="text-align: center;">0.41</td>
        <td style="text-align: center;">0.07</td>
        <td style="text-align: center;">0.71</td>
        <td style="text-align: center;">0.68</td>
        <td style="text-align: center;">0.63</td>
        <td style="text-align: center;">0.41</td>
        <td style="text-align: center;">0.75</td>
        <td style="text-align: center;">0.70</td>
        <td style="text-align: center;">0.67</td>
        <td style="text-align: center;">0.32</td>
    </tr>
    <tr>
        <td style="text-align: center;"><a href="https://github.com/kaiopen/tabdet/releases/download/HRNet/170.pth">HRNet-w18</a></td>
        <td style="text-align: center;">0.69</td>
        <td style="text-align: center;">0.66</td>
        <td style="text-align: center;">0.63</td>
        <td style="text-align: center;">0.56</td>
        <td style="text-align: center;">0.73</td>
        <td style="text-align: center;">0.75</td>
        <td style="text-align: center;">0.69</td>
        <td style="text-align: center;">0.54</td>
        <td style="text-align: center;">0.82</td>
        <td style="text-align: center;">0.83</td>
        <td style="text-align: center;">0.77</td>
        <td style="text-align: center;">0.62</td>
    </tr>
    <tr>
        <td style="text-align: center;">DeepLabV3+</td>
        <td style="text-align: center;">0.62</td>
        <td style="text-align: center;">0.57</td>
        <td style="text-align: center;">0.49</td>
        <td style="text-align: center;">0.35</td>
        <td style="text-align: center;">0.71</td>
        <td style="text-align: center;">0.67</td>
        <td style="text-align: center;">0.61</td>
        <td style="text-align: center;">0.43</td>
        <td style="text-align: center;">0.78</td>
        <td style="text-align: center;">0.73</td>
        <td style="text-align: center;">0.67</td>
        <td style="text-align: center;">0.46</td>
    </tr>
</table>

注：由于上传上限的限制，训练好的 DeepLabV3+ 的模型参数未能成功上传。可以在 issue 中留下邮箱地址以便模型参数能够尽快地找到您。

可以下载以上训练好的模型，并放置在相应的路径中，例如 `./checkpoints/TAB_Pillar256x512x20_UNet.yaml/`，即可进行测试。
