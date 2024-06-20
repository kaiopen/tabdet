# Travelable Area Boundary Detection

[中文](README_ZH.md)

### Python Environments
- tqdm
- NumPy
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)
- [TABKit](https://github.com/kaiopen/tab_kit)

### Preparing
The prepared will be save in the directory assigned by the item `preprocess.dst` in the configuration file. Run the command:
```
python tools/preprocess.py --config=<configuration file> --split=<split>
```
For example,
```shell
python tools/preprocess.py --config=TAB_Pillar256x512x20_UNet.yaml --split=train
```

### TRAINING
The training will use the GPU devices as much as can be gotten. You can limit and assign the device(s) via `CUDA_VISIBLE_DEVICES`. Checkpoints will be save in `./checkpoints/`.
```shell
torchrun --nproc_per_node=2 tools/train_ddp.py --config=TAB_Pillar256x512x20_UNet.yaml --split=train --batch_size=16 --num_worker=10 --end_epoch=200
```

### EVALUATION
A loss curve figure will be generated and saved in `./eval/`. We have not done evaluation when training. You can modify the `tools/train_ddp.py` and try to do evaluate during training.
```shell
python tools/eval.py --config=TAB_Pillar256x512x20_UNet.yaml --batch_size=64 --num_worker=10
```

### VISUALIZATION
Prediction results will be visualized and save in `./vis`.
```shell
python tools/vis_tab.py --config=TAB_Pillar256x512x20_UNet.yaml --split=test --batch_size=64 --num_worker=10 --checkpoint=199
```

### TEST
Assess the well-trained models. Results will be saved in `./results/`.
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
        <td style="text-align: center;">UNet</td>
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
        <td style="text-align: center;">HRNet-w18</td>
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

You can download well-trained checkpoints and move them into checkpoint directories such as `./checkpoints/TAB_Pillar256x512x20_UNet.yaml/`.
