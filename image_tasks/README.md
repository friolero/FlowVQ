

## Requirements


Install the requirements using a `conda` environment:
```
conda create --name flowvq python=3.10
conda activate flowvq
pip install -r image_tasks/requirements.txt
```

Afterwards install `torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1` that is compatible with you cuda version according to [PyTorch](https://pytorch.org/get-started/previous-versions/).

## Data preparation for ImageNet-1K

First download ImageNet-1K from [url](https://www.image-net.org/download.php). 

Then convert into [webdataset](https://github.com/webdataset/webdataset) format used here with:

    ```python3
    cd image_tasks/
    PYTHONPATH=./ python3 scripts/create_sharded_dataset.py --shards=./data/ --data=/PATH_TO_IMAGENET/
    ```

## Training

1. Tokenization configuration files for CIFAR-10 and ImageNet-1K as listed here as :

    ```bash
    (flowvq) ┌─(~/FlowVQ/image_tasks) ─────────────────────────────────────────────────────(username@hostname)─┐
             └─(07:30:16 on main ✖ ✹ ✭)──> tree configs                                            ──(Wkd,MthXX)─┘
                configs
                |-- cifar10
                |   |-- flow_tokenizer_cifar10_K10240Z10.yaml
                |   |-- flow_tokenizer_cifar10_K1024Z10.yaml
                |   |-- flow_tokenizer_cifar10_K16384Z14.yaml
                |   |-- flow_tokenizer_cifar10_K16384Z14_ent.yaml
                |   |-- maskbit_cifar10_K1024Z10.yaml
                |   |-- maskbit_cifar10_K16384Z14.yaml
                |   |-- maskbit_cifar10_K16384Z14_woent.yaml
                |   |-- optvq_tokenizer_cifar10_K10240Z10.yaml
                |   |-- optvq_tokenizer_cifar10_K1024Z10.yaml
                |   |-- optvq_tokenizer_cifar10_K16384Z14.yaml
                |   |-- vqgan_plus_cifar10_K10240Z10.yaml
                |   |-- vqgan_plus_cifar10_K1024Z10.yaml
                |   |-- vqgan_plus_cifar10_K16384Z14.yaml
                |   `-- vqgan_plus_cifar10_K16384Z14_ent.yaml
                `-- imagenet
                    `-- flow_tokenizer_imagenet_K16384Z14E1k.yaml
    ```

    MLM configuration files for ImageNet-1K:

    ```bash
    (flowvq) ┌─(~/FlowVQ/image_tasks) ─────────────────────────────────────────────────────(username@hostname)─┐
             └─(07:30:16 on main ✖ ✹ ✭)──> tree configs                                            ──(Wkd,MthXX)─┘
                configs
                `-- imagenet
                    `-- flow_mlm_imagenet_K16384Z14.yaml
    ```

2. ### Tokenization 

    Single GPU training after replacing the path to target config file with:

    ```python3
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ WORKSPACE=./ accelerate launch --num_machines=1  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_tokenizer.py config=/PATH_TO_CONFIG.YAML/

    ```

    Authenticate with `wandb login` or an exported `WANDB_API_KEY` when using WandB. The model checkpoint is saved under `outputs/{RUN_NAME}/checkpoint-X/ema_model/pytorch_model.bin`. 

    For cluster training with Accelerate, [Accerelate](https://huggingface.co/docs/accelerate/en/index) directly supports cluster training with modifications based on your cluster configuration. To keep batch size conistent, you may update batch size related hyper-parameter through arguments directly through command line with `training.per_gpu_batch_size=16 training.gradient_accumulation_steps=8`.  

    Here is the example on our cluster running command below once per node from the FlowVQ checkout (for example, in a separate tmux session), which uses two nodes with four GPUs each:

    ```bash
    pushd /path/to/maskbit
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export WORKSPACE=/path/to/shared/maskbit_runs
    export MASTER_ADDR=MASTER_NODE_IP  # Address reachable from every node
    export MASTER_PORT=29500
    export MACHINE_RANK=0             # Use 1 on the second node

    PYTHONPATH=./ uv run accelerate launch --multi_gpu \
      --num_machines=2 --num_processes=8 \
      --machine_rank="$MACHINE_RANK" \
      --main_process_ip="$MASTER_ADDR" --main_process_port="$MASTER_PORT" \
      --same_network --mixed_precision=no \
      scripts/train_tokenizer.py \
      config=./configs/flow_tokenizer_imagenet_K16384Z14E1k.yaml \
      training.per_gpu_batch_size=4 training.gradient_accumulation_steps=8
    popd
    ```

    `--num_processes` is the **total GPU count across all nodes**. Each node needs a unique rank from `0` to `num_machines - 1`; use the same master address, port, config, and shared `WORKSPACE` on every node. Dataset paths in the config must be accessible on each node. The example has an effective batch size of 256 (4 samples per GPU, 8 GPUs, 8 accumulation steps). 


3. ### Image Generation with MLM

    ```python3
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ WORKSPACE=./ accelerate launch --num_machines=1  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_mlm.py config=./configs/imagenet/flow_mlm_imagenet_K16384Z14.yaml training.per_gpu_batch_size=16 training.gradient_accumulation_steps=8 experiment.vqgan_checkpoint=outputs/{RUN_NAME}/checkpoint-X/ema_model/pytorch_model.bin
    ```

## Pretrained Checkpoints

The pretrained checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1Q5fkqgQTSiOVoJ2EoPxzB_kiTr1v4-HI?usp=drive_link)
