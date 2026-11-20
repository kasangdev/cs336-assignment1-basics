# CS336 Spring 2026 · A1 Language Modeling Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

这是基于 Stanford CS336 Spring 2026 A1 的个人自学实现，对应讲义 v26.0.3。项目从字节级 BPE 开始，用 PyTorch 实现因果 Transformer 和训练流程，并在 TinyStories 上比较学习率。[课程主页](https://cs336.stanford.edu/) · [课程起始仓库](https://github.com/stanford-cs336/assignment1-basics)

## 代码导航

| 路径 | 用途 |
| --- | --- |
| [`cs336_basics/train_bpe.py`](cs336_basics/train_bpe.py)、[`tokenizer.py`](cs336_basics/tokenizer.py) | 训练 BPE 词表、编码和解码 |
| [`cs336_basics/transformer.py`](cs336_basics/transformer.py)、[`attention.py`](cs336_basics/attention.py) | Transformer、因果注意力，以及 RoPE、RMSNorm 和前馈层的组装 |
| [`cs336_basics/optimizers.py`](cs336_basics/optimizers.py)、[`schedules.py`](cs336_basics/schedules.py) | AdamW、梯度裁剪与学习率调度 |
| [`train.py`](train.py) | 训练、验证、保存检查点与记录吞吐量 |
| [`generate.py`](generate.py) | 加载已训练的检查点与分词器，生成文本 |
| [`plot_metrics.py`](plot_metrics.py)、[`compare_runs.py`](compare_runs.py) | 绘制单次运行的损失和多组实验的验证损失 |
| [`tests/`](tests/) | 课程测试、适配器及测试用的固定样本 |

[`run_tinystories.py`](cs336_basics/run_tinystories.py) 和 [`check_tinystories.py`](cs336_basics/check_tinystories.py) 是带本机固定路径的早期词表实验脚本，目前读写的是 OpenWebText 词表。

## 环境与运行

从本项目根目录打开 PyCharm Terminal。项目使用 `uv` 管理依赖。本机使用 Python 3.12 与 PyTorch 2.11.0+cu128。

```powershell
uv sync --locked
uv run --locked pytest
uv run --locked python train.py --help
uv run --locked python generate.py --help
```

`train.py` 的训练集和验证集参数指向一维整数 token ID 的 NumPy `.npy` 文件。先用同一份 BPE 词表编码两个数据集，再把路径传入。下面的示例使用默认小模型做 20 步检查，输出目录需为新目录。

```powershell
uv run --locked python train.py --train-path data/tinystories_train.npy --valid-path data/tinystories_valid.npy --num-iterations 20 --checkpoint-dir outputs/tinystories-smoke
```

训练目录中的 `config.json`、`metrics.jsonl` 和 `performance.json` 分别保存配置、损失和性能指标。运行 `generate.py` 时，词表和架构参数需与训练配置匹配。`plot_metrics.py` 接收 `--metrics-path` 和 `--output-dir`。`compare_runs.py` 接收 `--run-dirs` 和新的 `--output-dir`。

## TinyStories 学习率实验

三组均在 RTX 3060 Laptop GPU 上训练 5,000 步。批量为 16，上下文长度为 256，随机种子为 42。三组的最低学习率均为各自峰值的十分之一。

| 峰值学习率 | 第 5,000 步验证交叉熵 | 最低观测验证交叉熵 | 训练耗时 |
| ---: | ---: | ---: | ---: |
| 0.0005 | 1.8427 | 1.7869 | 16 分 26 秒 |
| 0.001 | **1.7906** | **1.7352** | 15 分 19 秒 |
| 0.002 | 1.9419 | 1.8983 | 15 分 22 秒 |

峰值学习率 `0.001` 在这组单种子实验中取得了最低的最终验证交叉熵。OpenWebText 另做过短训练和解码尝试。架构变体的入口已接入，正式对照实验待完成。

## 来源与发布

课程起始代码使用 MIT 许可，分发其中的代码时保留 Stanford University 的版权和许可声明。数据、检查点、词表与合并表等实验产物单独保存在本地。课程相关的解答发布还要遵守适用的课程规定。[LICENSE](LICENSE) · [课程 Honor Code](https://cs336.stanford.edu/#honor-code)

<details>
<summary>课程起始仓库的安装与原始数据下载说明</summary>

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

</details>

