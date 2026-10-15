# 被测系统版本（运行前冻结）

| 系统 | 来源 | 版本 |
| --- | --- | --- |
| A-MEM | github.com/agiresearch/A-mem | commit ceffb860f0712bbae97b184d440df62bc910ca8d |
| MemoryOS | github.com/BAI-LAB/MemoryOS（memoryos-pypi） | commit 587ed7755c7aed179965792830ff1b5ad9a6fa92 |
| Mem0 | PyPI mem0ai | 2.2.1（spaCy en_core_web_sm 3.8.0） |
| 数据 | snap-research/locomo data/locomo10.json | sha256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4（与 pilot30 manifest 一致） |

复现：`git clone` 上述两个仓库到 `vendor/` 并 checkout 对应 commit；LoCoMo 放到 `data/locomo10.json`。
