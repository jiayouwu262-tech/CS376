# Intel BoW 跨域评估

本目录是小组已有 Stanford 复现之外新增的 **Intel 数据准备和 BoW 外部评估**。直接加载仓库已有的 `../results/model_BoW_wiki.joblib` 和 `../results/model_BoW_se.joblib`，不重复上传模型，不重新训练。

## 实际结果

同一份 Mapping A 官方 test 共 5,099 条：polite=1 为 2,567 条，impolite=0 为 2,532 条。

| 训练来源 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Wikipedia | 77.60% | 0.7893 | 0.7573 | 0.7730 | 0.8587 |
| Stack Exchange | 69.39% | 0.6706 | 0.7702 | 0.7170 | 0.7717 |

Precision、recall、F1 以 polite=1 为正类；AUC 使用指向 polite=1 的连续 SVM margin。CSV 中各项指标统一为0–1小数。

模型沿用小组 `CountVectorizer(min_df=10)` unigram counts 和 `LinearSVC(C=1.0,max_iter=20000,random_state=42)`。小组 Stanford 参考分数为 Wiki→SE 59.72%、SE→Wiki 71.07%；本目录没有重新运行这些 Stanford 实验。Intel 只输入 text，没有用于词表拟合、训练或调参。

## 文件说明

- `data/intel_cleaned/intel_test_mapping_a.csv`：本次主评估测试集。
- `outputs/intel_bow/dataset_counts.csv`：全部14份清洗产物的行数、类别数和SHA256；此次仅提交主评估所需的test CSV，其余可通过准备脚本生成。
- `outputs/intel_bow/metrics_summary.csv`：两组完整指标和混淆矩阵。
- `outputs/intel_bow/*_predictions.csv`：每个样本的ID、真实标签、预测标签和连续分数。
- `outputs/intel_bow/intel_mapping_a_sample_registry.csv`：ID、原始行号、文本和文本哈希，供Ling配对。
- `scripts/evaluate_intel_bow.py`：只调用已拟合模型的transform/predict/decision_function。
- `scripts/prepare_intel.py`：仓库原`clean_intel.py`的增强副本，增加稳定ID和输出路径参数，原清洗/映射/采样规则不变，保留原根目录脚本。
- `configs/intel_bow_models.json`：引用仓库已有`results/`模型；不要单独移动本目录中的文件。

## 复跑评估

在仓库的 `intel_bow/` 目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/evaluate_intel_bow.py --config configs/intel_bow_models.json
```

模型保存于scikit-learn 1.9.1，需要使用相同版本。模型pipeline已内嵌原拟合向量器；不需要ConvoKit或spaCy即可运行这次评估。

## 清洗与样本ID

数据来自 [Intel/polite-guard](https://huggingface.co/datasets/Intel/polite-guard)，revision `7b34586171914fa7ab75fd275aca8469e1464a64`。清洗包括去首尾空白、合并连续空白、split内去重、仅从训练集删除与test重叠的样本；保留大小写与标点。

数据集声明许可为CDLA-Permissive-2.0，完整文本附于`licenses/CDLA-Permissive-2.0.txt`。

清洗后train/validation/test为79,997/10,000/10,200；Mapping A仅保留polite=1与impolite=0，其test为5,099条。Mapping B及四份平衡训练样本仅作数据准备，本次不参与训练或主评估。

上游没有ID；脚本在过滤前生成`sample_id=intel_{原始split}_{原始零基行号:06d}`，预测文件沿用同一ID和行序。不要使用独立过滤后重编的行号与Ling配对。

从固定版本重新生成全部清洗产物：

```powershell
.venv/Scripts/python.exe scripts/prepare_intel.py --revision 7b34586171914fa7ab75fd275aca8469e1464a64 --output-dir data/intel_cleaned --cache-dir data/hf_cache
```

此命令需要联网；直接复跑评估使用已提交的test CSV即可。
