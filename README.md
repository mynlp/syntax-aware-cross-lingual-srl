# Graph Neural Networks for Syntax Encoding in Cross-Lingual Semantic Role Labeling

## About
This is the implementation code for our paper: "Graph Neural Networks for Syntax Encoding in Cross-Lingual Semantic Role Labeling". Feel free to use this repository to reproduce our results for research purposes. We do not provide any data in this repository, therefore please prepare the data in advance. We refer to the other repositories when implementing the models for our experiments:
1. [diegma/neural-dep-srl](https://github.com/diegma/neural-dep-srl) for Syntactic Graph Convolutional Networks (SGCNs)
2. [dmlc/dgl](https://github.com/dmlc/dgl) for Relational Graph Convolutional Networks (RGCNs)
3. [AnWang-AI/towe-eacl](https://github.com/AnWang-AI/towe-eacl) for Attention-Based Graph Convolutional Networks (ARGCNs)
4. [gordicaleksa/pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT) for Graph Attention Networks (GATs)
5. [shenwzh3/RGAT-ABSA](https://github.com/shenwzh3/RGAT-ABSA) for Relational Graph Attention Networks (RGATs)
6. [thudm/hgb](https://github.com/thudm/hgb) for Simple Heterogeneous Graph Neural Networks (SHGNs)
7. [deepakn97/relationPrediction](https://github.com/deepakn97/relationPrediction) for Knowledge-Based Graph Attention Networks (KBGATs)
8. [wasiahmad/GATE](https://github.com/wasiahmad/GATE) for Graph Attention Transformer Encoders (GATEs)

## Requirements
We provide the versions of libraries that we use:
1. `huggingface-hub`: `0.15.1`
2. `matplotlib`: `3.7.1`
3. `numpy`: `1.24.3`
4. `pandas`: `2.0.2`
5. `prettytable`: `3.7.0`
6. `python`: `3.8.16`
7. `scipy`: `1.10.1`
8. `seaborn`: `0.12.2`
9. `tokenizers`: `0.13.3`
10. `torch`: `1.13.0+cu116`
11. `torch-scatter`: `2.1.1+pt113cu116`
12. `tqdm`: `4.65.0`
13. `transformers`: `4.29.2`

## Corpus Preprocessing
1. Clone SRL annotations for target languages from [Universal Proposition Bank (UPB) v2](https://github.com/UniversalPropositions) and for [English](https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT).
2. Run `helper_scripts/fix_upb_2.py` to fix shifted annotation problem caused by enhanced dependency tree annotations in treebanks. Make sure the paths written in the Python script are correct. The treebanks that contain enhanced dependency tree annotations are:
   1. `Czech-CAC`
   2. `Czech-FicTree`
   3. `Czech-PDT`
   4. `Dutch-Alpino`
   5. `Dutch-LassySmall`
   6. `Finnish-TDT`
   7. `Italian-ISDT`
   8. `Spanish-AnCora`
   9. `Ukrainian-IU`
3. Download annotations from [Universal Dependencies (UD) v2.9](http://hdl.handle.net/11234/1-4611).
4. Clone [UPB tools](https://github.com/UniversalPropositions/tools) to merge UPB annotations with UD annotations.
5. Setup UPB tools.
6. Run this command from UPB tools to merge UPB annotations and UD annotations and obtain complete SRL annotations for a certain treebank:
```
python3 up2/merge_ud_up.py --input_ud=<ud-treebank> --input_up=<up-treebank> --output=<merged-treebank>
```
6. Change the paths mentioned at the `metadata_by_version_to_lang_to_treebank` inside `constants/dataset.py` file to point to the correct paths. `UP-2.0` must point to the complete SRL annotations using gold POS tags and dependency trees. `UP-2.0-predicted` must point to complete SRL annotations using predicted POS tags and dependency trees. 

## Traning
1. We provide a Python script in `helper_scripts/create_script.py` to generate a script to run the training and evaluation for each model. The codenames for models are as follows:
   1. `gcn_syn` for SGCNs
   2. `gcn_ar` for ARGCNs
   3. `gat_plain` for GATs
   4. `lstm` for BiLSTMs
   5. `gat_het` for SHGNs
   6. `gat_two_att` for RGATs
   7. `gat_kb` for KBGATs
   8. `gcn_r` for RGCNs
   9. `trans_rpr` for Self-Attention with Relative Position Representations (SAN-RPRs)
   10. `gate` for GATEs
   11. `trans` for Transformers
   12. `trans_spr` for Self-Attention with Structural Absolute Position Representations (SAN-SAPRs)
   13. `trans_spr_rel` for Self-Attention with Structural Relative Position Representations (SAN-SRPRs)
   14. `trans_x_spr_rel` for Transformers with SRPRs (Trans-SRPRs)
   15. `trans_rpr_x_spr` for SAN-SAPRs with SAN-RPRs (SAPR-RPRs)
   16. `trans_x_spr_rel_x_dr` for Trans-SRPRs with DR (Trans-SRPR-DRs)
   17. `trans_rpr_x_spr_x_dr` for SAPR-RPRs with DR (SAPR-RPR-DRs)
   18. `trans_rpr_x_spr_x_ldp` for SAPR-RPRs with LDPs (SAPR-RPR-LDPs)
   19. `trans_rpr_x_spr_x_sdp` for SAPR-RPRs with SDPs (SAPR-RPR-SDPs)
2. Fill the codename of the desirable model in `model_list` variable to generate the script for training a certain model.
3. Fill the desirable parameters for the model in `params` variable. The explanation for each parameter can be found in the Appendix of the paper.
4. Run the Python script in `helper_scripts/create_script.py`. Make sure the paths written in the Python script are correct.
5. The script to train and evaluate the model will be available at `scripts/tests`.
6. Run the script with `bash <script-name>` to start the training process.
7. After the training and evaluation finish, the logs and models will be available at directories with `_logs` and `_models` suffixes.

## Licence
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
