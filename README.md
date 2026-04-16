# RaPaCL-ST
RadiomicsFeature-Pathomics Contrastive Learning for Spatial Transcriptomics prediction. 

**RaPaCL-ST (RadiomicsFeature-Pathomics Contrastive Learning for Spatial Transcriptomics prediction)** is a multimodal representation learning framework designed to bridge handcrafted radiomics features and deep pathomics features derived from histopathology images, which aim to predict spatial gene expression value in patch-wise level. In this approach, radiomics features extracted from image patches serve as structured, interpretable signals, while deep learning based patch image encoders encode high-dimensional visual representations. RaPaCL leverages contrastive learning to align these two modalities in a shared latent space, encouraging consistency between radiomics-informed characteristics (e.g., texture, heterogeneity) and deep image embeddings. By doing so, the framework aims to enhance the biological relevance and interpretability of learned representations, ultimately improving downstream tasks such as spatial gene expression prediction and tumor characterization in whole-slide images. 

![mainfig](mainfig.png)

---

## Description 

### Prepare Data & Run Baselines

Please refer to: `src/dataset/README.md` and `src/baselines/README.md`. 

---

### Run RaPaCL

... 

### Pretrain Radiomics TransTab 

```bash
cd RaPaCL/
```

(i) prepare tabular custom data 
- from H5 (HEST-style)
  ```bash
  python -m src.pretrain_transtab.prepare_tabular \
    --config configs/data/prepare_tabular.yaml 
  ```

(ii) pretrain 

- mode explanation: 
  - `train`: contrastive pretraining
  - `eval`: classifier finetuning + test classification metric (accuracy / macro F1 / AUROC)
  - `eval_detailed`: representation quality 평가 전용 (표현 공간 자체 평가)
    - embedding 추출
    - clustering metrics: Silhouette / NMI / ARI
    - UMAP / t-SNE visualization 
    - label cluster visualization
    - radiomics feature representative points
    - feature quantile section visualization

- pretrain with single GPU 
  ```bash
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode train
  ```

- pretrain with multi GPU 
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --mode train
  ```

- simple evaluation of pretraining
  ```bash 
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode eval
  ```

- detailed evaluation of pretraining
  ```bash
  python -m src.pretrain_transtab.pretrain_transtab \
    --config configs/pretrain_transtab/idc_allxenium.yaml \
    --distributed false \
    --mode eval_detailed
  ```

---

