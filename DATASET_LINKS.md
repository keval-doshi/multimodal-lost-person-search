# Dataset and Model Links

MultiModal Lost Person Search System Using Image and Text Query Across Multiple Cameras
Keval Hirenbhai Doshi — M01086516 — CST4275 — Middlesex University

All data used in this project is public and released for research use. Nothing was
trained, so the three datasets and three pretrained models below are everything needed to
reproduce the results. Links were checked on 1 October 2026.

---

## 1. Datasets

### PRW — Person Re-identification in the Wild (main benchmark)

Used for: whole camera frames, person boxes and identities (all detection and retrieval
results in Chapter 5).

| | |
|---|---|
| Contents | 11,816 frames from 6 cameras, 932 identities, 2,057 query boxes |
| Official project page | https://zheng-lab-anu.github.io/Project/project_prw.html |
| Download (OneDrive / SharePoint, v2016.04.20, 2.67 GB) | https://anu365-my.sharepoint.com/:u:/g/personal/u1064892_anu_edu_au/EWLYrB0RPeJLksLl-6rG854BMJL9I_ZQxyYT3EnwgEZLnA?e=qn8sfY |
| Download (Baidu mirror) | http://pan.baidu.com/s/1i5tqYbj |
| Baseline code (also lists the download) | https://github.com/liangzheng06/PRW-baseline |
| Paper | Zheng et al. (2017) *Person re-identification in the wild*, CVPR — https://arxiv.org/abs/1604.02531 |

### PRW-TBPS — text descriptions for PRW

Used for: the written clothing descriptions (two independent descriptions per test search).
These are annotation files only; the images come from PRW above.

| | |
|---|---|
| Contents | test: 2,057 query boxes × 2 descriptions, 450 identities; train: 14,897 boxes, 483 identities |
| Official repository | https://github.com/Dacun/Text-based-Person-Search |
| PRW-TBPS_test (Google Drive) | https://drive.google.com/file/d/1iyxa_BqXFsNSxTbHpczdypB0shNKp86c/view?usp=sharing |
| PRW-TBPS_train (Google Drive) | https://drive.google.com/file/d/1Oh3nOWQfG4UkAJsILNeW3uZV1Mggwip7/view?usp=sharing |
| Paper | Zhang et al. (2023) *Text-based person search in full images via semantic-driven proposal generation*, ACM MM Workshop — preprint https://arxiv.org/abs/2109.12965 |

### Market-1501 — instrument check only

Used for: one check that the appearance model (OSNet) loaded correctly before encoding
(Section 3.3.3). It plays no part in the reported results.

| | |
|---|---|
| Contents | 32,668 cropped images, 1,501 identities, 6 cameras (3,368 queries) |
| Official project page | https://zheng-lab-anu.github.io/Project/project_reid.html |
| Download (Google Drive) | https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing&resourcekey=0-8nyl7K9_x37HlQm34MmrYQ |
| Download (Academic Torrents, the copy used in this project) | https://academictorrents.com/details/3ea1f8ae1d3155addff586a96006d122587663ee |
| Paper | Zheng et al. (2015) *Scalable person re-identification: a benchmark*, ICCV — https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html |

---

## 2. Pretrained models (published weights, used without any training)

| Model | Role | Download |
|---|---|---|
| YOLO11m (Ultralytics) | person detector | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt — also downloaded automatically by `phase2_detect.py` on first use. Docs: https://docs.ultralytics.com/models/yolo11/ |
| OSNet x1.0, trained on MSMT17 | appearance model | https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth — downloaded automatically by `phase3_sanity.py`. Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html |
| IRRA (CLIP ViT-B/16), CUHK-PEDES checkpoint | vision-language / text model | Checkpoint: https://drive.google.com/file/d/1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW/view?usp=share_link — code: https://github.com/anosorae/IRRA |

---

## 3. Expected folder layout

The scripts are run from the repository root with `--root .\dataset`. After downloading, arrange the files as:

```
dataset/
  PRW/
    frames/            (11,816 .jpg frames)
    annotations/
    query_box/
  PRW-TBPS/
    PRW-TBPS_train
    PRW-TBPS_test
  Market-1501/
    bounding_box_train/
    bounding_box_test/
    query/
models/
  osnet_x1_0_msmt17.pth   (OSNet weights, saved under this short name)
  irra/best.pth           (IRRA checkpoint)
```

Run `python code\verify.py --root .\dataset` first: it checks every count above against the
published figures and confirms that each PRW-TBPS description joins to a real PRW frame.
The full command sequence is in `README.md`.

## 4. Terms of use

These datasets are provided by their authors for non-commercial research. Anyone
downloading them should cite the papers listed above and follow the terms on each
project page. No images from these datasets are redistributed in this repository.
