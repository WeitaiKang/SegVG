**SegVG: Transferring Object Bounding Box to Segmentation for Visual Grounding**
========

This repository is an official PyTorch implementation of the ECCV 2024 paper **SegVG: Transferring Object Bounding Box to Segmentation for Visual Grounding**

## **Introduction**
We present **SegVG**, a novel method transfers the box-level annotation as **Seg**mentation signals to provide an additional pixel-level supervision for **V**isual **G**rounding.
To mitigate domain discrepancy, we introduce the Triple Alignment module, where the query, text, and vision tokens are triangularly updated to share the same space.
For more details, please refer to our paper. Code is coming soon!

## **Model Zoo**
**Referring Expression Comprehension**
- **RefCOCO**

| Model             | val     | testA  | testB  | model |
|-------------------|---------|--------|--------|-------|
| SegVG     | 86\.84  | 89\.46 | 83\.07 | Coming soon  |

- **RefCOCO+**

| Model             | val     | testA  | testB  | model |
|-------------------|---------|--------|--------|-------|
| SegVG     | 77\.18  | 82\.63 | 67\.59 | Coming soon  |   

- **RefCOCOg**

| Model             | val-g     | val-u  | test-u  | model |
|-------------------|---------|--------|--------|-------|
| SegVG     | 76\.01  | 78\.35 | 77\.42 | Coming soon  |    

- **ReferItGame**

| Model             | test  | model |
|-------------------|--------|-------|
| SegVG     | 75\.59 | Coming soon  |    
