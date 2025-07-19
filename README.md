# **RecBole-RBSR**
**Review-based Sequential Recommendation (RBSR) with RecBole**

**Recbole-RBSR** implements review-based sequential recommendation on top of RecBole library.
  - Shares the unified API(e.g., models) and input files with [RecBole library](https://github.com/RUCAIBox/RecBole). 
  - Reflects RecBole v1.2.1. (Latest version)   
     
**List of Modified and Added Files/Folders**

    RecBole-RBSR/
    ├── recbole/
    │   ├── data/
    │   │   └── dataset/
    │   │       └── customized_dataset.py
    │   ├── model/
    │   │   ├── layers.py
    │   │   └── sequential_recommender_review_based/
    │   ├── utils/
    │   │   └── utils.py
    │   └── properties/
    │       └── model/
    │           ├── CCA.yaml
    │           ├── IntentRec.yaml
    │           ├── PIRSP.yaml
    │           └── RNS.yaml
    ├── run_rbsr.py


## **Implemented Models**  
| Model      | Paper |
|------------|-------------|
| **RNS**       | A Review-Driven Neural Model for Sequential Recommendation (2019, IJCAI) |
| **PIRSP**    | Integrates review-based user-item interactions into sequential modeling (2021, EAAI) |
| **CCA**      | Cascaded Cross Attention for Review-based Sequential Recommendation (2023, ICDM) |
| **IntentRec**| IntentRec: Incorporating latent user intent via contrastive alignment for sequential recommendation (2025, ECRA) | 
| |

## Dataset
- 아마존 5-score 데이터셋(리뷰포함)
    - 업로드(MI, AM) 
- 변환 코드 구현 Based on [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) (update soon).

## Example Usage

```
python run_rbsr.py --model IntentRec --dataset MI
```