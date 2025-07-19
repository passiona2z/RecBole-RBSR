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
| **RNS**       | [A Review-Driven Neural Model for Sequential Recommendation](https://www.ijcai.org/proceedings/2019/397) (2019, IJCAI) [[code]](https://github.com/WHUIR/RNS) |
| **PIRSP**    | [Integrates review-based user-item interactions into sequential modeling](https://www.sciencedirect.com/science/article/pii/S0952197621001962) (2021, EAAI) |
| **CCA**      | [Cascaded Cross Attention for Review-based Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/10415676) (2023, ICDM) [[code]](https://github.com/BING303/CCA)|
| **IntentRec**| [IntentRec: Incorporating latent user intent via contrastive alignment for sequential recommendation](https://www.sciencedirect.com/science/article/abs/pii/S156742232500047X) (2025, ECRA) | 
| |


- (Note)
  - The origin code of RNS and CCA was revised for the RecBole library.
  - PIRSP extends RNS with an item sequence encoder.
  - IntentRec is the contributor's own paper and implementation.


## Dataset
[Amazon Review 5-core dataset](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)

- Provided subsets:
  - Musical Instruments (MI)
  - Automotive (AM)

- Data conversion scripts based on [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) will be update soon. 

## Example Usage

```
python run_rbsr.py --model IntentRec --dataset MI
```