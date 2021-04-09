<!-- ![alt-text-1]("title-1") ![alt-text-2](image2.png "title-2") -->
The implementation of NAACL 2021 paper 
["Event Representation with Sequential, Semi-Supervised Discrete Variables".](https://arxiv.org/pdf/2010.04361.pdf)

<img src="figs/model_structure.png" width="600"/> 

This Pytorch code implements the model and reproduces the results from the paper.

# Training:
```
./train.sh $obsv_prob $exp_num $seed
```
---
# data_mode:
 {'valid','test'}

---

# Perplexity:
```
./test_ppx.sh $obsv_prob $exp_num $seed $data_mode
```
# Wiki Inverse Narrative Cloze:
```
./wiki_inv_narr.sh $obsv_prob $exp_num $seed $data_mode
```

# NYT Inverse Narrative Cloze:
```
./nyt_inv_narr.sh $obsv_prob $exp_num $seed $data_mode
```

