<!-- ![alt-text-1]("title-1") ![alt-text-2](image2.png "title-2") -->
The implementation of NAACL 2021 paper 
["Event Representation with Sequential, Semi-Supervised Discrete Variables".](https://arxiv.org/pdf/2010.04361.pdf)

<img src="figs/example_final.png" width="200"/> <img src="figs/model_structure.png" width="600"/> 

This Pytorch code implements the model and reproduces the results from the paper.
# Data:
[Wikipedia Dataset](https://drive.google.com/file/d/1abSJI7Kbm_EaZfZYqTEGocwGVgX4_mBy/view?usp=sharing)

[Wikipedia Inverse Narrative Cloze](https://drive.google.com/file/d/1markcg4CfjJQeKbZ_qtCd17rmhCDO4TH/view?usp=sharing)

[NYT Inverse Narrative Cloze](https://drive.google.com/file/d/1Cjiz2aGdpT9wEHz395VG8bcDEZzjh-4N/view?usp=sharing)


# Training:
```
./train.sh $obsv_prob $exp_num $seed
```

# data_mode:
 {'valid','test'}

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

