python ./cnn.py \
--data ./data/audio \
--dtype audio \
--label intent \
--num_classes 6 \
--workers 8 \
--num_epochs 75 \
--batch-size 32 \
--lr 0.001 





# epochs batch lr    acc   dropout
# 150    96    0.001  68     0.5
# 200    64    0.001  68     0.5
# 100    32    0.001  72     0.4
#> relu on fc 

# new data

# 100    32    0.001  67     0.4
#  80    32    0.001  68.974 0.4
#  70    32    0.001  68.026 0.4
#  85    32    0.001  68.154 0.4
