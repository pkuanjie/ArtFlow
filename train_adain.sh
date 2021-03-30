python -u train.py \
--content_dir './data/coco_train/train2014' --style_dir './data/wikiart_train/train' \
--save_dir=experiments/ArtFlow-AdaIN \
--n_flow 8 --n_block 2 --batch_size 4 --operator adain 


