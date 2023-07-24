
MAINDIR=/home/mzlu/lungsound/sepformer
python preprocess.py \
--in_dir $MAINDIR/data/Libri2Mix/wav8k/min \
--out_dir $MAINDIR/data/Libri2Mix/wav8k/min/json \
--sample_rate 8000