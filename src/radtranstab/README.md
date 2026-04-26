## Pretrain Radiomics TransTab 

```bash
cd RaPaCL/src/radtranstab
```

(i) prepare data 
```bash 

```

(ii) pretrain 
```bash 
python -m radtranstab.run \
  --train_jsonl_file /path/to/train.jsonl \
  --val_jsonl_file /path/to/val.jsonl \
  --root_dir /path/to/root \
  --hdf5_file /path/to/data.h5 \
  --checkpoint_path /path/to/full_model_checkpoint.pth \
  --output_dir outputs/radiomics_only \
  --device cuda:0 \
  --batch_size 16 \
  --epochs 100 \
  --use_amp
```

(iii) evaluate 
```bash 

```

