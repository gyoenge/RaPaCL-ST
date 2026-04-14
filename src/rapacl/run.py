"""
Step 1. TransTab pretrain (offline)
Step 2. CNN pretrained load
Step 3. projection head 추가
Step 4. contrastive learning
    - projection 항상 학습
    - encoder는 선택적으로 fine-tune

"""



def main():

    # image encoder model 
    model_image = _ 
    # backbone 교체해서 넣을 수 있도록 

    # radiomics tabular encoder model (transtab)
    model_radiomics = _ 
    # builder 방식 

    # train_dataset, val_dataset 
    # train_collate_fn, val_collate_fn 
    # train_loader, val_loader

    # all losses, contrastive losses, classification losses
    # metric scoring definition 
    # def simclr_nt_xent_loss_multi_pos 
    # def compute_multimodal_contrastive_loss_singleSimCLR 

    # train setting (epoch, progress_bar, resume_checkpoint, ..)
    # - def train_epoch으로 한 에폭 관리 
    # - def val_epoch으로 한 에폭 관리 
    # epoch loop (한 에폭에서 train/val)

    # log 찍기 
