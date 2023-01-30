import os

dir_mask = "./masks_gif"
dir_check_train = "./imgs/train"
dir_check_test = "./imgs/val"
throw_dir = "./bin"

train_lst = [f.split(".")[0] for f in os.listdir(dir_check_train)]
val_lst = [f.split(".")[0] for f in os.listdir(dir_check_test)]
mask_lst = [f.split("_")[0] for f in os.listdir(dir_mask)]

for f in os.listdir(dir_mask):
    idx = f.split("_")[0]
    if (idx in train_lst or idx in val_lst):
        print(idx)
        # os.rename(os.path.join(dir_mask,f),os.path.join(throw_dir,f))
    else:
        os.rename(os.path.join(dir_mask,f),os.path.join(throw_dir,f))

for f in os.listdir(dir_check_train):
    idx = f.split(".")[0]
    if (idx in mask_lst):
        print(idx)
    else:
        os.rename(os.path.join(dir_check_train,f),os.path.join(throw_dir,f))

for f in os.listdir(dir_check_test):
    idx = f.split(".")[0]
    if (idx in mask_lst):
        print(idx)
    else:
        os.rename(os.path.join(dir_check_test,f),os.path.join(throw_dir,f))

