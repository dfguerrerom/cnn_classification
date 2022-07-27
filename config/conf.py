from pathlib import Path

root = Path.home()/"1_modules/4_MultiLabel"

data_file = root/"data/img_lbls.csv"
train_imgs_path = root/"train-tif-ce"
label_path = root/"labels"

out_history = root/"output/history"
out_model_path = root/"output/model"
out_checkpoint = root/"output/checkpoint"


out_checkpoint.mkdir(exist_ok=True, parents=True)
out_model_path.mkdir(exist_ok=True, parents=True)
out_history.mkdir(exist_ok=True, parents=True)