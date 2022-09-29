# Configuration for impress project
from pathlib import Path

root = Path.home() / "1_modules/4_MultiLabel"

raw_data_file = root / "data/kenya_impress.csv"
data_file = root / "data/kenya_impress_lbls.csv"
# Splitted has randomly splited in train and test, with group column
data_file = root / "data/img_lbls_splitted.csv"

train_imgs_path = root / "train-impress"
label_path = root / "labels"

out_history = root / "output/history"
out_model_path = root / "output/model"
out_checkpoint = root / "output/checkpoint"
out_prediction = root / "output/prediction"

out_checkpoint.mkdir(exist_ok=True, parents=True)
out_model_path.mkdir(exist_ok=True, parents=True)
out_history.mkdir(exist_ok=True, parents=True)
out_prediction.mkdir(exist_ok=True, parents=True)


# Prediction variables

imgs_to_predict = root.parent / "3_WADL/notebooks/train-jpg-kenya/src"
width = 224
height = 224
stride = 122
