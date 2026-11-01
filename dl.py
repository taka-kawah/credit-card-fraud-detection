import os
import shutil

import kagglehub

pwd = os.path.dirname(os.path.realpath(__file__))
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
new_path = shutil.move(path, pwd)
print(f"downloaded: {new_path}")

