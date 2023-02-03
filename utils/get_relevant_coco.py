import json 
import os 
ROOT = "../datasets/coco/"
DEST = "datasets/coco/train2014"
os.makedirs(DEST, exist_ok=True)
splits = ["refcoco","refcoco+","refcocog"]

total_set = set()
for split in splits:
    dat = json.load(open(os.path.join(ROOT, split, "instances.json"),"r"))
    cur = set([x["file_name"] for x in dat["images"]])
    # print(len(set(dat["images"][])))
    total_set = total_set.union(cur)
    # break
print(len(total_set))

from tqdm import tqdm 
import shutil 
for image in tqdm(total_set, total = len(total_set)):
    src = os.path.join(ROOT, "train2014", image)
    dst = os.path.join(DEST, image)
    shutil.copy(src, dst)