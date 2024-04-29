#!/usr/bin/env python
# coding: utf-8

import torch
print(torch.__version__)
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import requests
from PIL import Image
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import gc
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from transformers import BitsAndBytesConfig

model_id = "google/matcha-base"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

DATASET_DIR = "/home/ritaban/smart/SMART101-release-v1/SMART101-Data/"
device = 2

def read_csv(csvfilename, puzzle_id):
    import csv
    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
    return qa_info

SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
SIGNS = np.array(["+", "-", "x", "/"])
MAX_DECODE_STEPS = 10

def get_puzzle_class_info(puzzle_ids, icon_class_ids):
    #    global SEQ_PUZZLES, puzzle_diff_str, puzzle_diff
    puzzle_classes = {}
    for puzzle_id in puzzle_ids:
        puzzle_root = puzzle_id
        csv_file = "puzzle_%s.csv" % (puzzle_id)
        qa_info = read_csv(os.path.join(DATASET_DIR, puzzle_root, csv_file), puzzle_id)

        pid = int(puzzle_id)
        if pid not in SEQ_PUZZLES:
            num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids) for qa in qa_info]).max() + 1
        else:
            if pid in [16, 39, 100]:
                num_classes = 26 + 1  # if the output is a string of numbers, and the max classes is - max val.
            elif pid in [18, 35]:
                num_classes = 5 + 1  # the minus one is for end of items.
            elif pid in [63]:
                num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids).max() for qa in qa_info]).max() + 1
        puzzle_classes[str(puzzle_id)] = num_classes
    return puzzle_classes

def get_icon_dataset_classes(icon_path):
    """returns the classes in ICONs-50 dataset"""
    with open(icon_path, "r") as f:
        icon_classes = f.readlines()
    return [ii.rstrip() for ii in icon_classes]

def str_replace(ans):
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    return ans

def pad_with_max_val(gt_list, val):
    """if the number of elements in gt is less than MAX_DECODE_STEPS, we pad it with the max value in a class"""
    if len(gt_list) < MAX_DECODE_STEPS:
        gt_list = (
            gt_list
            + (
                np.ones(
                    MAX_DECODE_STEPS - len(gt_list),
                )
                * val
            ).tolist()
        )
    return gt_list

def get_val(qinfo, ans_opt, num_classes_per_puzzle, icon_class_ids, is_one_of_option=False):
    """get the value of the answer option. This code also encodes the value into a number by removing extreneous strings"""
    """ is_one_of_option is True, when ans_opt is one of the options, need not be the correct answer option."""
    where = lambda x, y: np.where(np.array(x) == y)[0][0]
    pid = int(qinfo["puzzle_id"])
    if pid in SEQ_PUZZLES:
        ans = qinfo[ans_opt]
        if pid == 16:
            ans_opt_val = [int(ii) for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 18:
            ans_opt_val = [int(ii) for ii in ans.split("-")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 35:
            ans_opt_val = [
                ord(ii) - ord("A") for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")
            ]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 39:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 63:
            ans_opt_val = [
                int(ii)
                for ii in ans.replace("and", ",")
                .replace("or", ",")
                .replace(", ,", ",")
                .replace("only", "")
                .replace(" ", "")
                .split(",")
            ]
            key = str(63)
            if key in num_classes_per_puzzle:
                ans_opt_val = pad_with_max_val(ans_opt_val, num_classes_per_puzzle[key] - 1)
        elif pid == 100:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        ans_opt_val = np.array(ans_opt_val)

    elif pid == 58:
        # puzzle 58 has answers as <operator><one digit number>, e.g./4,-5, etc.
        # we use +=1, -=2, x=3, /=4. so /4 will be 44, -5=25, +2= 2.
        ans_opt_val = qinfo[ans_opt]
        ans_opt_val = (where(SIGNS, ans_opt_val[0]) + 1) * 10 + int(ans_opt_val[1:])
    elif pid == 25:
        # we need to fix the time in AM/PM format properly.
        ans = qinfo[ans_opt]
        ans_opt_val = int(ans.replace(":00 AM", "").replace(":00 PM", ""))
        if ans.find("PM") > -1:
            ans_opt_val += 12
    else:
        try:
            ans_opt_val = int(qinfo[ans_opt])
        except:
            if len(qinfo[ans_opt]) > 0:
                try:
                    ans_opt_val = ord(qinfo[ans_opt]) - ord("A")
                except:
                    try:
                        ans_opt_val = str_replace(qinfo[ans_opt])
                        ans_opt_val = ans_opt_val.replace("Impossible", "0")  # puzzle 58.
                        if int(qinfo["puzzle_id"]) == 1:  # if the puzzle id is 1, then the options are icon classes.
                            ans_opt_val = "_".join(ans_opt_val.split(" "))
                            if ans_opt_val in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val)
                            elif ans_opt_val + "s" in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val + "s")
                        ans_opt_val = int(ans_opt_val)
                    except:
                        print(qinfo)
                        pdb.set_trace()
            else:
                ans_opt_val = ord(ans_opt) - ord("A")
    if not is_one_of_option:  # implies we are encoding the correct answer.
        qinfo["AnswerValue"] = ans_opt_val
    return ans_opt_val


# In[10]:


def split_data(info, split):
    """
    split_type=standard is to use the split_ratio in the instance order
    split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
    split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
    """
    split_ratio = "80:5:15"
    splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
    n = len(info)
    if split == "train":
        st = 0
        en = int(np.floor(n * splits[0] / 100.0))
        info = info[st:en]
    elif split == "val":
        st = int(np.ceil(n * splits[0] / 100.0))
        en = int(np.floor(n * splits[1] / 100.0))
        info = info[st:en]
    else:
        st = int(np.ceil(n * splits[1] / 100.0))
        info = info[st:]
    return info


# In[11]:


import random
random.seed(1007)
from torch.utils.data import Dataset, DataLoader

PS_VAL_IDX = [7, 43, 64]
PS_TEST_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]
tokenizer = AutoTokenizer.from_pretrained(model_id)
MAX_PATCHES = 1024

def str_replace_(info, ans_opt):
    ans = info[ans_opt]
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    ans = ans.replace("Impossible", "0")
    info[ans_opt] = ans
    return ans

class SMARTData(Dataset):
    def __init__(self, split):
        super(SMARTData, self).__init__()
        MAX_VAL = 0
        self.qa_info = []
        self.icon_class_ids = get_icon_dataset_classes(DATASET_DIR + "icon-classes.txt")

        if split == "train":
            puzzle_ids = os.listdir(DATASET_DIR)
            puzzle_ids = np.array(puzzle_ids)[np.array([x.find(".") == -1 for x in puzzle_ids])]
            puzzle_ids = puzzle_ids.tolist()
            val_test = PS_VAL_IDX + PS_TEST_IDX
            val_test = set([str(ii) for ii in val_test])
            puzzle_ids = list(set(puzzle_ids).difference(val_test))
        elif split == "val":
            puzzle_ids = [str(ii) for ii in PS_VAL_IDX]
        else:
            puzzle_ids = [str(ii) for ii in PS_TEST_IDX]

        self.split = split
        self.num_classes_per_puzzle = get_puzzle_class_info(puzzle_ids, self.icon_class_ids)
        print("number of train puzzles = %d" % (len(puzzle_ids)))
        for puzzle_id in puzzle_ids:
          csv_file = "puzzle_%s.csv" % (puzzle_id)
          tqa_info = read_csv(os.path.join(DATASET_DIR, puzzle_id, csv_file), puzzle_id)
          for t in range(len(tqa_info)):
              tqa_info[t]["AnswerValue"] = get_val(tqa_info[t], tqa_info[t]["Answer"], self.num_classes_per_puzzle, self.icon_class_ids)
          self.qa_info += split_data(tqa_info, split)
        print(len(self.qa_info))

    def __len__(self):
        return len(self.qa_info)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/"
        im = Image.open(os.path.join(DATASET_DIR, puzzle_root, "img", info["image"]))
        im = im.resize((300, 300))
        qa = info["Question"]
        _ = [str_replace_(info, key) for key in ["A", "B", "C", "D", "E"]]
        opt_vals = [get_val(info, key, self.num_classes_per_puzzle, self.icon_class_ids, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
        lbl = info["Answer"]
        answer_value = info["AnswerValue"]
        answer = np.zeros(MAX_DECODE_STEPS,)
        if int(pid) not in SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value

        opts = ["A", "B", "C", "D", "E"]
        prompt = "Answer with the option’s letter from the given choices directly. Q: {} Options: {}".format(qa, " ".join(["({}) {}".format(opt, str(val)) for opt, val in zip(opts, opt_vals)]))
        processed_inputs = processor(images=im, text=prompt, add_special_tokens=True, return_tensors="pt", max_patches=MAX_PATCHES)
        processed_inputs["text"] = lbl
        processed_inputs["answers"] = answer_value
        processed_inputs["im"] = info["image"]
        processed_inputs["qa"] = qa
        processed_inputs["puzzle_id"] = info["puzzle_id"]
        return processed_inputs


# In[12]:


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["text"] for item in batch]
  answers = [item["answers"] for item in batch]
  im = [item["im"] for item in batch]
  qa = [item["qa"] for item in batch]
  puzzle_id = [item["puzzle_id"] for item in batch]

  text_inputs = tokenizer(texts, return_tensors="pt")

  new_batch["labels"] = text_inputs["input_ids"]

  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"][0])
    new_batch["attention_mask"].append(item["attention_mask"][0])

  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
  new_batch["lbls"] = texts
  new_batch["answers"] = answers
  new_batch["im"] = im
  new_batch["qa"] = qa
  new_batch["puzzle_id"] = puzzle_id

  return new_batch


# In[13]:
test_data = SMARTData("test")
test_loader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)

print(torch.cuda.device_count())
for j in range(1, 6):
    torch.cuda.empty_cache()
    print(f"Loading checkpoint: {j}")
    checkpoint = torch.load(f"ckpts_b/matcha_algebra_noaug_{j}.ckpt", map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint)
    dp_model = model.to(device)
    model.eval()
    test_loss = 0
    test_acc = 0
    test_dicts = []
    for _, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            labels = batch.pop("labels")
            answers = batch.pop("answers")
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            outputs = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=50)
        generated_answers = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = processor.batch_decode(labels, skip_special_tokens=True)
        for i in range(len(generated_answers)):
            test_dict = dict()
            test_dict["im"] = batch["im"][i]
            test_dict["qa"] = batch["qa"][i]
            test_dict["puzzle_id"] = batch["puzzle_id"][i]
            test_dict["lbl"] = batch["lbls"][i]
            test_dict["answer_value"] = answers[i]
            test_dict["answer_matcha"] = generated_answers[i]
            if generated_answers[i] == labels[i]:
                test_acc += 1
            elif generated_answers[i] == str(answers[i]):
                test_acc += 1
            elif generated_answers[i] == batch["lbls"][i]:
                test_acc += 1
            test_dicts.append(test_dict)
    test_acc /= len(test_data)
    print(f"Accuracy for checkpoint {j}: {test_acc}")
    with open("ckpts_b/test_data_matcha_algebra_noaug.json", "w+") as openfile:
        json.dump(test_dicts, openfile, indent=2)
