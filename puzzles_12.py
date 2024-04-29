from PIL import Image, ImageDraw, ImageFont
import csv
import os
import random

DATASET_DIR = "/home/ritaban/smart/SMART101-release-v1/SMART101-Data/"
di = "12-1"
fields = ["id", "Question", "image", "A", "B", "C", "D", "E", "Answer", "Note"]
dirs = ["monkey", "feline", "bird"]

from random import shuffle, seed
from faker.providers.person.en import Provider

first_names = list(set(Provider.first_names))

seed(4321)
shuffle(first_names)
with open(f"{DATASET_DIR}/{di}/puzzle_{di}.csv", "w+") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for idx in range(1, 1601):
        # Create a new image with a white background
        d = dict()
        d["id"] = idx
        d["image"] = f'puzzle_{di}_{idx}.png'
        d["Note"] = ""
        
        class_name = random.randint(0, 2)
        num_icons = random.randint(10, 30)
        pos1 = random.randint(1, num_icons)
        if pos1 == 1:
            position1 = str(pos1) + "st"
        elif pos1 == 2:
            position1 = str(pos1) + "nd"
        elif pos1 == 3:
            position1 = str(pos1) + "rd"
        else:
            position1 = str(pos1) + "th"
        pos2 = random.randint(1, num_icons)
        while pos1 == pos2:
            pos2 = random.randint(1, num_icons)
        if pos2 == 1:
            position2 = str(pos2) + "st"
        elif pos2 == 2:
            position2 = str(pos2) + "nd"
        elif pos2 == 3:
            position2 = str(pos2) + "rd"
        else:
            position2 = str(pos2) + "th"
        d["Question"] = f"There are {num_icons} {dirs[class_name]}s in a line, each with a unique name. {first_names[idx*2]} is the {position1} from the front and {first_names[idx*2 + 1]} is the {position2} from the front. How many {dirs[class_name]}s are there between {first_names[idx*2]} and {first_names[idx*2 + 1]} in the line?"
        ans = abs(pos2 - pos1) - 1
        images = []
        total_width = 0
        max_height = 0
        icon_dir = os.path.join("icons50/Icons-50/Icons-50", dirs[class_name])
        icons = os.listdir(icon_dir)
        for _ in range(num_icons):
            icon_path = os.path.join(icon_dir, random.choice(icons))
            icon_image = Image.open(icon_path)
            images.append(icon_image)
            total_width += icon_image.width
            max_height = max(max_height, icon_image.height)

        opt = random.randint(0, 4)
        opts = ["A", "B", "C", "D", "E"]
        for i in range(5):
            if i == opt:
                d[opts[opt]] = ans
            else:
                d[opts[i]] = random.randint(1, num_icons)

        d["Answer"] = opts[opt]

        width, height = 600, 600
        image = Image.new('RGB', (width, height), color='white')
        # Paste images on the canvas
        x_offset = (width - num_icons * 20) // 2
        for img in images:
            img = img.resize((20, 20))
            image.paste(img, (x_offset, int(height/2)))
            x_offset += img.width

        # Save the image
        image.save(f"{DATASET_DIR}/{di}/img/puzzle_{di}_{idx}.png")
        writer.writerow(d)