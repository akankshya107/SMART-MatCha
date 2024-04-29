from PIL import Image, ImageDraw, ImageFont
import csv
import os
import random

DATASET_DIR = "/home/ritaban/smart/SMART101-release-v1/SMART101-Data/"
di = "28-1"
fields = ["id", "Question", "image", "A", "B", "C", "D", "E", "Answer", "Note"]
dirs = os.listdir("icons50/Icons-50/Icons-50")

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
        
        class_name = random.randint(0, len(dirs)-1)
        d["Question"] = "The question mark should be replaced by:"
        images = []
        total_width = 0
        max_height = 0
        icon_dir = os.path.join("icons50/Icons-50/Icons-50", dirs[class_name])
        icons = os.listdir(icon_dir)
        for _ in range(2):
            icon_path = os.path.join(icon_dir, random.choice(icons))
            icon_image = Image.open(icon_path)
            images.append(icon_image)
            total_width += icon_image.width
            max_height = max(max_height, icon_image.height)

        item1 = random.randint(1, 10)
        item2 = random.randint(1, 10)
        opt = random.randint(0, 4)
        opts = ["A", "B", "C", "D", "E"]
        for i in range(5):
            if i == opt:
                d[opts[opt]] = item2
            else:
                d[opts[i]] = random.randint(1, 20)

        d["Answer"] = opts[opt]
        texts = [f"= {item1 + item2}", f"= {item1}", "= ?"]

        width, height = 200, 200
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)
        # Paste images on the canvas
        x_offset = (width - 50) // 2
        y_offset = height // 3
        
        images[0] = images[0].resize((20, 20))
        image.paste(images[0], (x_offset, y_offset))
        draw.text((x_offset + images[0].width, y_offset), "+", font=font, fill='black')
        images[1] = images[1].resize((20, 20))
        image.paste(images[1], (x_offset + 2 * images[0].width, y_offset))
        draw.text((x_offset + 3 * images[0].width, y_offset), texts[0], font=font, fill='black')

        y_offset += images[0].height
        for e, img in enumerate(images):
            img = img.resize((20, 20))
            image.paste(img, (x_offset, y_offset))
            draw.text((x_offset + img.width, y_offset), texts[e+1], font=font, fill='black')
            y_offset += img.height

        # Save the image
        image.save(f"{DATASET_DIR}/{di}/img/puzzle_{di}_{idx}.png")
        writer.writerow(d)