from PIL import Image, ImageDraw, ImageFont
import csv
import random

DATASET_DIR = "/home/ritaban/smart/SMART101-release-v1/SMART101-Data/"
di = "10-2"
fields = ["id", "Question", "image", "A", "B", "C", "D", "E", "Answer", "Note"]

with open(f"{DATASET_DIR}/{di}/puzzle_{di}.csv", "w+") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for idx in range(1, 1601):
        # Create a new image with a white background
        d = dict()
        d["id"] = idx
        d["Question"] = "The correct additions in the squares were performed according to the pattern shown in the table. What number is covered by the question mark?"
        d["image"] = f'puzzle_{di}_{idx}.png'
        d["Note"] = ""
        width, height = 200, 200
        image = Image.new('RGB', (width, height), color='white')

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Define the font and size
        font = ImageFont.load_default(size=20)

        # Draw the grid
        grid_width, grid_height = 120, 120
        grid_x, grid_y = (width - grid_width) // 2, (height - grid_height) // 2
        draw.rectangle([(grid_x, grid_y), (grid_x + grid_width, grid_y + grid_height)], outline='black', width=2)
        draw.line([(grid_x + grid_width // 3, grid_y), (grid_x + grid_width // 3, grid_y + grid_height)], fill='black', width=2)
        draw.line([(grid_x + 2 * grid_width // 3, grid_y), (grid_x + 2 * grid_width // 3, grid_y + grid_height)], fill='black', width=2)
        draw.line([(grid_x, grid_y + grid_height // 3), (grid_x + grid_width, grid_y + grid_height // 3)], fill='black', width=2)
        draw.line([(grid_x, grid_y + 2 * grid_height // 3), (grid_x + grid_width, grid_y + 2 * grid_height // 3)], fill='black', width=2)

        random_numbers = [random.randint(1, 20) for _ in range(4)]
        
        numbers = ['+', random_numbers[0], random_numbers[1], random_numbers[2], \
            random_numbers[0] + random_numbers[2], random_numbers[1] + random_numbers[2], \
            random_numbers[3], random_numbers[0] + random_numbers[3], random_numbers[1] + random_numbers[3]]
        
        opt = random.randint(0, 4)
        opts = ["A", "B", "C", "D", "E"]
        if idx % 4 == 0:
            ans = numbers[1]
            numbers[1] = '?'
        elif idx % 4 == 1:
            ans = numbers[2]
            numbers[2] = '?'
        elif idx % 4 == 2:
            ans = numbers[3]
            numbers[3] = '?'
        else:
            ans = numbers[6]
            numbers[6] = '?'
        
        for i in range(5):
            if i == opt:
                d[opts[opt]] = ans
            else:
                d[opts[i]] = random.randint(1, 20)

        d["Answer"] = opts[opt]

        for i, value in enumerate(numbers):
            x = grid_x + (i % 3) * grid_width // 3 + 10
            y = grid_y + (i // 3) * grid_height // 3 + 10
            draw.text((x, y), str(value), font=font, fill='black')

        # Save the image
        image.save(f"{DATASET_DIR}/{di}/img/puzzle_{di}_{idx}.png")
        writer.writerow(d)