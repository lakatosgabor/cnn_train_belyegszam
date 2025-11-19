import os
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np
import cv2

# Forrás- és célmappa
DATA_DIR = "small_data"
TARGET_COUNT = 2400  # minden osztályban ennyi kép lesz
OUTPUT_DIR = "SMALL_balanced_dataset_" + str(TARGET_COUNT)

# Augmentációs függvény
def augment_image(img):
    w, h = img.size

    # Kitakarás (random alak)
    if random.random() > 0.35:
        shape_type = random.choice(["rectangle", "circle", "noise"])
        rect_w = random.randint(int(w * 0.25), int(w * 0.35))
        rect_h = random.randint(int(h * 0.25), int(h * 0.35))
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)

        # Áttetszőség
        alpha = random.randint(90, 140)

        if shape_type == "rectangle":
            color = tuple(random.randint(0, 255) for _ in range(3)) + (alpha,)
            temp = Image.new("RGBA", (rect_w, rect_h), color)
            temp = temp.filter(ImageFilter.GaussianBlur(1))
            img.paste(temp, (x1, y1), temp)

        elif shape_type == "circle":
            temp = Image.new("RGBA", (rect_w, rect_h), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp)
            color = tuple(random.randint(0, 255) for _ in range(3)) + (alpha,)
            temp_draw.ellipse([0, 0, rect_w, rect_h], fill=color)
            temp = temp.filter(ImageFilter.GaussianBlur(1))
            img.paste(temp, (x1, y1), temp)

        elif shape_type == "noise":
            noise = np.random.randint(0, 256, (rect_h, rect_w, 3), dtype=np.uint8)
            noise_img = Image.fromarray(noise, "RGB").filter(ImageFilter.GaussianBlur(1))
            mask = Image.new("L", (rect_w, rect_h), alpha)
            img.paste(noise_img, (x1, y1), mask)

    # Zoom
    zoom_factor = random.uniform(0.8, 1.2)
    if zoom_factor < 1.0:
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        img = img.resize((new_w, new_h))
        new_img = Image.new("RGB", (w, h))
        offset = ((w - new_w) // 2, (h - new_h) // 2)
        new_img.paste(img, offset)
        img = new_img
    else:
        crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize((w, h))

    background_color = (52, 36, 47)
    # Eltolás
    max_shift = 0.15
    shift_x = int(w * random.uniform(-max_shift, max_shift))
    shift_y = int(h * random.uniform(-max_shift, max_shift))
    shifted = Image.new("RGB", (w, h), background_color)
    shifted.paste(img, (shift_x, shift_y))
    img = shifted

    # Forgatás
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, expand=False, fillcolor=background_color)

    return img


# Osztályok feldolgozása
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
os.makedirs(OUTPUT_DIR, exist_ok=True)

for cls in classes:
    input_path = os.path.join(DATA_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(input_path) if f.lower().endswith(".png")]
    current_count = len(images)
    print(f"\n[{cls}] {current_count} kép → cél: {TARGET_COUNT}")

    # Eredeti képek másolása
    for img_name in images:
        src = os.path.join(input_path, img_name)
        dst = os.path.join(output_path, img_name)
        Image.open(src).save(dst)

    # Ha kevesebb, mint TARGET_COUNT → pótlás augmentációval
    if current_count < TARGET_COUNT:
        needed = TARGET_COUNT - current_count
        for i in tqdm(range(needed), desc=f"Augmentálás {cls} (hiány pótlása)"):
            base_name = random.choice(images)
            base_img = Image.open(os.path.join(input_path, base_name))
            aug_img = augment_image(base_img)
            aug_img.save(os.path.join(output_path, f"aug_{i}.png"))

    # TARGET_COUNT kép biztosítása (ha több, véletlenszerűen csökkentjük)
    all_imgs = os.listdir(output_path)
    if len(all_imgs) > TARGET_COUNT:
        all_imgs = random.sample(all_imgs, TARGET_COUNT)
        # Töröljük a többit
        for f in os.listdir(output_path):
            if f not in all_imgs:
                os.remove(os.path.join(output_path, f))

    # Most már minden osztálynak pontosan TARGET_COUNT képe van
    all_imgs = os.listdir(output_path)
    aug_count = int(TARGET_COUNT * 0.2)  # 20% augmentálás
    aug_subset = random.sample(all_imgs, aug_count)
    delete_subset = random.sample(all_imgs, aug_count)
    for name in tqdm(delete_subset, desc=f"{cls} - 20% törlése"):
        file_path = os.path.join(output_path, name)
        if os.path.exists(file_path):
            os.remove(file_path)

    remaining_imgs = os.listdir(output_path)
    aug_subset = random.sample(remaining_imgs, aug_count)
    for i, name in enumerate(tqdm(aug_subset, desc=f"Augmentálás {cls} (20%)")):
        img_path = os.path.join(output_path, name)
        base_img = Image.open(img_path)
        aug_img = augment_image(base_img)
        aug_img.save(os.path.join(output_path, f"aug_final_{i}_{name}"))
