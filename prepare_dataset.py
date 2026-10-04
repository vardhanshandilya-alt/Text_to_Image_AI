import os
import csv
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS_FILE = os.path.join(BASE_DIR, "data", "caption.txt")
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
OUT_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images_prepared")
OUT_CSV_FILE = os.path.join(BASE_DIR, "data", "captions_prepared.txt")

IMG_SIZE = 512  # keep 512 for Stable Diffusion

def load_captions():
    captions = []
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: image_name.jpg|caption
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue

            img_name = parts[0].strip()
            caption = parts[1].strip()
            captions.append((img_name, caption))

    return captions


def resize_and_save(img_path, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(out_path, format="JPEG", quality=95)


def main():
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)

    captions = load_captions()
    print("Total captions found:", len(captions))

    rows = []
    count = 0

    for img_name, caption in tqdm(captions):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)

        if not os.path.exists(raw_img_path):
            continue

        new_name = f"{count:06d}.jpg"
        out_img_path = os.path.join(OUT_IMAGES_DIR, new_name)

        try:
            resize_and_save(raw_img_path, out_img_path)
            rows.append([new_name, caption])
            count += 1
        except:
            continue

    with open(OUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        writer.writerows(rows)

    print("\nProcessed images:", count)
    print("Saved CSV:", OUT_CSV_FILE)


if __name__ == "__main__":
    main()
