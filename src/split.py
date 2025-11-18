import os
import shutil
import random


# Raw dir and destination dir paths
RAW_DIR = "raw-img"
OUT_DIR = "data"

# Train / val / test split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Dictionary for mapping names
TRANSLATE = {"cane": "dog", 
             "cavallo": "horse", 
             "elefante": "elephant", 
             "farfalla": "butterfly", 
             "gallina": "chicken", 
             "gatto": "cat",
             "mucca": "cow", 
             "pecora": "sheep", 
             "ragno": "spider", 
             "scoiattolo": "squirrel"}

# Random seed
random.seed(2137)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # Create directory structure (data --> train/val/test)
    for split in ["train", "val", "test"]:
        for en_name in TRANSLATE.values():
            ensure_dir(os.path.join(OUT_DIR, split, en_name))

    # Iterate over animal directories (italian)
    for it_name in os.listdir(RAW_DIR):
        it_path = os.path.join(RAW_DIR, it_name)
        if not os.path.isdir(it_path):
            continue

        # Case, when directory name not specified in `translate` dictionary
        if it_name not in TRANSLATE:
            print(f"Directory not specified. Skip.")
            continue

        en_name = TRANSLATE[it_name]
        print(f"Processing {it_name} --> {en_name}")

        # Shuffle images randomly
        images = os.listdir(it_path)
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        # The rest of images are for testing

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Function copying files
        def copy_images(img_list: list[str], split_name: str) -> None:
            for img in img_list:
                src = os.path.join(it_path, img)
                dst = os.path.join(OUT_DIR, split_name, en_name, img)
                shutil.copy(src, dst)

        copy_images(train_imgs, "train")
        copy_images(val_imgs, "val")
        copy_images(test_imgs, "test")

    print("Splitting dataset is done!")



if __name__ == "__main__":
    main()
    shutil.rmtree(RAW_DIR)
