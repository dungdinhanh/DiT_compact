import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil

def get_png_files(sample_dir):
    list_files = []
    max = 0
    for file in os.listdir(sample_dir):
        if file.endswith(".png"):
            list_files.append(file)
            num_image = int(file.split(".")[0])
            if num_image > max:
                max = num_image
    return list_files, max+1


def create_npz_from_sample_folder(sample_dir, num=50_000, image_size=256, raw_images=False):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    if os.path.isfile(npz_path):
        print(f"Completed sampling _ file found at {npz_path}")
        return npz_path

    images_dir = os.path.join(sample_dir, "images")
    list_png_files, _ = get_png_files(images_dir)
    no_png_files = len(list_png_files)
    os.makedirs(reference_dir, exist_ok=True)
    assert no_png_files >= num, print("not enough images, generate more")
    print("Building .npz file from samples")
    for i in range(num):
        image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            continue
        sample_pil = Image.open(os.path.join(images_dir, f"{list_png_files[i]}"))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape[1] == image_size, "what the heck?"
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = os.path.join(reference_dir, f"samples_{num}x{samples.shape[1]}x{samples.shape[2]}x3.npz")

    np.savez(npz_path, arr_0=samples)
    if not raw_images:
        shutil.rmtree(images_dir)
        print("Removed all raw images")
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")


    return npz_path

def remove_prev_npz(sample_dir, num=50_000, image_size=256):
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    if os.path.isfile(npz_path):
        print(f"Removing {npz_path}")
        os.remove(npz_path)

def get_output_name(sample_dir, num=50000, image_size=256):
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    return npz_path
