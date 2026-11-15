Image Classification Using a Pretrained Model
"""
Image Classification Using a Pretrained Model
=============================================
This program uses MobileNetV2, a small pretrained model from PyTorch
(torchvision), to classify a few sample images and print the predicted labels.
What "pretrained" means (in simple words)----------------------------------------
MobileNetV2 was already trained by researchers on ImageNet, a huge collection
of about 1.2 million photos belonging to 1000 categories (dog breeds, cars,
fruits, tools, etc.). During that training it learned what things look like.
We do NOT train anything ourselves. We simply download the finished model and
ask it: "What is in this picture?"
How the model makes a prediction-------------------------------
1. The image is resized and normalised so it matches what the model saw in
   training (preprocessing).
2. The image passes through many layers. Early layers notice simple things
   like edges and colours, middle layers notice textures and shapes, and the
   last layers recognise whole objects such as "a dog's face".
3. The model outputs 1000 scores, one per category.
4. The softmax function turns the scores into percentages that add up to 100%.
5. We show the categories with the highest percentages (the top 3).
Run:  python image_classifier.py
"""
import os
import urllib.request
import torch
from PIL import Image
from torchvision import models
# -----------------------------------------------------------------
# Settings (easy to change)
# -----------------------------------------------------------------
IMAGE_FOLDER = "sample_images"   # put your own .jpg / .png images here too
TOP_K = 3                        # how many predictions to show per image
# A tiny "dataset" of 4 sample images, downloaded once if they are missing.
SAMPLE_IMAGES = {
    "dog.jpg": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "labrador.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLab\
    radorLooking_new.jpg",
    "cat.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_ca\
    tus-cat_on_snow.jpg",
    "sunflower.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Re\
    d_sunflower.jpg",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# -----------------------------------------------------------------
# Step 1: Get the sample images
# -----------------------------------------------------------------
def download_samples():
    """Download the sample images into IMAGE_FOLDER (only if not present)."""
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    for filename, url in SAMPLE_IMAGES.items():
        path = os.path.join(IMAGE_FOLDER, filename)
        if os.path.exists(path):
            continue
        try:
            print(f"Downloading {filename} ...")
            # Some sites reject requests without a browser-like header.
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=20) as response, \
                    open(path, "wb") as f:
                f.write(response.read())
        except Exception as error:
            print(f"  Could not download {filename}: {error}")
def list_images():
    """Return the paths of all images found in IMAGE_FOLDER."""
    if not os.path.isdir(IMAGE_FOLDER):
        return []
    files = sorted(os.listdir(IMAGE_FOLDER))
    return [os.path.join(IMAGE_FOLDER, f) for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)]
# -----------------------------------------------------------------
# Step 2: Load the pretrained model
# -----------------------------------------------------------------
def load_model():
    """Load MobileNetV2 with ImageNet weights (downloaded on the first run)."""
    weights = models.MobileNet_V2_Weights.DEFAULT   # the pretrained knowledge
    model = models.mobilenet_v2(weights=weights)
    model.eval()                        # evaluation mode: we only predict, not train
    preprocess = weights.transforms()   # the resize/normalise steps the model expects
    labels = weights.meta["categories"] # the 1000 category names
    return model, preprocess, labels
# -----------------------------------------------------------------
# Step 3: Classify one image
# -----------------------------------------------------------------
def classify_image(path, model, preprocess, labels, top_k=TOP_K):
    """Return a list of (label, percentage) for the top_k predictions."""
    image = Image.open(path).convert("RGB")      # open the picture
    batch = preprocess(image).unsqueeze(0)       # prepare it; add a "batch" dimension
    with torch.no_grad():                        # no training, so save memory
        scores = model(batch)[0]                 # 1000 raw scores
    probabilities = torch.nn.functional.softmax(scores, dim=0)  # scores -> percentages
    top_probs, top_ids = torch.topk(probabilities, top_k)       # best guesses
    return [(labels[i], p.item() * 100) for p, i in zip(top_probs, top_ids)]
# -----------------------------------------------------------------
# Step 4: Show the results
# -----------------------------------------------------------------
def print_results(all_results):
    """Print the predictions for every image."""
    print("\n" + "=" * 50)
    print("  PREDICTIONS")
    print("=" * 50)
    for path, predictions in all_results:
        print(f"\nImage: {os.path.basename(path)}")
        for rank, (label, percent) in enumerate(predictions, start=1):
            print(f"  {rank}. {label:<25} {percent:5.1f}%")
def show_plot(all_results):
    """Show all images with their best prediction as the title (optional)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib is not installed, so the picture grid is skipped.)")
        return
    count = len(all_results)
    columns = min(count, 4)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = axes.flatten() if count > 1 else [axes]
    for ax, (path, predictions) in zip(axes, all_results):
        label, percent = predictions[0]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(f"{label}\n{percent:.1f}%", fontsize=11)
        ax.axis("off")
    for ax in axes[count:]:          # hide any unused boxes
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("results.png", dpi=120)
    print("\nPicture grid saved as results.png")
    plt.show()
# -----------------------------------------------------------------
# Main program
# -----------------------------------------------------------------
def main():
    download_samples()
    image_paths = list_images()
    if not image_paths:
        print(f"No images found. Add .jpg or .png files to the '{IMAGE_FOLDER}' folder.")
        return
    print("\nLoading the pretrained model (first run downloads about 14 MB)...")
    model, preprocess, labels = load_model()
    all_results = []
    for path in image_paths:
        predictions = classify_image(path, model, preprocess, labels)
        all_results.append((path, predictions))
    print_results(all_results)
    show_plot(all_results)
if __name__ == "__main__":
    main()