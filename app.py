import os
import random
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

# FutureWarning kezelés (opcionális)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Secure key for flashing messages

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
MASK_FOLDER = 'static/masks/'
OVERLAY_FOLDER = 'static/overlays/'
TRAIN_FOLDER = 'static/train/'
TEST_FOLDER = 'static/test/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.config['OVERLAY_FOLDER'] = OVERLAY_FOLDER
app.config['TRAIN_FOLDER'] = TRAIN_FOLDER
app.config['TEST_FOLDER'] = TEST_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB file size limit

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)
os.makedirs('static/css/', exist_ok=True)  # CSS mappa létrehozása

# Load CSV and Create Image-Mask Mapping
csv_path = 'train_ship_segmentations_v2.csv'  # Győződj meg róla, hogy a CSV elérhető
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"A CSV fájl nem található: {csv_path}")

df_masks = pd.read_csv(csv_path)
image_mask_map = defaultdict(list)
for idx, row in df_masks.iterrows():
    image_id = row['ImageId']
    encoded_pixels = row['EncodedPixels']
    if pd.notna(encoded_pixels):
        image_mask_map[image_id].append(encoded_pixels)

# RLE Decode Function
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Pre-decode Masks and Save as Images
def decode_and_save_masks(output_dir='static/masks/', shape=(768, 768)):
    os.makedirs(output_dir, exist_ok=True)
    for image_id, mask_rles in image_mask_map.items():
        mask_path = os.path.join(output_dir, f"{image_id}_mask.png")
        if not os.path.exists(mask_path):  # Decode only if not already saved
            combined_mask = np.zeros(shape, dtype=np.uint8)
            for rle in mask_rles:
                combined_mask |= rle_decode(rle, shape=shape)
            mask_image = Image.fromarray(combined_mask * 255).convert("L")
            mask_image.save(mask_path)
    print("Maszkok előzetesen dekódolva és mentve.")

# Run Decoding (only once)
decode_and_save_masks()

# Initialize DeepLabV3 Model
# Használjuk a weights=None helyett a weights paramétert, hogy elkerüljük a figyelmeztetéseket
model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1, aux_loss=False)

# Modify the main classifier to output 1 channel (binary segmentation)
model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, 1)

# Ne módosítsd az auxiliary classifier-t

# Load the state_dict, filtering out aux_classifier parameters (mivel aux_loss=False)
model_path = 'best_deeplabv3_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"A modell fájl nem található: {model_path}")

try:
    state_dict = torch.load(model_path, map_location=device)
    # Remove auxiliary classifier keys if any (biztosra megyünk)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Modell sikeresen betöltve és inicializálva.")
except Exception as e:
    print(f"Hiba a modell betöltése során: {e}")
    exit(1)

# Allowed File Check
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image.unsqueeze(0).to(device)

# Generate Mask
def generate_mask(image_path, mask_path):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)['out']
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8) * 255  # Binarize
    mask = Image.fromarray(pred).resize(Image.open(image_path).size)
    mask.save(mask_path)
    return mask

# Create Overlay
def create_overlay(original_image_path, predicted_mask_path, overlay_path):
    # Original Image
    image = Image.open(original_image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # Correct orientation

    # Predicted Mask
    pred_mask = Image.open(predicted_mask_path).convert("L")
    pred_mask = pred_mask.resize(image.size)
    pred_mask_np = np.array(pred_mask)

    # Colored Mask (Red)
    pred_mask_color = ImageOps.colorize(pred_mask, black=(0, 0, 0), white=(255, 0, 0))

    # Create Overlay
    overlay = Image.blend(image, pred_mask_color, alpha=0.5)
    overlay.save(overlay_path)
    return overlay

# Calculate IoU
def calculate_iou(pred_mask, true_mask):
    pred = np.array(pred_mask).astype(bool)
    true = np.array(true_mask).astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return float('nan')  # Undefined IoU
    else:
        return intersection / union

# Calculate Precision, Recall, F1
def calculate_metrics(pred_mask, true_mask):
    pred = np.array(pred_mask).astype(bool)
    true = np.array(true_mask).astype(bool)

    TP = np.logical_and(pred, true).sum()
    FP = np.logical_and(pred, np.logical_not(true)).sum()
    FN = np.logical_and(np.logical_not(pred), true).sum()

    if TP + FP == 0:
        precision = float('nan')
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = float('nan')
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = float('nan')
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

# Helper Function to Process Image
def process_image(image_path, image_id=None, source_folder='uploads'):
    try:
        # Generate Predicted Mask
        predicted_mask_filename = f"predicted_{os.path.basename(image_path)}"
        predicted_mask_path = os.path.join(app.config['MASK_FOLDER'], predicted_mask_filename)
        predicted_mask = generate_mask(image_path, predicted_mask_path)
        print(f"Predikált maszk generálva: {predicted_mask_path}")

        # Create Overlay
        overlay_filename = f"overlay_{os.path.basename(image_path)}"
        overlay_path = os.path.join(app.config['OVERLAY_FOLDER'], overlay_filename)
        create_overlay(image_path, predicted_mask_path, overlay_path)
        print(f"Overlay generálva: {overlay_path}")

        # Initialize variables
        iou_score = "N/A"
        precision = "N/A"
        recall = "N/A"
        f1 = "N/A"
        has_mask = False
        original_image = None
        original_mask_filename = None
        ship_detected = False  # Új flag a hajó detektálásához

        # Check if any ship is detected
        pred_mask_np = np.array(predicted_mask)
        ship_detected = np.any(pred_mask_np > 0)
        print(f"Ship detected: {ship_detected}")

        if image_id and image_id in image_mask_map:
            # Get Original Mask
            original_mask_filename = f"{image_id}_mask.png"
            original_mask_path = os.path.join(app.config['MASK_FOLDER'], original_mask_filename)
            if os.path.exists(original_mask_path):
                has_mask = True
                # Determine the correct original_image path based on source_folder
                # Ensuring forward slashes for URLs
                original_image = f"{source_folder}/{os.path.basename(image_path)}"
                # Calculate IoU
                true_mask = Image.open(original_mask_path).convert("L").resize(predicted_mask.size)
                iou_score = calculate_iou(predicted_mask, true_mask)

                # Calculate Metrics
                precision, recall, f1 = calculate_metrics(predicted_mask, true_mask)

                # Format Results
                if np.isnan(iou_score):
                    iou_score = "N/A (Union = 0)"
                else:
                    iou_score = f"{iou_score:.4f}"

                if np.isnan(precision):
                    precision = "N/A"
                else:
                    precision = f"{precision:.4f}"

                if np.isnan(recall):
                    recall = "N/A"
                else:
                    recall = f"{recall:.4f}"

                if np.isnan(f1):
                    f1 = "N/A"
                else:
                    f1 = f"{f1:.4f}"
            else:
                print("Eredeti maszk nem található.")
        else:
            print("Nincs hozzá tartozó eredeti maszk a megadott képhez.")

        return {
            'original_image': original_image,
            'original_mask': original_mask_filename,
            'predicted_mask': predicted_mask_filename,
            'overlay_image': overlay_filename,
            'iou_score': iou_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'has_mask': has_mask,
            'ship_detected': ship_detected  # Új flag átadása a sablonnak
        }

    except Exception as e:
        print(f"Hiba a kép feldolgozása során: {e}")
        raise e

# Routes
@app.route('/')
def index():
    return render_template('upload.html')  # Használjuk az upload.html-t a formhoz

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Feltöltési kérés érkezett.")
    if 'file' not in request.files:
        flash('Nincs kép kiválasztva.')
        print("Nincs kép kiválasztva.")
        return redirect(url_for('index'))
    file = request.files['file']

    if file.filename == '':
        flash('Nincs kép kiválasztva.')
        print("Nincs kép kiválasztva.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Save Image
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            print(f"Kép sikeresen mentve: {image_path}")

            # Check if image_id exists in CSV
            image_id = filename  # Feltételezve, hogy ImageId a fájlnévvel együtt tartalmazza az extension-t
            result = process_image(image_path, image_id=image_id, source_folder='uploads')

            return render_template('result.html',
                                   original_image=result['original_image'],
                                   original_mask=result['original_mask'],
                                   predicted_mask=result['predicted_mask'],
                                   overlay_image=result['overlay_image'],
                                   iou_score=result['iou_score'],
                                   precision=result['precision'],
                                   recall=result['recall'],
                                   f1_score=result['f1_score'],
                                   has_mask=result['has_mask'],
                                   ship_detected=result['ship_detected'],
                                   source='upload')  # Source átadása
        except Exception as e:
            flash(f'Hiba történt a kép feldolgozása során: {e}')
            print(f"Hiba történt a kép feldolgozása során: {e}")
            return redirect(url_for('index'))
    else:
        flash('Érvénytelen fájltípus. Csak PNG, JPG és JPEG fájlokat engedélyezünk.')
        print("Érvénytelen fájltípus.")
        return redirect(url_for('index'))

@app.route('/random_test', methods=['GET'])
def random_test():
    print("Random Test Kép Kérése")
    try:
        # List all images in the test folder
        test_images = [f for f in os.listdir(app.config['TEST_FOLDER']) if allowed_file(f)]
        if not test_images:
            flash('Nincsenek képek a test mappában.')
            print("Nincsenek képek a test mappában.")
            return redirect(url_for('index'))

        # Select a random image
        selected_image = random.choice(test_images)
        image_path = os.path.join(app.config['TEST_FOLDER'], selected_image)
        print(f"Választott Test Kép: {image_path}")

        # Determine image_id (assumes ImageId is the filename)
        image_id = selected_image

        # Process the image
        result = process_image(image_path, image_id=image_id, source_folder='test')

        return render_template('result.html',
                               original_image=result['original_image'],
                               original_mask=result['original_mask'],
                               predicted_mask=result['predicted_mask'],
                               overlay_image=result['overlay_image'],
                               iou_score=result['iou_score'],
                               precision=result['precision'],
                               recall=result['recall'],
                               f1_score=result['f1_score'],
                               has_mask=result['has_mask'],
                               ship_detected=result['ship_detected'],
                               source='test')  # Source átadása
    except Exception as e:
        flash(f'Hiba történt a random test kép feldolgozása során: {e}')
        print(f"Hiba történt a random test kép feldolgozása során: {e}")
        return redirect(url_for('index'))

@app.route('/random_train', methods=['GET'])
def random_train():
    print("Random Train Kép Kérése")
    try:
        # List all images in the train folder
        train_images = [f for f in os.listdir(app.config['TRAIN_FOLDER']) if allowed_file(f)]
        if not train_images:
            flash('Nincsenek képek a train mappában.')
            print("Nincsenek képek a train mappában.")
            return redirect(url_for('index'))

        # Select a random image
        selected_image = random.choice(train_images)
        image_path = os.path.join(app.config['TRAIN_FOLDER'], selected_image)
        print(f"Választott Train Kép: {image_path}")

        # Determine image_id (assumes ImageId is the filename)
        image_id = selected_image

        # Process the image
        result = process_image(image_path, image_id=image_id, source_folder='train')

        return render_template('result.html',
                               original_image=result['original_image'],
                               original_mask=result['original_mask'],
                               predicted_mask=result['predicted_mask'],
                               overlay_image=result['overlay_image'],
                               iou_score=result['iou_score'],
                               precision=result['precision'],
                               recall=result['recall'],
                               f1_score=result['f1_score'],
                               has_mask=result['has_mask'],
                               ship_detected=result['ship_detected'],
                               source='train')  # Source átadása
    except Exception as e:
        flash(f'Hiba történt a random train kép feldolgozása során: {e}')
        print(f"Hiba történt a random train kép feldolgozása során: {e}")
        return redirect(url_for('index'))

# Route for Retry Functionality
@app.route('/retry/<source>', methods=['GET'])
def retry(source):
    print(f"Retry kérés érkezett a {source} forrásból.")
    if source == 'test':
        return redirect(url_for('random_test'))
    elif source == 'train':
        return redirect(url_for('random_train'))
    else:
        flash('Érvénytelen forrás.')
        print("Érvénytelen forrás a retry kérésben.")
        return redirect(url_for('index'))


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True,port=5001)
