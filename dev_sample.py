import pandas as pd
import random
import os
import shutil

N_train = 1500 

N_test = 300

# Paths to CSV file and image directories
csv_path = 'train_ship_segmentations_v2.csv'
train_images_dir = 'train_v2/'
test_images_dir = 'test_v2/'

# Development directories
dev_train_images_dir = 'dev_train_v2/'
dev_test_images_dir = 'dev_test_v2/'

# Ensure the development directories exist
os.makedirs(dev_train_images_dir, exist_ok=True)
os.makedirs(dev_test_images_dir, exist_ok=True)

#Create Development Training Set

# Read the CSV file
df = pd.read_csv(csv_path)

# Create a boolean column 'has_ship' indicating if an image has any ships
df['has_ship'] = df['EncodedPixels'].notnull()

# Group by 'ImageId' and determine if each image contains ships
df_grouped = df.groupby('ImageId')['has_ship'].max().reset_index()

# Lists of image IDs with and without ships
images_with_ships = df_grouped[df_grouped['has_ship']]['ImageId'].tolist()
images_without_ships = df_grouped[~df_grouped['has_ship']]['ImageId'].tolist()

# Calculate the number of images to sample from each group
N_with_ships = int(N_train * 0.28)
N_without_ships = N_train - N_with_ships

# Randomly sample image IDs from each group
random.seed(42)  # Set seed for reproducibility
sampled_with_ships = random.sample(images_with_ships, N_with_ships)
sampled_without_ships = random.sample(images_without_ships, N_without_ships)

# Combine the sampled image IDs
sampled_train_images = sampled_with_ships + sampled_without_ships

# Copy the sampled training images to the development training directory
for image_id in sampled_train_images:
    src = os.path.join(train_images_dir, image_id)
    dst = os.path.join(dev_train_images_dir, image_id)
    shutil.copyfile(src, dst)

print(f"Copied {len(sampled_train_images)} training images to the development training directory.")

# Save the subset CSV for the development training set
df_dev = df[df['ImageId'].isin(sampled_train_images)]
df_dev.to_csv('train_ship_segmentations_dev.csv', index=False)

# Create Development Testing Set

# Get a list of all test image IDs
test_image_ids = os.listdir(test_images_dir)

# Randomly sample a subset of test images
sampled_test_images = random.sample(test_image_ids, N_test)

# Copy the sampled test images to the development test directory
for image_id in sampled_test_images:
    src = os.path.join(test_images_dir, image_id)
    dst = os.path.join(dev_test_images_dir, image_id)
    shutil.copyfile(src, dst)

print(f"Copied {len(sampled_test_images)} testing images to the development testing directory.")
