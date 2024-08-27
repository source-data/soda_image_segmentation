import os
import xml.etree.ElementTree as ET
import zipfile
import shutil
import random
from PIL import Image, ImageDraw

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_extracted_dir(base_dir):
    sub_dirs = os.listdir(base_dir)
    if len(sub_dirs) == 1 and os.path.isdir(os.path.join(base_dir, sub_dirs[0])):
        return os.path.join(base_dir, sub_dirs[0])
    return base_dir

def convert_annotations(xml_path, images_dir, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    visualize_output_dir = os.path.join(output_dir, 'test_image_clef')

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(visualize_output_dir, exist_ok=True)

    missing_images = []
    processed_images = 0

    for annotation in root.findall('annotation'):
        filename = annotation.find('filename').text
        image_file = os.path.join(images_dir, f"{filename}.jpg")
        if not os.path.exists(image_file):
            missing_images.append(image_file)
            continue

        image_output_path = os.path.join(images_output_dir, f"{filename}.jpg")
        shutil.copy(image_file, image_output_path)

        label_file = os.path.join(labels_output_dir, f"{filename}.txt")
        label_content = []
        bounding_boxes = []

        img = Image.open(image_file)
        img_width, img_height = img.size

        for obj in annotation.findall('object'):
            points = obj.findall('point')
            if len(points) != 4:
                continue
            x1, y1 = float(points[0].get('x')), float(points[0].get('y'))
            x2, y2 = float(points[1].get('x')), float(points[1].get('y'))
            x3, y3 = float(points[2].get('x')), float(points[2].get('y'))
            x4, y4 = float(points[3].get('x')), float(points[3].get('y'))

            min_x = min(x1, x2, x3, x4)
            max_x = max(x1, x2, x3, x4)
            min_y = min(y1, y2, y3, y4)
            max_y = max(y1, y2, y3, y4)

            if max_x - min_x == img_width and max_y - min_y == img_height:
                continue

            center_x = ((min_x + max_x) / 2) / img_width
            center_y = ((min_y + max_y) / 2) / img_height
            width = (max_x - min_x) / img_width
            height = (max_y - min_y) / img_height

            label_content.append(f"0 {center_x} {center_y} {width} {height}")
            bounding_boxes.append(((min_x, min_y), (max_x, max_y)))

        if label_content:
            with open(label_file, 'w') as f:
                for line in label_content:
                    f.write(line + "\n")
            processed_images += 1

        if bounding_boxes:
            visualize_image(image_file, bounding_boxes, os.path.join(visualize_output_dir, f"{filename}.jpg"))

    if missing_images:
        print(f"Warning: The following images were not found in the directory '{images_dir}':")
        for img in missing_images:
            print(img)

    print(f"Processed {processed_images} images.")
    return processed_images

def visualize_image(image_path, bounding_boxes, save_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    colors = ['red', 'blue', 'green', 'yellow', 'purple']

    for i, box in enumerate(bounding_boxes):
        draw.rectangle(box, outline=colors[i % len(colors)], width=2)

    img.save(save_path)

def main():
    zip_path = 'data/FigureSeparationTest2016.zip'
    extract_dir = 'data/FigureSeparationTest2016'
    xml_path = 'data/FigureSeparationTest2016GT.xml'
    output_dir = 'data/ImageCLEF/test'

    print(f"Extracting images from {zip_path} to {extract_dir}...")
    extract_zip(zip_path, extract_dir)

    extract_dir = get_extracted_dir(extract_dir)
    if not os.path.exists(extract_dir):
        print(f"Error: Extraction failed or directory {extract_dir} does not exist.")
        return

    print(f"Converting annotations from {xml_path}...")
    processed_images = convert_annotations(xml_path, extract_dir, output_dir)

    if processed_images == 0:
        print("Error: Conversion failed, no files written to the output directory.")
    else:
        print(f"Completed conversion. Output saved to {output_dir}")

    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    print(f"Images in output directory: {os.listdir(images_output_dir)}")
    print(f"Labels in output directory: {os.listdir(labels_output_dir)}")

    shutil.rmtree(extract_dir)

if __name__ == "__main__":
    main()
