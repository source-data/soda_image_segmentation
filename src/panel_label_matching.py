import os
import json
from io import BytesIO
from PIL import Image
from assistants import map_panel_to_description
import shutil
from tqdm import tqdm

def load_image(image_path):
    with open(image_path, 'rb') as f:
        return BytesIO(f.read())

def save_failure(image_path, caption, figure_path, output_dir, predicted_label):
    os.makedirs(output_dir, exist_ok=True)

    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, image_filename)
    with open(output_image_path, 'wb') as f:
        f.write(load_image(image_path).getvalue())

    caption_filename = image_filename.replace('.png', '.txt')
    output_caption_path = os.path.join(output_dir, caption_filename)
    with open(output_caption_path, 'w') as f:
        f.write(f"Caption: {caption}\nPredicted Label: {predicted_label}")

    figure_filename = os.path.basename(figure_path)
    output_figure_path = os.path.join(output_dir, figure_filename)
    shutil.copyfile(figure_path, output_figure_path)

def evaluate_accuracy(image_dir, captions_file, test_figures_dir, failure_dir, cache_file):
    total_images = 0
    correct_matches = 0
    false_positives = 0
    false_negatives = 0

    # Load cached results
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
    else:
        cached_results = {}

    with open(captions_file, 'r') as f:
        captions = [json.loads(line) for line in f]

    test_figures = set(os.path.splitext(f)[0] for f in os.listdir(test_figures_dir) if f.endswith('.jpg'))

    for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
        if filename.endswith(".png"):
            figure_id, panel_label = filename.rsplit('_', 1)
            panel_label = panel_label.split('.')[0]

            if figure_id not in test_figures:
                continue

            image_path = os.path.join(image_dir, filename)
            panel_image = load_image(image_path)

            caption = next((item['figure_caption'] for item in captions if item['figure_id'] == figure_id), None)
            if not caption:
                print(f"Caption for figure ID {figure_id} not found. Skipping.")
                continue

            if filename in cached_results:
                panel_description_json = cached_results[filename]
            else:
                try:
                    panel_description_json = map_panel_to_description(panel_image, caption)
                    panel_description_json = panel_description_json.replace("```json", "").replace("```", "")
                    cached_results[filename] = panel_description_json

                    # Save cached result after each iteration
                    with open(cache_file, 'w') as f:
                        json.dump(cached_results, f, indent=4)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

            try:
                panel_description = json.loads(panel_description_json)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {filename}: {e}")
                continue

            predicted_label = panel_description.get("panel_label")
            if predicted_label == panel_label:
                correct_matches += 1
            else:
                figure_path = os.path.join(test_figures_dir, f"{figure_id}.jpg")
                save_failure(image_path, caption, figure_path, failure_dir, predicted_label)
                
                if predicted_label:  # If a label was predicted, it's a false positive
                    false_positives += 1
                else:  # If no label was predicted but it should have been, it's a false negative
                    false_negatives += 1
                    
            total_images += 1

    accuracy = correct_matches / total_images if total_images > 0 else 0
    return accuracy, false_positives, false_negatives

def save_results(results_file, accuracy, false_positives, false_negatives):
    results = {
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(results_file):
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def main():
    image_dir = 'data/segmented_images/'
    captions_file = 'data/figure_captions.jsonl'
    test_figures_dir = 'data/soda_panelization_figures/test/images'
    failure_dir = 'data/failures/'
    cache_file = 'data/panel_description_cache.json'
    results_file = 'data/results.json'

    # Check if results are already computed and saved
    results = load_results(results_file)
    if results:
        accuracy = results["accuracy"]
        false_positives = results["false_positives"]
        false_negatives = results["false_negatives"]
    else:
        accuracy, false_positives, false_negatives = evaluate_accuracy(image_dir, captions_file, test_figures_dir, failure_dir, cache_file)
        save_results(results_file, accuracy, false_positives, false_negatives)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

if __name__ == "__main__":
    main()
