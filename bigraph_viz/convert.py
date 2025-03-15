import os
from pdf2image import convert_from_path
from PIL import Image

# Print current working directory
cwd = os.getcwd()
print("Current working directory:", cwd)

# Go up one level in the directory
cwd = os.path.dirname(cwd)

# Input PDF file (single page)
pdf_path = os.path.join(cwd, "notebooks/out/ecoli.pdf")

# Output directory and files
output_dir = os.path.join(cwd, "notebooks/out/")
output_full_res = os.path.join(output_dir, "ecoli_full_res.png")
output_scaled = os.path.join(output_dir, "ecoli_google_slides.png")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to an image at high DPI
images = convert_from_path(pdf_path, dpi=1200)  # Use high DPI for sharpness

if images:
    # Save the **full resolution** image first
    images[0].save(output_full_res, "PNG")
    print(f"Full resolution image saved at: {output_full_res}")

    # Resize image to match Google Slides width (3000 px)
    target_width = 3000
    aspect_ratio = images[0].width / images[0].height
    target_height = int(target_width / aspect_ratio)

    resized_image = images[0].resize((target_width, target_height), Image.LANCZOS)
    resized_image.save(output_scaled, "PNG")

    print(f"Resized Google Slides image saved at: {output_scaled}")
else:
    print("Error: No images found in PDF!")
