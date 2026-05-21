from PIL import Image, ImageDraw, ImageFont
import io
import random
import time

def create_dummy_image_with_text(text: str, width: int = 600, height: int = 200) -> Image.Image:
    """
    Creates a simple dummy image with text for demonstration purposes.
    Adds some noise to simulate a real-world scanned document.
    """
    img = Image.new('RGB', (width, height), color=(255, 255, 255)) # White background
    d = ImageDraw.Draw(img)
    try:
        # Try to use a common font, fallback if not found
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: 'arial.ttf' not found, using default font.")

    # Add some random noise to simulate a scanned document
    for _ in range(int(width * height * 0.005)): # 0.5% of pixels as noise
        x, y = (random.randint(0, width - 1), random.randint(0, height - 1))
        d.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # Draw the main text
    text_bbox = d.textbbox((0,0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x_pos = (width - text_width) // 2
    y_pos = (height - text_height) // 2
    d.text((x_pos, y_pos), text, fill=(0, 0, 0), font=font) # Black text
    return img

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Applies a preprocessing trick (binarization) to enhance text for OCR.
    This is the "preprocessing hilesi" (preprocessing trick) mentioned in the article.
    Converting to grayscale and then binarizing helps to remove noise and
    clarify text boundaries, improving OCR accuracy.
    """
    print("Applying preprocessing: converting to grayscale and binarizing...")
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Apply binarization (thresholding)
    # Pixels above the threshold become white (255), below become black (0).
    # A common threshold value is 128-180, adjust based on image quality.
    threshold = 180 
    binarized_image = gray_image.point(lambda p: 255 if p > threshold else 0)
    
    return binarized_image

def gemma_ocr_mock(image: Image.Image) -> str:
    """
    Mocks the Gemma 4 (4B parameter model) local OCR functionality.
    In a real application, this function would:
    1. Load the Gemma 4 model (e.g., using Hugging Face transformers).
    2. Convert the preprocessed 'image' object into a format suitable for Gemma 4 (e.g., base64, pixel array).
    3. Craft a prompt for Gemma 4 to extract text from the image.
       Example prompt: "Extract all text from this image: [image_data]"
    4. Run inference with the local Gemma 4 model.
    5. Parse Gemma 4's output to retrieve the extracted text.

    For this self-contained example, we simulate the output based on the expected text,
    introducing slight imperfections to mimic real-world OCR.
    """
    print("Simulating Gemma 4 (4B) local OCR inference...")
    time.sleep(1) # Simulate processing time
    
    # This is where Gemma 4 would analyze the 'image' and return text.
    # We'll return a slightly altered version of our original text.
    mock_extracted_text = "Hello Gemma 4! Save 50 USD/month with local OCR. This is a test document."
    
    return mock_extracted_text

if __name__ == "__main__":
    print("--- Local OCR Demonstration with Simulated Gemma 4 ---")
    
    # 1. Create a dummy image with text (simulating a scanned document)
    # In a real scenario, you would load an image from a file:
    # original_image = Image.open("path/to/your/document.png")
    
    original_text = "Hello Gemma 4! Save $50/month with local OCR. This is a test document."
    original_image = create_dummy_image_with_text(original_text)
    print("Dummy image created with text for demonstration.")
    
    # Optional: Save the original and preprocessed images to visualize the steps
    # original_image.save("original_document.png")
    # print("Original image saved as original_document.png")

    # 2. Apply the preprocessing trick to enhance the image for OCR
    preprocessed_image = preprocess_image_for_ocr(original_image)
    
    # Optional: Save the preprocessed image
    # preprocessed_image.save("preprocessed_document.png")
    # print("Preprocessed image saved as preprocessed_document.png")

    # 3. Perform OCR using the (simulated) local Gemma 4 model
    extracted_text = gemma_ocr_mock(preprocessed_image)
    
    print("\n--- OCR Result ---")
    print(f"Original Intended Text:   '{original_text}'")
    print(f"Extracted Text (Simulated): '{extracted_text}'")
    print("\nDemonstration complete. In a real setup, Gemma 4 would process the preprocessed image.")
