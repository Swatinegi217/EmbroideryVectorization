import numpy as np
import cv2
from vector_generator import EmbroideryVectorizer
import os

# Create a test image with both filled areas and outlines
def create_test_image():
    # Create a blank canvas
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # Add a filled black rectangle (should be detected as filled area)
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), -1)
    
    # Add an outline circle (should be detected as outline)
    cv2.circle(img, (125, 100), 30, (0, 0, 0), 2)
    
    # Add some text
    cv2.putText(img, "TEST", (75, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

# Test the vectorizer
def test_vectorizer():
    # Create test image
    test_img = create_test_image()
    cv2.imwrite("test_artwork.png", test_img)
    
    # Initialize vectorizer
    vectorizer = EmbroideryVectorizer()
    
    # Process the image
    svg_path, processed_img, _ = vectorizer.process_hand_artwork(
        "test_artwork.png", 
        "test_output.svg", 
        stroke_width=3
    )
    
    if svg_path and os.path.exists(svg_path):
        print("✅ SVG file generated successfully!")
        
        # Check file size
        file_size = os.path.getsize(svg_path)
        print(f"📁 File size: {file_size} bytes")
        
        # Display first few lines of the SVG
        with open(svg_path, 'r') as f:
            content = f.read()
            print("📄 SVG content preview:")
            print(content[:300] + "..." if len(content) > 300 else content)
            
        return True
    else:
        print("❌ Failed to generate SVG file")
        return False

if __name__ == "__main__":
    print("Testing SVG generation...")
    success = test_vectorizer()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")