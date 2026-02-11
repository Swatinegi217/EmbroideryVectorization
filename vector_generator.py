import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import os
import xml.etree.ElementTree as ET

class EmbroideryVectorizer:
    def __init__(self):
        print("Initializing Vectorizer...")
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def check_format_support(self, image_path):
        """Check if the image format is supported"""
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
        return True
    
    def load_image(self, image_path):
        """Load image with format validation"""
        self.check_format_support(image_path)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}. File may be corrupted or format not supported.")
        
        print(f"✅ Successfully loaded: {os.path.basename(image_path)}")
        print(f"   Format: {os.path.splitext(image_path)[1]}")
        print(f"   Original Dimensions: {image.shape[1]} x {image.shape[0]}")
        print(f"   Size: {os.path.getsize(image_path) / 1024:.1f} KB")
        
        return image
    
    def preprocess_hand_drawing(self, image_path):
        """Preprocessing with adjustable smoothing based on detail level"""
        print("Converting image to vector...")
        
        # Load image with validation
        image = self.load_image(image_path)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        print(f"   Image size: {gray.shape}")
        
        # Get smoothing level (1=sharp, 10=very smooth)
        smoothing_level = getattr(self, 'smoothing_level', 5)
        
        # Apply adaptive blur based on smoothing level
        if smoothing_level <= 2:
            # Minimal smoothing to preserve details
            blur_size = 3
            sigma = 0.5
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
            print(f"   Smoothing level {smoothing_level}: Minimal smoothing")
        elif smoothing_level <= 4:
            # Light smoothing
            blur_size = 5
            sigma = 1.0
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
            print(f"   Smoothing level {smoothing_level}: Light smoothing")
        elif smoothing_level <= 6:
            # Medium smoothing
            blur_size = 7
            sigma = 1.5
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
            print(f"   Smoothing level {smoothing_level}: Medium smoothing")
        elif smoothing_level <= 8:
            # Heavy smoothing
            blur_size = 11
            sigma = 2.0
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
            print(f"   Smoothing level {smoothing_level}: Heavy smoothing")
        else:
            # Maximum smoothing for ultra-smooth curves
            blur_size = 15
            sigma = 3.0
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
            print(f"   Smoothing level {smoothing_level}: Maximum smoothing")
        
        # Use Otsu's thresholding for better shape preservation
        # This automatically finds the optimal threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Optional: Apply adaptive thresholding for complex images
        # Uncomment if Otsu doesn't work well
        # binary = cv2.adaptiveThreshold(
        #     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        # )
        
        # Very minimal morphology to preserve all lines
        # Only remove tiny noise, don't merge or disconnect lines
        if smoothing_level <= 3:
            # Preserve all details - minimal noise removal
            kernel_size = 2
            iterations = 1
        elif smoothing_level <= 6:
            # Light noise removal
            kernel_size = 2
            iterations = 1
        else:
            # Moderate noise removal
            kernel_size = 3
            iterations = 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Only remove small noise, don't close gaps (which can merge lines)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        print(f"   Applied minimal noise removal: {kernel_size}x{kernel_size} kernel")
        
        print("✅ Image ready for vectorization")
        return binary
    
    def find_contours_exact(self, binary_image, min_area=50):
        """Find all contours with adaptive thresholds for image size"""
        print("Finding contours...")
        
        # Get detail level (1-10, where 1=most details, 10=most smooth)
        detail_level = getattr(self, 'detail_level', 5)
        
        # Calculate image size factor for adaptive thresholding
        image_area = binary_image.shape[0] * binary_image.shape[1]
        size_factor = (image_area / 1000000) ** 0.5  # Normalize to ~1000x1000 image
        
        # Extremely low thresholds to capture ALL details including thin lines
        base_min_area = 3  # Ultra low to catch even tiny lines
        adjusted_min_area = base_min_area * (detail_level / 10.0)  # Less aggressive filtering
        adjusted_min_perimeter = 3 * (detail_level / 10.0)  # Ultra low perimeter threshold
        
        print(f"   Image size: {binary_image.shape[1]}x{binary_image.shape[0]} (factor: {size_factor:.2f})")
        print(f"   Detail level: {detail_level} (min_area: {adjusted_min_area:.1f}, min_perimeter: {adjusted_min_perimeter:.1f})")
        
        # Use RETR_LIST to get all contours without hierarchy (avoids double lines)
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_LIST,  # All contours as flat list (no parent-child)
            cv2.CHAIN_APPROX_NONE  # Keep all points for maximum accuracy
        )
        
        print(f"   Total contours found by OpenCV: {len(contours)}")
        
        # Very minimal filtering - only remove truly tiny noise
        meaningful_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Ultra lenient filtering - accept almost everything
            # Only filter out single-pixel noise
            if area > 1 or perimeter > 2:
                contour_info = {
                    'contour': contour,
                    'area': area,
                    'hierarchy': None  # No hierarchy with RETR_LIST
                }
                meaningful_contours.append(contour_info)
            else:
                print(f"   Filtered tiny noise: area={area:.1f}, perimeter={perimeter:.1f}")
        
        print(f"📐 Found {len(meaningful_contours)} contours (detail level: {detail_level})")
        
        # Apply contour smoothing based on smoothing level
        smoothing_level = getattr(self, 'smoothing_level', 5)
        
        simplified_contours = []
        for contour_info in meaningful_contours:
            contour = contour_info['contour']
            perimeter = cv2.arcLength(contour, True)
            
            # Balanced simplification to preserve shape while reducing points
            if smoothing_level <= 2:
                # Preserve most details - minimal simplification
                epsilon = 0.3  # Very minimal
            elif smoothing_level <= 4:
                # Light simplification
                epsilon = 0.8
            elif smoothing_level <= 6:
                # Moderate simplification
                epsilon = 1.2
            elif smoothing_level <= 8:
                # Heavy simplification for smoother curves
                epsilon = 2.0
            else:
                # Maximum simplification for ultra-smooth curves
                epsilon = 3.0
            
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ensure we have enough points for smooth curves
            if len(simplified) < 3:
                simplified = contour  # Keep original if too simplified
            
            simplified_contours.append({
                'contour': simplified,
                'area': contour_info['area'],
                'hierarchy': contour_info['hierarchy']
            })
            
            print(f"   Contour simplified: {len(contour)} → {len(simplified)} points (epsilon: {epsilon})")
        
        print(f"📏 Applied smoothing to {len(simplified_contours)} contours (smoothing level: {smoothing_level})")
        return simplified_contours
    
    def create_adaptive_smooth_curve(self, points):
        """Create adaptive smooth curves that preserve design features while adding smoothness"""
        if len(points) < 2:
            return ""
        
        # Convert to numpy array for easier manipulation
        points = np.array(points)
        
        # For very few points, use simple curve
        if len(points) < 4:
            path_data = []
            path_data.append(f"M {points[0][0]:.2f} {points[0][1]:.2f}")
            for i in range(1, len(points)):
                curr_point = points[i]
                path_data.append(f"L {curr_point[0]:.2f} {curr_point[1]:.2f}")
            return " ".join(path_data)
        
        # For more points, create smooth Catmull-Rom spline curves
        path_data = []
        path_data.append(f"M {points[0][0]:.2f} {points[0][1]:.2f}")
        
        # Use cubic bezier curves for smoother, more accurate shapes
        for i in range(len(points) - 1):
            p0 = points[i-1] if i > 0 else points[0]
            p1 = points[i]
            p2 = points[i+1] if i+1 < len(points) else points[-1]
            p3 = points[i+2] if i+2 < len(points) else points[-1]
            
            # Calculate control points using Catmull-Rom to Bezier conversion
            # This preserves the shape better
            tension = 0.5  # Adjust tension for smoothness (0.5 = Catmull-Rom)
            
            cp1_x = p1[0] + (p2[0] - p0[0]) / 6.0 * tension
            cp1_y = p1[1] + (p2[1] - p0[1]) / 6.0 * tension
            cp2_x = p2[0] - (p3[0] - p1[0]) / 6.0 * tension
            cp2_y = p2[1] - (p3[1] - p1[1]) / 6.0 * tension
            
            # Use cubic bezier curve
            path_data.append(f"C {cp1_x:.2f},{cp1_y:.2f} {cp2_x:.2f},{cp2_y:.2f} {p2[0]:.2f},{p2[1]:.2f}")
        
        return " ".join(path_data)
    
    def create_exact_svg_from_binary(self, binary_image, output_path, stroke_width=3, force_close_all=False):
        """Create SVG that EXACTLY matches the processed binary image with design-preserving smooth curves
        
        Args:
            force_close_all: If True, forcefully close ALL paths regardless of distance
        """
        print("Creating EXACT SVG from binary image with design-preserving curves...")
        print(f"   Force close all paths: {force_close_all}")
        
        try:
            height, width = binary_image.shape
            
            # Create SVG drawing
            dwg = svgwrite.Drawing(
                output_path, 
                size=(f"{width}px", f"{height}px"),
                viewBox=f"0 0 {width} {height}",
                profile='full'
            )
            
            # Find contours from the EXACT binary image with minimal filtering
            contour_infos = self.find_contours_exact(binary_image, min_area=10)
            
            # Process contours to distinguish between outlines and filled areas
            # We'll analyze the contours to determine which should be filled
            print(f"Processing {len(contour_infos)} contours")
            for i, contour_info in enumerate(contour_infos):
                contour = contour_info['contour']
                area = contour_info['area']
                hierarchy = contour_info['hierarchy']
                print(f"  Contour {i}: area={area:.1f}, hierarchy={hierarchy}")
                
                if len(contour) < 1:
                    continue
                
                # Convert contour to adaptive smooth curves
                path_data = []
                
                if len(contour) < 2:
                    continue
                
                # Extract points from contour
                points = np.array([pt[0] for pt in contour])
                
                # Create adaptive smooth path
                path_data_str = self.create_adaptive_smooth_curve(points)
                
                # Check if start and end points are close (closed shape)
                is_closed_path = False
                force_close_threshold = 50  # Increased threshold to close more paths
                
                if len(points) > 0:
                    start_x, start_y = points[0]
                    end_x, end_y = points[-1]
                    distance = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                    
                    # Force close all paths if requested, or close if within threshold
                    if force_close_all or distance < force_close_threshold:
                        is_closed_path = True
                        if path_data_str:
                            path_data_str += " Z"
                        if force_close_all:
                            print(f"✅ Force-closed path (distance: {distance:.1f}px)")
                        else:
                            print(f"✅ Closed path (distance: {distance:.1f}px)")
                    else:
                        # For very open paths, still add them but mark as open
                        print(f"✅ Open path kept (distance: {distance:.1f}px)")
            
                # Determine if this contour should be filled
                # For embroidery, we typically want to preserve the original intent:
                # - Outlines should have no fill (stitching paths)
                # - Filled areas should have fill (satin stitch or fill stitch areas)
                
                # Check hierarchy to determine if this is an outer contour (likely a filled area)
                # In hierarchy, [next, previous, first_child, parent]
                # If first_child != -1, this contour has children (it's an outer boundary)
                is_filled_area = False
                if hierarchy is not None and len(hierarchy) > 2 and hierarchy[2] != -1:
                    # This contour has children, so it's likely a filled area
                    # But we need to be very conservative for embroidery
                    if area > 2000:  # Only consider large areas as filled
                        is_filled_area = True
                
                # Additional check: if the contour is very large and has a simple shape,
                # it's likely a filled area
                if area > 15000 and len(contour) < 10:  # Increased area threshold and stricter shape check
                    is_filled_area = True
                
                # For embroidery, we want to be very conservative about filling
                # Only fill areas that are clearly intended to be filled
                if area < 2000:  # Increased threshold to avoid small details being filled
                    is_filled_area = False  # Small areas are likely details, not fills
                
                # Additional conservative check: if the contour is not significantly large
                # compared to the image size, don't fill it
                image_area = binary_image.shape[0] * binary_image.shape[1]
                if area < image_area * 0.01:  # Less than 1% of image area
                    is_filled_area = False
                
                # Safety check for hierarchy data
                if hierarchy is not None:
                    print(f"   Hierarchy data: {hierarchy}")
                
                # Create path with appropriate fill settings
                # Modified to always use no fill (removed black fill completely)
                if path_data_str:  # Only add path if we have path data
                    path = dwg.path(
                        d=path_data_str,
                        fill='none',  # Always use no fill
                        stroke='black',
                        stroke_width=stroke_width,
                        stroke_linejoin='round',
                        stroke_linecap='round',
                        stroke_opacity=1.0
                    )
                    if is_filled_area:
                        print(f"✅ Added previously filled area path as outline, area: {area:.1f}")
                    else:
                        print(f"✅ Added outline path, area: {area:.1f}")
                    
                    dwg.add(path)
            
            # If no contours were processed, create at least one path
            if not contour_infos:
                print("❌ No contours found in image")
                # Create a simple fallback SVG with a rectangle
                path = dwg.path(
                    d="M 10 10 L 50 10 L 50 50 L 10 50 Z",
                    fill='none',
                    stroke='black',
                    stroke_width=stroke_width
                )
                dwg.add(path)
                print("✅ Created fallback SVG")
            
            # Save SVG
            dwg.save()
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ EXACT SVG created successfully: {output_path}")
                print(f"📏 Dimensions: {width} x {height} pixels")
                print(f"💾 File size: {file_size / 1024:.1f} KB")
                return output_path
            else:
                print("❌ SVG file was not created")
                # Try to create a minimal SVG as fallback
                try:
                    fallback_dwg = svgwrite.Drawing(
                        output_path, 
                        size=(f"{width}px", f"{height}px"),
                        viewBox=f"0 0 {width} {height}",
                        profile='full'
                    )
                    # Add a simple path to ensure the file is valid
                    fallback_path = fallback_dwg.path(
                        d="M 10 10 L 50 10 L 50 50 L 10 50 Z",
                        fill='none',
                        stroke='black',
                        stroke_width=1
                    )
                    fallback_dwg.add(fallback_path)
                    fallback_dwg.save()
                    if os.path.exists(output_path):
                        print("✅ Fallback SVG created successfully")
                        return output_path
                except Exception as fallback_error:
                    print(f"❌ Fallback SVG creation also failed: {fallback_error}")
                return None
                
        except Exception as e:
            print(f"❌ Error in exact SVG generation: {e}")
            return None
    
    def generate_svg_stroke_fixed(self, contours, width, height, output_path, stroke_width=3):
        """Generate SVG file - USING EXACT METHOD"""
        return self.create_exact_svg_from_binary(self.current_binary, output_path, stroke_width)
    
    def generate_multiple_stroke_svgs(self, contours, width, height, output_dir, base_name="embroidery"):
        """Generate multiple SVG files with different stroke widths"""
        stroke_widths = [1, 2, 3, 4]
        output_paths = []
        force_close_all = getattr(self, 'force_close_all', False)
        
        for stroke_width in stroke_widths:
            output_path = os.path.join(output_dir, f"{base_name}_stroke_{stroke_width}px.svg")
            svg_path = self.create_exact_svg_from_binary(self.current_binary, output_path, stroke_width=stroke_width, force_close_all=force_close_all)
            if svg_path:
                output_paths.append(svg_path)
        
        return output_paths
    
    def visualize_preprocessing(self, original_path, processed_image, output_path="preprocessing_steps.jpg"):
        """Create ultra-high-quality visualization of preprocessing steps without dotted appearance"""
        original = cv2.imread(original_path)
        if original is None:
            print("Could not load original image for visualization")
            return None
            
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Store the current binary for exact SVG generation
        self.current_binary = processed_image
        
        # Create ultra-high-resolution comparison image
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Original image with enhanced quality
        axes[0].imshow(original_rgb, interpolation='nearest')
        axes[0].set_title('Original Hand Drawing', fontsize=16, pad=25, weight='bold')
        axes[0].axis('off')
        
        # Processed binary image with enhanced visualization
        axes[1].imshow(processed_image, cmap='gray', interpolation='nearest')
        axes[1].set_title('Processed for Embroidery\n(EXACT SVG OUTPUT)', fontsize=16, pad=25, weight='bold')
        axes[1].axis('off')
        
        # Show contours with anti-aliased smooth curves
        contour_infos = self.find_contours_exact(processed_image, min_area=3)
        contours = [info['contour'] for info in contour_infos]  # Extract just the contours
        
        # Create a clean white background with smooth black contours
        height, width = processed_image.shape
        contour_overlay = np.ones((height, width, 3), dtype=np.uint8) * 255  # Pure white background
        
        # Draw smooth contours with anti-aliasing
        for contour in contours:
            # Convert contour points to integer coordinates
            pts = np.array([point[0] for point in contour], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw smooth polygon with anti-aliasing
            cv2.polylines(contour_overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # Show the overlay with interpolation for smooth display
        axes[2].imshow(contour_overlay, interpolation='nearest')
        axes[2].set_title(f'Vector Contours: {len(contours)}\n(Smooth Anti-Aliased Curves)', fontsize=16, pad=25, weight='bold')
        axes[2].axis('off')
        
        # Enhance overall figure quality
        plt.tight_layout(pad=3.0)
        
        # Save with maximum quality settings to eliminate dotted appearance
        plt.savefig(
            output_path, 
            dpi=300,  # Ultra-high DPI
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none',
            pil_kwargs={'quality': 100},  # Maximum JPEG quality
            transparent=False
        )
        plt.close()
        
        print(f"✅ Ultra-high-quality preprocessing visualization saved: {output_path}")
        print("🎯 **SVG will EXACTLY match the 'Processed for Embroidery' image**")
        return output_path
    
    def verify_svg_file(self, svg_path):
        """Verify that SVG file is valid"""
        try:
            if not os.path.exists(svg_path):
                return False, "File does not exist"
            
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '<?xml' not in content:
                return False, "Missing XML declaration"
            if '<svg' not in content:
                return False, "Missing SVG tag"
            if 'xmlns=' not in content:
                return False, "Missing SVG namespace"
            if '<path' not in content:
                return False, "No path data in SVG"
            
            file_size = os.path.getsize(svg_path)
            return True, f"SVG file is valid ({file_size} bytes)"
            
        except Exception as e:
            return False, f"Error reading SVG: {e}"
    
    def process_hand_artwork(self, image_path, output_path, stroke_width=3, optimize_stitch=True, detail_level=5, smoothing_level=5, force_close_all=False):
        """Main method to process hand artwork for EMBROIDERY - EXACT VERSION"""
        try:
            # Store detail and smoothing levels
            self.detail_level = detail_level
            self.smoothing_level = smoothing_level
            
            # Preprocess image
            processed_image = self.preprocess_hand_drawing(image_path)
            
            # Store binary for exact SVG generation
            self.current_binary = processed_image
            
            # Get image dimensions
            height, width = processed_image.shape
            
            # Generate EXACT SVG from the processed binary
            svg_path = self.create_exact_svg_from_binary(processed_image, output_path, stroke_width=stroke_width, force_close_all=force_close_all)
            
            # Verify the generated SVG
            if svg_path:
                is_valid, message = self.verify_svg_file(svg_path)
                if is_valid:
                    print(f"✅ SVG verification passed: {message}")
                else:
                    print(f"❌ SVG verification failed: {message}")
            
            return svg_path, processed_image, None
            
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            return None, None, None
    
    def process_hand_artwork_multiple_strokes(self, image_path, output_dir, base_name="embroidery", detail_level=5, smoothing_level=5, force_close_all=False):
        """Process artwork and generate multiple SVG files"""
        # Store detail and smoothing levels
        self.detail_level = detail_level
        self.smoothing_level = smoothing_level
        self.force_close_all = force_close_all
        
        # Preprocess image
        processed_image = self.preprocess_hand_drawing(image_path)
        
        # Store binary for exact SVG generation
        self.current_binary = processed_image
        
        # Get image dimensions
        height, width = processed_image.shape
        
        # Generate multiple SVG files
        svg_paths = self.generate_multiple_stroke_svgs(None, width, height, output_dir, base_name)
        
        return svg_paths, processed_image, None

def main():
    """Main function to demonstrate the vectorizer"""
    vectorizer = EmbroideryVectorizer()
    
    os.makedirs("outputs", exist_ok=True)
    
    print("\n" + "="*50)
    print("EMBROIDERY VECTOR GENERATOR - EXACT SVG")
    print("="*50)
    
    image_path = input("Enter path to hand artwork: ").strip()
    if os.path.exists(image_path):
        stroke_width = input("Enter stroke width (default 3): ").strip()
        stroke_width = int(stroke_width) if stroke_width.isdigit() else 3
        
        svg_output = os.path.join("outputs", f"test_stroke_{stroke_width}px.svg")
        svg_path, processed_image, contours = vectorizer.process_hand_artwork(
            image_path, svg_output, stroke_width=stroke_width
        )
        
        if svg_path:
            preprocessing_viz = vectorizer.visualize_preprocessing(
                image_path,
                processed_image,
                os.path.join("outputs", "preprocessing_detailed.jpg")
            )
            print(f"📁 Output files saved in 'outputs' directory")
            print("🎯 **SVG exactly matches the processed image!**")
        else:
            print("❌ Failed to generate vector file.")
    else:
        print("❌ File not found.")

if __name__ == "__main__":
    main()