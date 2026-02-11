import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import os
import tempfile
import time
import base64
import re
from vector_generator import EmbroideryVectorizer

def sanitize_filename(filename):
    """Sanitize filename by removing non-ASCII characters and special characters"""
    # Remove file extension first
    name_without_ext = os.path.splitext(filename)[0]
    
    # Remove non-ASCII characters
    name_without_ext = re.sub(r'[^\x00-\x7F]+', '_', name_without_ext)
    # Remove special characters except dots, hyphens, and underscores
    name_without_ext = re.sub(r'[^a-zA-Z0-9\._\-]', '_', name_without_ext)
    # Ensure it doesn't start or end with dot or underscore
    name_without_ext = name_without_ext.strip('._')
    # Replace multiple underscores with single underscore
    name_without_ext = re.sub(r'_+', '_', name_without_ext)
    
    # If filename is empty after sanitization, use a default name
    if not name_without_ext:
        name_without_ext = "embroidery_design"
    
    return name_without_ext

def extract_svg_path_data(svg_content):
    """Extract path data and fill information from SVG content"""
    try:
        # Find all path elements in SVG with their attributes
        path_matches = re.findall(r'<path[^>]*d="([^"]*)"[^>]*fill="([^"]*)"[^>]*>', svg_content)
        if path_matches:
            # Return list of tuples (path_data, fill_value)
            return path_matches
        else:
            # Fallback to old method if no fill attributes found
            simple_matches = re.findall(r'<path[^>]*d="([^"]*)"[^>]*>', svg_content)
            # Return list of tuples (path_data, 'none') for backward compatibility
            return [(path, 'none') for path in simple_matches]
    except Exception as e:
        st.error(f"Error extracting SVG paths: {e}")
        return []

def get_svg_dimensions(svg_content):
    """Extract SVG dimensions from content"""
    try:
        # Extract width and height from SVG
        width_match = re.search(r'width="([^"]*)"', svg_content)
        height_match = re.search(r'height="([^"]*)"', svg_content)
        
        if width_match and height_match:
            width = float(width_match.group(1).replace('px', ''))
            height = float(height_match.group(1).replace('px', ''))
            return width, height
        else:
            # Default dimensions if not found
            return 612, 792
    except Exception as e:
        return 612, 792

def calculate_bounding_box(path_data_list):
    """Calculate bounding box for all paths to center the design"""
    if not path_data_list:
        return 0, 0, 612, 792
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for path_data in path_data_list:
        commands = path_data.split()
        i = 0
        while i < len(commands):
            cmd = commands[i]
            if cmd in ['M', 'm', 'L', 'l'] and i + 2 < len(commands):
                try:
                    x = float(commands[i+1])
                    y = float(commands[i+2])
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    i += 3
                except:
                    i += 1
            else:
                i += 1
    
    # If no valid coordinates found, use default
    if min_x == float('inf'):
        return 0, 0, 612, 792
    
    return min_x, min_y, max_x, max_y

def create_proper_ai_file_with_design(svg_path, ai_output_path):
    """Create proper Adobe Illustrator file with ACTUAL DESIGN - FIXED VERSION"""
    try:
        # Read SVG content
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Extract path data and fill information from SVG
        path_data_list = extract_svg_path_data(svg_content)
        
        # Convert to list of path data strings for backward compatibility in other functions
        path_strings = [path[0] if isinstance(path, tuple) else path for path in path_data_list]
        
        if not path_data_list:
            st.error("No path data found in SVG")
            return None
        
        # Extract stroke width
        stroke_width_match = re.search(r'stroke-width="([^"]*)"', svg_content)
        stroke_width = stroke_width_match.group(1) if stroke_width_match else "1"
        
        # Calculate bounding box and center the design
        min_x, min_y, max_x, max_y = calculate_bounding_box(path_strings)
        design_width = max_x - min_x
        design_height = max_y - min_y
        
        # AI page dimensions (standard letter size)
        page_width = 612
        page_height = 792
        
        # Calculate scaling to fit design on page
        scale_x = (page_width - 100) / design_width if design_width > 0 else 1
        scale_y = (page_height - 100) / design_height if design_height > 0 else 1
        scale = min(scale_x, scale_y, 2.0)  # Limit maximum scale
        
        # Calculate translation to center the design
        translate_x = (page_width - (design_width * scale)) / 2 - (min_x * scale)
        translate_y = (page_height - (design_height * scale)) / 2 - (min_y * scale)
        
        # Create AI file content with actual design
        ai_content = f"""%!PS-Adobe-3.0 EPSF-3.0
%%Creator: Fashion Design Studio
%%Title: {os.path.basename(svg_path)}
%%CreationDate: {time.strftime("%Y-%m-%d %H:%M:%S")
}
%%BoundingBox: 0 0 {page_width} {page_height}
%%LanguageLevel: 3
%%EndComments

%%BeginProlog
/inch {{ 72 mul }} def
/cm {{ 28.34646 mul }} def
%%EndProlog

%%BeginSetup
%%EndSetup

% Center and scale the design properly
{translate_x} {translate_y} translate
{scale} {scale} scale

% Set stroke properties
{stroke_width} setlinewidth
0 setgray  % Black color
[] 0 setdash  % Solid line

% Draw the actual design from SVG
"""

        # Add each path to the AI file
        for i, path_info in enumerate(path_data_list):
            # Extract path data and fill information
            if isinstance(path_info, tuple):
                path_data, fill_value = path_info
                is_filled = bool(fill_value and fill_value != 'none')
            else:
                path_data = path_info
                is_filled = False  # Default to stroke for backward compatibility
            
            ai_content += f"\n% Path {i+1} (filled: {is_filled})\n"
            ai_content += convert_svg_path_to_postscript(path_data, is_filled)
        
        ai_content += """
showpage
%%Trailer
%%EOF
"""
        
        # Save the AI file
        with open(ai_output_path, 'w', encoding='utf-8') as f:
            f.write(ai_content)
        
        return ai_output_path
        
    except Exception as e:
        st.error(f"AI file creation error: {e}")
        return None

def convert_svg_path_to_postscript(path_data, is_filled=False):
    """Convert SVG path data to PostScript commands - IMPROVED VERSION"""
    try:
        ps_commands = []
        commands = path_data.split()
        
        i = 0
        current_x, current_y = 0, 0
        
        while i < len(commands):
            cmd = commands[i]
            
            if cmd in ['M', 'm']:  # Move to
                if i + 2 < len(commands):
                    x, y = float(commands[i+1]), float(commands[i+2])
                    if cmd == 'm':  # Relative move
                        current_x += x
                        current_y += y
                    else:  # Absolute move
                        current_x, current_y = x, y
                    
                    ps_commands.append(f"newpath\n{current_x} {current_y} moveto")
                    i += 3
                    
            elif cmd in ['L', 'l']:  # Line to
                if i + 2 < len(commands):
                    x, y = float(commands[i+1]), float(commands[i+2])
                    if cmd == 'l':  # Relative line
                        current_x += x
                        current_y += y
                    else:  # Absolute line
                        current_x, current_y = x, y
                    
                    ps_commands.append(f"{current_x} {current_y} lineto")
                    i += 3
                    
            elif cmd in ['Z', 'z']:  # Close path
                ps_commands.append("closepath")
                # Handle fill vs stroke based on path type
                if is_filled:
                    ps_commands.append("fill")  # Fill the closed path
                else:
                    ps_commands.append("stroke")  # Stroke the closed path
                i += 1
                
            else:
                # Handle coordinates without explicit commands
                try:
                    # Try to parse as coordinate pair
                    x = float(cmd)
                    if i + 1 < len(commands):
                        y = float(commands[i+1])
                        current_x, current_y = x, y
                        ps_commands.append(f"{current_x} {current_y} lineto")
                        i += 2
                    else:
                        i += 1
                except:
                    i += 1
        
        return "\n".join(ps_commands) + "\n"
        
    except Exception as e:
        return f"% Error converting path: {e}\n"

def create_simple_ai_with_design(svg_path, ai_output_path):
    """Create simple AI file with properly scaled design"""
    try:
        # Read SVG content
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Extract path data and fill information
        path_data_list = extract_svg_path_data(svg_content)
        
        # Convert to list of path data strings for backward compatibility in other functions
        path_strings = [path[0] if isinstance(path, tuple) else path for path in path_data_list]
        
        if not path_data_list:
            return None
        
        # Calculate bounding box
        min_x, min_y, max_x, max_y = calculate_bounding_box(path_strings)
        design_width = max_x - min_x
        design_height = max_y - min_y
        
        # Page dimensions
        page_width, page_height = 612, 792
        
        # Calculate scaling and translation
        scale = min((page_width - 100) / design_width, (page_height - 100) / design_height, 1.5)
        translate_x = (page_width - design_width * scale) / 2
        translate_y = (page_height - design_height * scale) / 2
        
        # Create AI content
        ai_content = f"""%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 {page_width} {page_height}
%%Title: {os.path.basename(svg_path)}
%%Creator: Fashion Design Studio
%%CreationDate: {time.strftime("%Y-%m-%d %H:%M:%S")
}
%%EndComments

% Design centered and scaled properly
{translate_x} {translate_y} translate
{scale} {scale} scale
{-min_x} {-min_y} translate

1 setlinewidth
0 setgray
[] 0 setdash

% Design paths
"""
        
        # Add paths
        for path_info in path_data_list:
            # Extract path data and fill information
            if isinstance(path_info, tuple):
                path_data, fill_value = path_info
                is_filled = bool(fill_value and fill_value != 'none')
            else:
                path_data = path_info
                is_filled = False  # Default to stroke for backward compatibility
            
            ai_content += f"% Path (filled: {is_filled})\n"
            ai_content += convert_svg_path_to_postscript(path_data, is_filled)
        
        ai_content += "showpage\n%%EOF"
        
        with open(ai_output_path, 'w', encoding='utf-8') as f:
            f.write(ai_content)
        
        return ai_output_path
        
    except Exception as e:
        st.error(f"Simple AI creation error: {e}")
        return None

def create_working_ai_file(svg_path, ai_output_path):
    """Create a working AI file with properly scaled design"""
    try:
        return create_simple_ai_with_design(svg_path, ai_output_path)
    except Exception as e:
        st.error(f"Working AI creation error: {e}")
        return None

def display_svg_preview(svg_path):
    """Display SVG preview"""
    try:
        with open(svg_path, "r", encoding="utf-8") as file:
            svg_content = file.read()
        
        st.markdown("### SVG Preview")
        components.html(
            f"""
            <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; background: white; text-align: center;">
                {svg_content}
            </div>
            """,
            height=400
        )
        
    except Exception as e:
        st.error(f"Could not display SVG preview: {e}")

def main():
    st.set_page_config(
        page_title="Fashion Design Studio",
        page_icon="👗",
        layout="wide"
    )
    
    st.title("👗 Fashion Design Studio")
    st.markdown("Professional embroidery design studio")
    
    # Initialize systems (force reload if code changed)
    # Add a version check to force reload when code changes
    VECTORIZER_VERSION = "2.0"  # Increment this when you update vector_generator.py
    
    if 'vectorizer' not in st.session_state or st.session_state.get('vectorizer_version') != VECTORIZER_VERSION:
        with st.spinner("Loading design systems..."):
            st.session_state.vectorizer = EmbroideryVectorizer()
            st.session_state.vectorizer_version = VECTORIZER_VERSION
            # Removed tryon_system initialization
    
    # --- FIX: Initialize session state for results ---
    if 'vector_result' not in st.session_state:
        st.session_state.vector_result = None
    # Removed tryon_result initialization

    # --- FIX: Callbacks to clear results when files change ---
    def clear_vector_results():
        st.session_state.vector_result = None
        
    # Removed clear_tryon_results function

    # Create tabs - only vectorization tab now
    tab1 = st.tabs(["✏️ Vectorization"])[0]
    
    with tab1:
        st.header("Embroidery Vectorization")
        st.markdown("Convert hand drawings to vector files for embroidery machines")
        
        # Embroidery options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stroke_option = st.radio(
                "Stroke Option:",
                ["Single Stroke Width", "Multiple Stroke Widths"],
                help="Single: One SVG file\nMultiple: Multiple SVG files with different stroke widths"
            )
        
        with col2:
            if stroke_option == "Single Stroke Width":
                stroke_width = st.slider("Stroke Width (pixels):", 1, 10, 3)
            else:
                stroke_width = 3 # Default, not used directly
                st.info("Will generate SVG files with stroke widths: 1px, 2px, 3px, 4px")
        
        with col3:
            # Add smoothing level control
            smoothing_level = st.slider(
                "Curve Smoothness:",
                min_value=1,
                max_value=10,
                value=7,
                help="Lower: More angular lines\nHigher: Smoother curves"
            )
        
        # --- NEW: Export Format Selection ---
        st.subheader("Export Options")
        # Force export format to SVG + Adobe Illustrator (removed SVG Only option)
        export_format = "SVG + Adobe Illustrator"
        st.info("✅ Export format: SVG + Adobe Illustrator")
        
        # Add force close paths option
        force_close_all = st.checkbox(
            "Force Close All Paths",
            value=False,
            help="Automatically close all open paths (recommended for embroidery machines)"
        )
        
        uploaded_file = st.file_uploader(
            "Upload Hand Drawn Artwork", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
            help="Supported formats: JPG, PNG, BMP, TIFF, WebP",
            key="vector_upload",
            on_change=clear_vector_results # <-- FIX: Clear state on new upload
        )
        
        if uploaded_file is not None:
            # Sanitize the filename
            original_filename = uploaded_file.name
            sanitized_filename = sanitize_filename(original_filename)
            
            # Display file info
            file_details = {
                "Original Filename": original_filename,
                "Sanitized Filename": sanitized_filename,
                "File size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.write("File details:", file_details)
            
            # Display uploaded image
            st.image(uploaded_file, caption=f"Uploaded Artwork: {sanitized_filename}", width=400)
            
            # Live Preview Section
            st.subheader("🔍 Live Preview & Adjustment")
            
            # Detail level slider with live preview
            detail_level = st.slider(
                "Detail Level (Adjust to see live preview):",
                min_value=1,
                max_value=10,
                value=5,
                help="Lower: More details (more contours)\nHigher: Smoother (fewer contours)",
                key="detail_slider"
            )
            
            # Generate live preview when slider changes
            if 'last_detail_level' not in st.session_state or st.session_state.last_detail_level != detail_level:
                st.session_state.last_detail_level = detail_level
                
                with st.spinner("Generating preview..."):
                    file_extension = os.path.splitext(original_filename)[1].lower()
                    if file_extension == '':
                        file_extension = '.jpg'
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Store detail level
                        st.session_state.vectorizer.detail_level = detail_level
                        st.session_state.vectorizer.smoothing_level = smoothing_level
                        
                        # Preprocess image
                        processed_image = st.session_state.vectorizer.preprocess_hand_drawing(tmp_path)
                        
                        # Find contours for preview
                        st.session_state.vectorizer.current_binary = processed_image
                        contour_infos = st.session_state.vectorizer.find_contours_exact(processed_image, min_area=30)
                        
                        # Create preview visualization with white background
                        import cv2
                        import numpy as np
                        
                        # Create white background for better visibility
                        height, width = processed_image.shape
                        preview_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                        
                        # Draw contours in black with thicker lines
                        contours = [info['contour'] for info in contour_infos]
                        cv2.drawContours(preview_img, contours, -1, (0, 0, 0), 2)
                        
                        # Display preview
                        col_prev1, col_prev2 = st.columns(2)
                        with col_prev1:
                            st.image(processed_image, caption="Processed Image", use_container_width=True)
                        with col_prev2:
                            st.image(preview_img, caption=f"Vector Preview: {len(contours)} contours", use_container_width=True)
                        
                        st.success(f"✅ Preview generated with {len(contours)} contours at detail level {detail_level}")
                        
                    except Exception as e:
                        st.error(f"Preview error: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
            
            if st.button("Generate Vector File(s)", key="vector_btn", use_container_width=True):
                with st.spinner("Processing image..."):
                    file_extension = os.path.splitext(original_filename)[1].lower()
                    if file_extension == '':
                        file_extension = '.jpg'
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        output_dir = "outputs"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        base_name = sanitized_filename
                        if not base_name or base_name == "embroidery_design":
                            import datetime
                            base_name = f"design_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        result = None # Initialize result
                        ai_path = None # Initialize ai_path earlier to fix linter issue
                        
                        if stroke_option == "Single Stroke Width":
                            svg_output = os.path.join(output_dir, f"{base_name}_stroke_{stroke_width}px.svg")
                            svg_path, processed_image, contours = st.session_state.vectorizer.process_hand_artwork(
                                tmp_path, svg_output, stroke_width=stroke_width, detail_level=detail_level, smoothing_level=smoothing_level, force_close_all=force_close_all
                            )
                            
                            if svg_path:
                                is_valid, message = st.session_state.vectorizer.verify_svg_file(svg_path)
                                if not is_valid:
                                    st.warning(f"SVG file may have issues: {message}")
                                
                                # --- NEW: Generate AI file if requested ---
                                ai_path = None  # Initialize ai_path
                                if export_format == "SVG + Adobe Illustrator":
                                    ai_output = os.path.join(output_dir, f"{base_name}_stroke_{stroke_width}px.ai")
                                    # Try multiple methods to create AI file
                                    ai_path = create_proper_ai_file_with_design(svg_path, ai_output)
                                    if not ai_path:
                                        ai_path = create_working_ai_file(svg_path, ai_output)
                                    
                                    if ai_path and os.path.exists(ai_path):
                                        st.success(f"✅ Adobe Illustrator file created: {os.path.basename(ai_path)}")
                                    else:
                                        st.warning("⚠️ Could not create proper Adobe Illustrator file. SVG file is ready for download.")
                            
                            preprocessing_viz = st.session_state.vectorizer.visualize_preprocessing(
                                tmp_path,
                                processed_image,
                                os.path.join(output_dir, f"{base_name}_preprocessing.jpg")
                            )
                            
                            result = {
                                'preprocessing_viz': preprocessing_viz,
                                'svg_path': svg_path,
                                'ai_path': ai_path,  # <-- NEW: Add AI path
                                'contours': contours,
                                'multiple_files': False,
                                'base_name': base_name,
                                'stroke_width': stroke_width,
                                'export_format': export_format  # <-- NEW: Save export format
                            }
                        else:
                            # Generate multiple SVG files
                            svg_paths, processed_image, contours = st.session_state.vectorizer.process_hand_artwork_multiple_strokes(
                                tmp_path, output_dir, base_name, detail_level=detail_level, smoothing_level=smoothing_level, force_close_all=force_close_all
                            )
                            
                            # --- NEW: Generate AI files for multiple strokes if requested ---
                            ai_paths = []
                            if export_format == "SVG + Adobe Illustrator" and svg_paths:
                                for i, svg_path in enumerate(svg_paths):
                                    stroke_size = i + 1
                                    ai_output = os.path.join(output_dir, f"{base_name}_stroke_{stroke_size}px.ai")
                                    ai_path = create_proper_ai_file_with_design(svg_path, ai_output)
                                    if not ai_path:
                                        ai_path = create_working_ai_file(svg_path, ai_output)
                                    
                                    if ai_path and os.path.exists(ai_path):
                                        ai_paths.append(ai_path)
                                        st.success(f"✅ Adobe Illustrator file created for {stroke_size}px stroke")
                            
                            preprocessing_viz = st.session_state.vectorizer.visualize_preprocessing(
                                tmp_path,
                                processed_image,
                                os.path.join(output_dir, f"{base_name}_preprocessing.jpg")
                            )
                            
                            result = {
                                'preprocessing_viz': preprocessing_viz,
                                'svg_paths': svg_paths,
                                'ai_paths': ai_paths,  # <-- NEW: Add AI paths
                                'contours': contours,
                                'multiple_files': True,
                                'base_name': base_name,
                                'export_format': export_format  # <-- NEW: Save export format
                            }
                        
                        # --- FIX: Save result to session state ---
                        if result:
                            st.session_state.vector_result = result
                            st.success("Vector Files Generated Successfully!")
                        else:
                            st.session_state.vector_result = None
                            st.error("Failed to generate vector files.")
                            
                    except Exception as e:
                        st.session_state.vector_result = None
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

        # --- FIX: Display results and downloads OUTSIDE the button block ---
        if st.session_state.vector_result:
            result = st.session_state.vector_result
            
            # Display preprocessing results - FIXED: use_container_width
            if result.get('preprocessing_viz') and os.path.exists(result['preprocessing_viz']):
                st.image(result['preprocessing_viz'], caption="Processing Steps", use_container_width=True)
            
            # Display download options
            if result.get('multiple_files', False):
                st.subheader("Download Files")
                
                # SVG Files
                st.markdown("#### 📁 SVG Files (For Embroidery Machines)")
                st.info("✅ **Ready for embroidery machines - Multiple stroke width options**")
                cols_svg = st.columns(4)
                for i, svg_path in enumerate(result['svg_paths']):
                    with cols_svg[i]:
                        stroke_size = i + 1
                        if os.path.exists(svg_path):
                            with open(svg_path, "rb") as file:
                                download_filename = f"{result['base_name']}_stroke_{stroke_size}px.svg"
                                st.download_button(
                                    label=f"📥 Download SVG {stroke_size}px",
                                    data=file,
                                    file_name=download_filename,
                                    mime="image/svg+xml",
                                    use_container_width=True,
                                    key=f"download_svg_multi_{i}"
                                )
                                st.caption(f"Stroke: {stroke_size}px")
                        else:
                            st.error(f"File not found: {svg_path}")
                
                # Add a note about using SVG files
                st.caption("These SVG files can be used directly with embroidery machines. Different stroke widths are provided for various fabric types.")
                
                # Adobe Illustrator Files
                if result.get('ai_paths') and result['export_format'] == "SVG + Adobe Illustrator":
                    st.markdown("#### 🎨 Adobe Illustrator Files (For Professional Editing)")
                    st.success("✅ **Designs are properly centered and scaled**")
                    cols_ai = st.columns(4)
                    for i, ai_path in enumerate(result['ai_paths']):
                        with cols_ai[i]:
                            stroke_size = i + 1
                            if os.path.exists(ai_path):
                                with open(ai_path, "rb") as file:
                                    download_filename = f"{result['base_name']}_stroke_{stroke_size}px.ai"
                                    st.download_button(
                                        label=f"📥 Download AI {stroke_size}px",
                                        data=file,
                                        file_name=download_filename,
                                        mime="application/illustrator",
                                        use_container_width=True,
                                        key=f"download_ai_multi_{i}"
                                    )
                                    st.caption("For professional editing in Adobe Illustrator")
                            else:
                                st.warning(f"AI file not available for {stroke_size}px")
                    
            else:
                st.subheader("Download Files")
                
                # Single SVG File
                if result.get('svg_path') and os.path.exists(result['svg_path']):
                    download_stroke_width = result.get('stroke_width', 3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📁 SVG File (For Embroidery Machines)")
                        st.info("✅ **Ready for embroidery machines**")
                        with open(result['svg_path'], "rb") as file:
                            download_filename = f"{result['base_name']}_stroke_{download_stroke_width}px.svg"
                            st.download_button(
                                label=f"📥 Download SVG ({download_stroke_width}px)",
                                data=file,
                                file_name=download_filename,
                                mime="image/svg+xml",
                                use_container_width=True,
                                key="download_svg_single"
                            )
                        st.caption("This SVG file can be used directly with embroidery machines")
                        
                        # Add a preview of the SVG
                        if result.get('svg_path') and os.path.exists(result['svg_path']):
                            with st.expander("🔍 Preview SVG Design"):
                                display_svg_preview(result['svg_path'])
                    
                    # Single Adobe Illustrator File
                    with col2:
                        if result.get('ai_path') and result['export_format'] == "SVG + Adobe Illustrator":
                            st.markdown("#### 🎨 Adobe Illustrator File")
                            if os.path.exists(result['ai_path']):
                                st.success("✅ **Design properly centered and scaled**")
                                with open(result['ai_path'], "rb") as file:
                                    download_filename = f"{result['base_name']}_stroke_{download_stroke_width}px.ai"
                                    st.download_button(
                                        label=f"📥 Download AI ({download_stroke_width}px)",
                                        data=file,
                                        file_name=download_filename,
                                        mime="application/illustrator",
                                        use_container_width=True,
                                        key="download_ai_single"
                                    )
                                st.caption("Open in Adobe Illustrator - design will be centered")
                                

                            else:
                                st.warning("Adobe Illustrator file not available")
                        elif result['export_format'] == "SVG + Adobe Illustrator":
                            st.info("Adobe Illustrator file generation failed")
                        

                else:
                    st.error("SVG file not generated properly")

if __name__ == "__main__":
    main()