# 👗 Fashion Design Studio

Professional embroidery design vectorization tool that converts hand-drawn artwork into machine-ready vector files.

## Features

- **Image to Vector Conversion**: Convert hand drawings to SVG format for embroidery machines
- **Multiple Stroke Widths**: Generate designs with different stroke widths (1px, 2px, 3px, 4px)
- **Adobe Illustrator Export**: Export designs as AI files for professional editing
- **Live Preview**: Real-time preview with adjustable detail and smoothness levels
- **Smart Path Processing**: Automatic path closing and smoothing for embroidery machines
- **Multiple Format Support**: JPG, PNG, BMP, TIFF, WebP

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-design-studio.git
cd fashion-design-studio
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. Upload your hand-drawn artwork (JPG, PNG, BMP, TIFF, or WebP)
2. Adjust the detail level and curve smoothness using the sliders
3. Preview the vectorization in real-time
4. Choose single or multiple stroke width options
5. Enable "Force Close All Paths" for embroidery machine compatibility
6. Click "Generate Vector File(s)" to create SVG and AI files
7. Download the generated files

## Output Files

- **SVG Files**: Ready for embroidery machines
- **AI Files**: For professional editing in Adobe Illustrator
- **Preprocessing Visualization**: Shows the conversion process

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- NumPy
- svgwrite
- matplotlib
- scikit-image
- scipy
- Pillow

## Project Structure

```
fashion-design-studio/
├── app.py                  # Main Streamlit application
├── vector_generator.py     # Core vectorization engine
├── verify_svg.py          # SVG verification utility
├── requirements.txt       # Python dependencies
├── outputs/               # Generated files directory
└── README.md             # This file
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
