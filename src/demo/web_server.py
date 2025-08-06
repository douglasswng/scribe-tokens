from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_schema import DigitalInk
from enhancer.main import Enhancer


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the enhancer
enhancer = Enhancer()


@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory(os.path.dirname(__file__), 'web_demo.html')


@app.route('/enhance', methods=['POST'])
def enhance_strokes():
    """Enhance digital ink strokes"""
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'strokes' not in data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No strokes data provided'
            }), 400
        
        strokes = data['strokes']
        text = data['text']
        
        if not strokes or not text:
            return jsonify({
                'success': False,
                'error': 'No strokes to enhance'
            }), 400
        
        # Convert to DigitalInk format
        # The strokes come as a list of lists of [x, y] coordinates
        # We need to convert them to tuples for DigitalInk.from_coords
        formatted_strokes = []
        for stroke in strokes:
            if len(stroke) > 1:  # Only include strokes with more than one point
                formatted_stroke = [(float(point[0]), float(point[1])) for point in stroke]
                formatted_strokes.append(formatted_stroke)
        
        if not formatted_strokes:
            return jsonify({
                'success': False,
                'error': 'No valid strokes found'
            }), 400
        
        # Create DigitalInk object
        digital_ink = DigitalInk.from_coords(formatted_strokes, to_origin=False)
        
        # Run enhancement
        enhanced_ink = enhancer.enhance(digital_ink, text)
        
        # Convert enhanced ink back to JSON-serializable format
        enhanced_data = {
            'strokes': []
        }
        
        for stroke in enhanced_ink.strokes:
            stroke_data = {
                'points': [{'x': point.x, 'y': point.y} for point in stroke.points]
            }
            enhanced_data['strokes'].append(stroke_data)
        
        return jsonify({
            'success': True,
            'enhanced_ink': enhanced_data
        })
        
    except Exception as e:
        # Log the full traceback for debugging
        print("Error in enhance_strokes:")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("Starting Digital Ink Web Demo...")
    print("Open your browser and go to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002) 