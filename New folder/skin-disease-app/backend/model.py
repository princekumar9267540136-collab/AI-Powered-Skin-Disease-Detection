import sys
import os
import traceback
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

try:
    # Debug: Print current working directory and arguments
    print(f"Debug: Current working directory: {os.getcwd()}", file=sys.stderr)
    print(f"Debug: Script location: {__file__}", file=sys.stderr)
    print(f"Debug: Command line arguments: {sys.argv}", file=sys.stderr)
    
    # Load the model from the exact path
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'skin_disease_model.h5')
    
    print(f"Debug: Attempting to load model from: {model_path}", file=sys.stderr)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        print(f"Debug: Current directory contents:", file=sys.stderr)
        try:
            print("\n".join(os.listdir(model_dir)), file=sys.stderr)
        except Exception as e:
            print(f"Error listing directory: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        print("Debug: Loading model...", file=sys.stderr)
        model = load_model(model_path)
        print("Debug: Model loaded successfully!", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Model loading verification complete

    # Get and verify image path
    if len(sys.argv) < 2:
        print("Error: No image path provided", file=sys.stderr)
        sys.exit(1)
        
    img_path = os.path.abspath(sys.argv[1])
    print(f"Debug: Image path received: {img_path}", file=sys.stderr)
    
    if not os.path.exists(img_path):
        print(f"Error: Image file not found: {img_path}", file=sys.stderr)
        print(f"Debug: Checking parent directory contents:", file=sys.stderr)
        parent_dir = os.path.dirname(img_path)
        try:
            print("\n".join(os.listdir(parent_dir)), file=sys.stderr)
        except Exception as e:
            print(f"Error listing directory: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Debug: Image file exists at: {img_path}", file=sys.stderr)

    # Try to preprocess a small portion of the image first
    try:
        print("Debug: Starting image preprocessing test", file=sys.stderr)
        img = image.load_img(img_path, target_size=(96, 96))
        print("Debug: Image loaded successfully", file=sys.stderr)
        
        # Just do a small test prediction to verify
        test_array = np.zeros((1, 18432))  # Create a dummy array of the right size
        print("Debug: Running test prediction", file=sys.stderr)
        try:
            model.predict(test_array, verbose=0)
            print("Debug: Test prediction successful", file=sys.stderr)
        except Exception as e:
            print(f"Debug: Test prediction failed: {str(e)}", file=sys.stderr)
            raise
            
        # If we got here, everything works, return a test result
        print("Debug: All tests passed", file=sys.stderr)
        print("Normal")  # Simple test output
        
    except Exception as e:
        print(f"Debug: Image processing/prediction test failed: {str(e)}", file=sys.stderr)
        raise

except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    print("Stack trace:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
