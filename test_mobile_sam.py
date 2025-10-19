#!/usr/bin/env python3
"""
Test script to verify MobileSAM environment setup and inference on dog.jpg
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add mobile-sam to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mobile-sam'))

from mobile_sam import sam_model_registry, SamPredictor

def test_mobile_sam_inference():
    """Test MobileSAM inference on dog.jpg"""
    
    print("=" * 60)
    print("MobileSAM Environment Test")
    print("=" * 60)
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    # Load image
    image_path = "images/dog.jpg"
    if not os.path.exists(image_path):
        print(f"\n✗ Error: Image not found at {image_path}")
        return False
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"\n✓ Image loaded: {image_np.shape}")
    
    # Check if model weights exist
    model_path = "sam-convert/mobile_sam.pt"
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model weights not found at {model_path}")
        print("  Please download from: https://github.com/ChaoningZhang/MobileSAM/tree/master/weights")
        return False
    
    print(f"✓ Model weights found: {model_path}")
    
    # Load MobileSAM model
    print("\nLoading MobileSAM model...")
    try:
        sam = sam_model_registry["vit_t"](checkpoint=model_path)
        sam.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Create predictor
    predictor = SamPredictor(sam)
    
    # Set image
    print("\nProcessing image...")
    predictor.set_image(image_np)
    print("✓ Image processed and features extracted")
    
    # Test inference with a point prompt (center of image)
    h, w = image_np.shape[:2]
    input_point = np.array([[w//2 - 100, h//2 + 50]])
    input_label = np.array([1])  # foreground point
    
    print(f"\nRunning inference with point prompt: {input_point[0]}")
    
    try:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        print(f"✓ Inference successful!")
        print(f"  Generated {len(masks)} masks")
        print(f"  Mask shapes: {masks[0].shape}")
        print(f"  Scores: {scores}")
        
        # Select best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        print(f"\n✓ Best mask (index {best_mask_idx}): score = {best_score:.3f}")
        print(f"  Mask covers {best_mask.sum() / best_mask.size * 100:.1f}% of image")
        
        # Save visualization
        output_path = "target/dog_mobile_sam_test.png"
        os.makedirs("target", exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with point
        axes[0].imshow(image_np)
        axes[0].scatter(input_point[0, 0], input_point[0, 1], c='red', s=100, marker='*')
        axes[0].set_title("Input: Dog with point prompt")
        axes[0].axis('off')
        
        # Masked result
        axes[1].imshow(image_np)
        axes[1].imshow(best_mask, alpha=0.5, cmap='jet')
        axes[1].scatter(input_point[0, 0], input_point[0, 1], c='red', s=100, marker='*')
        axes[1].set_title(f"Output: Best mask (score={best_score:.3f})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! MobileSAM environment is properly setup.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_mobile_sam_inference()
    sys.exit(0 if success else 1)

