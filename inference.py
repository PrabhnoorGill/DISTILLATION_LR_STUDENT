import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2 # Using OpenCV for consistent image loading/saving if needed
import time # For timing inference

# --- Add paths to model definitions ---
# Assuming standard structure where run_inference.py is one level above 'model' and 'archs'
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to sys.path to find student.py
if script_dir not in sys.path:
    sys.path.append(script_dir)
# Add the parent directory if model/ and archs/ are relative to the script's dir
# (Adjust if your structure differs)
base_dir = script_dir # Assume model/ and archs/ are in the same dir as script
if base_dir not in sys.path:
     sys.path.append(base_dir)


# --- Import Model Architectures ---
try:
    # Import the Net class from the student.py file you created
    from student import Net as StudentModelArchitectureClass
    print("Successfully imported Student model architecture class (Net from student.py)")
except ImportError as e:
    print(f"Error importing Student model architecture (Net from student.py): {e}")
    print(f"Ensure 'student.py' is in the correct location ('{script_dir}')")
    print(f"Also ensure the 'model' directory with 'ops.py' is accessible from '{base_dir}'. Check PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during student model import: {e}")
    sys.exit(1)


try:
    # Teacher model architecture
    from archs.rrdbnet_arch import RRDBNet # Import teacher model definition
    print("Successfully imported Teacher model architecture class (RRDBNet)")
except ImportError as e:
    print(f"Error importing Teacher model architecture (RRDBNet): {e}")
    print(f"Ensure the 'archs' directory with 'rrdbnet_arch.py' is accessible from '{base_dir}'. Check PYTHONPATH.")
    sys.exit(1)
except ModuleNotFoundError as e:
     print(f"Error importing Teacher model architecture (RRDBNet): {e}")
     print(f"Ensure the 'archs' directory exists relative to the script. Check PYTHONPATH.")
     sys.exit(1)

# --- Helper Functions --- (Mostly unchanged from previous versions)

def load_image_pil(image_path):
    """Loads an image using PIL."""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(img_pil, device):
    """Converts PIL image to tensor, adds batch dim, moves to device."""
    if img_pil is None: return None
    # Check if student model sub_mean/add_mean handles this
    # For now, just ToTensor assuming model handles mean internally
    transform = transforms.Compose([
        transforms.ToTensor() # Converts to [C, H, W] and scales to [0.0, 1.0]
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # Add batch dim [B, C, H, W]
    return img_tensor

def postprocess_output(tensor, add_mean_layer=None):
    """Converts output tensor back to a NumPy image (H, W, C) uint8."""
    if tensor is None: return None
    # Apply add_mean if provided (for student model)
    if add_mean_layer is not None:
        tensor = add_mean_layer(tensor)

    output_img = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    output_img = (output_img * 255.0).round().astype(np.uint8)
    output_img = np.transpose(output_img, (1, 2, 0)) # HWC
    return output_img

def load_model(model_path, model_instance, device, is_teacher=False):
    """Loads state dict into model instance."""
    model_type = "Teacher" if is_teacher else "Student"
    if model_instance is None: # Check if instantiation failed
        print(f"{model_type} model instance is None, cannot load weights.")
        return None
    try:
        load_net = torch.load(model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
         print(f"ERROR: {model_type} weights file not found at: {model_path}")
         return None
    except Exception as e:
        print(f"Error loading {model_type} weights from {model_path}: {e}")
        return None

    # Determine the actual state dictionary
    state_dict = None
    if is_teacher:
        if 'params_ema' in load_net: params_key = 'params_ema'
        elif 'params' in load_net: params_key = 'params'
        else: params_key = None

        if params_key:
            state_dict = load_net[params_key]
            print(f"Teacher weights loaded from key '{params_key}'.")
        elif isinstance(load_net, dict) and all(isinstance(k, str) for k in load_net.keys()):
            state_dict = load_net
            print("Teacher weights loaded directly from state_dict.")
        else:
            print(f"ERROR: Cannot find parameters ('params_ema'/'params') in teacher '{model_path}' and it's not a raw state_dict.")
            return None
    else: # Student model
         # LESRCNN often saves the whole model or just state_dict
         if isinstance(load_net, nn.Module): # Check if it's the whole model
             state_dict = load_net.state_dict()
             print("Student weights loaded from saved model instance.")
         elif isinstance(load_net, dict) and 'state_dict' in load_net: # Check for common 'state_dict' key
             state_dict = load_net['state_dict']
             print("Student weights loaded from 'state_dict' key.")
         elif isinstance(load_net, dict) and all(isinstance(k, str) for k in load_net.keys()): # Assume it's just the state_dict
            state_dict = load_net
            print("Student weights loaded directly from state_dict.")
         else:
            print(f"ERROR: Could not interpret student weights file '{model_path}'. Expected a model instance, dict with 'state_dict', or raw state_dict.")
            return None

    if state_dict is None:
        print(f"ERROR: Failed to extract state_dict for {model_type} model.")
        return None

    # Clean keys (remove 'module.' prefix if present)
    cleaned_state_dict = {}
    has_module_prefix = False
    # Also handle potential 'model.' prefix if saved differently
    prefixes_to_remove = ['module.', 'model.']
    for k, v in state_dict.items():
        original_key = k
        cleaned_key = k
        for prefix in prefixes_to_remove:
             if k.startswith(prefix):
                 cleaned_key = k[len(prefix):]
                 has_module_prefix = True # Use same flag for simplicity
                 break # Remove only one prefix if nested (unlikely)
        cleaned_state_dict[cleaned_key] = v

    if has_module_prefix:
        print("Removed 'module.' or 'model.' prefix from state_dict keys.")

    # Load the state dict
    try:
        # Use strict=False initially if unsure about keys, but True is better
        missing_keys, unexpected_keys = model_instance.load_state_dict(cleaned_state_dict, strict=False)
        if unexpected_keys:
            print(f"Warning: Unexpected keys found in checkpoint: {unexpected_keys}")
        if missing_keys:
            print(f"Warning: Missing keys in model state_dict: {missing_keys}")
        print(f"Successfully loaded weights into {model_type} model ({model_instance.__class__.__name__})")
    except Exception as e:
        print(f"ERROR loading state_dict into {model_type} model ({model_instance.__class__.__name__}): {e}")
        print("--- Model keys (first 10):")
        print(list(model_instance.state_dict().keys())[:10])
        print("--- Checkpoint keys (first 10):")
        print(list(cleaned_state_dict.keys())[:10])
        return None

    model_instance.eval()
    model_instance.to(device)
    return model_instance

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference comparison between Student and Teacher SR models')

    # --- Paths ---
    parser.add_argument('--lr_image', type=str, required=True, help='Path to the low-resolution input image')
    parser.add_argument('--student_pth', type=str, required=True, help='Path to the student model weights (student.pth)')
    parser.add_argument('--teacher_pth', type=str, required=True, help='Path to the pre-trained teacher model weights (.pth)')
    parser.add_argument('--hr_image', type=str, default=None, help='(Optional) Path to the high-resolution ground truth image for metrics')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save output images and comparison')

    # --- Model Config ---
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor (must match models)')
    # Student specific args (check student.py Net.__init__)
    parser.add_argument('--student_multi_scale', action='store_true', help='Set student model to multi-scale mode (if applicable)')
    parser.add_argument('--student_group', type=int, default=1, help='Group argument for student model blocks')

    # Teacher (RealESRGAN) specific args (MUST match the teacher .pth file)
    parser.add_argument('--teacher_num_feat', type=int, default=64, help='Number of features in the teacher RRDBNet model')
    parser.add_argument('--teacher_num_block', type=int, default=23, help='Number of RRDB blocks in the teacher RRDBNet model')
    parser.add_argument('--teacher_num_grow_ch', type=int, default=32, help='Growth channel in teacher RRDBNet')

    # --- System ---
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Load Student Model (using Net from student.py) ---
    print("\nLoading Student Model...")
    student_model = None
    try:
        # Instantiate the imported student model class 'Net'
        # Pass necessary arguments from args based on Net.__init__
        student_model = StudentModelArchitectureClass(
            scale=args.scale,
            multi_scale=args.student_multi_scale,
            group=args.student_group
        )
        print(f"Instantiated Student model ({student_model.__class__.__name__}) with scale={args.scale}, multi_scale={args.student_multi_scale}, group={args.student_group}")
        # Keep add_mean layer separate for postprocessing if needed
        student_add_mean = student_model.add_mean.to(device)
    except TypeError as e:
        print(f"Error instantiating Student model ({StudentModelArchitectureClass.__name__}): {e}")
        print(f"Please check the {StudentModelArchitectureClass.__name__} class definition in student.py for required arguments.")
        sys.exit(1)
    except NameError:
        print("Student model architecture class not found. Check imports and sys.path.")
        sys.exit(1)
    except AttributeError as e:
         print(f"Error accessing attribute during student instantiation (maybe 'add_mean'?): {e}")
         sys.exit(1)


    student_model = load_model(args.student_pth, student_model, device, is_teacher=False)
    if student_model is None:
        print("Failed to load student model. Exiting.")
        sys.exit(1)

    # --- Load Teacher Model ---
    print("\nLoading Teacher Model...")
    teacher_model = None
    try:
        teacher_model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=args.teacher_num_feat, num_block=args.teacher_num_block,
            num_grow_ch=args.teacher_num_grow_ch, scale=args.scale
        )
        print(f"Instantiated Teacher model ({teacher_model.__class__.__name__})")
    except NameError:
         print("Teacher model architecture class (RRDBNet) not found. Check imports and sys.path.")
         sys.exit(1)
    except Exception as e:
         print(f"Error instantiating Teacher model (RRDBNet): {e}")
         sys.exit(1)

    teacher_model = load_model(args.teacher_pth, teacher_model, device, is_teacher=True)
    if teacher_model is None:
        print("Failed to load teacher model. Exiting.")
        sys.exit(1)

    # --- Load and Preprocess Image ---
    print(f"\nLoading LR image: {args.lr_image}")
    lr_img_pil = load_image_pil(args.lr_image)
    if lr_img_pil is None: sys.exit(1)

    lr_tensor = preprocess_image(lr_img_pil, device)
    if lr_tensor is None: sys.exit(1)

    hr_img_gt = None
    if args.hr_image:
        print(f"Loading HR image: {args.hr_image}")
        hr_img_gt_pil = load_image_pil(args.hr_image)
        if hr_img_gt_pil:
             hr_img_gt = np.array(hr_img_gt_pil) # HWC, RGB, uint8
        else:
             print("Warning: Could not load HR image, skipping metrics.")

    # --- Run Inference ---
    print("\nRunning inference...")
    student_output_tensor = None
    teacher_output_tensor = None
    student_time = -1
    teacher_time = -1

    with torch.no_grad():
        try:
            print("  Inferring with Student...")
            start = time.time()
            if device.type == 'cuda': torch.cuda.synchronize()
            # Use the specific forward signature for the student 'Net' model
            student_output_tensor = student_model(lr_tensor, scale=args.scale)
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            student_time = end - start
            print(f"  Student Inference Time: {student_time:.4f} s")
        except Exception as e:
            print(f"  ERROR during Student inference: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed traceback

        try:
            print("  Inferring with Teacher...")
            start = time.time()
            if device.type == 'cuda': torch.cuda.synchronize()
            teacher_output_tensor = teacher_model(lr_tensor)
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            teacher_time = end - start
            print(f"  Teacher Inference Time: {teacher_time:.4f} s")
        except Exception as e:
            print(f"  ERROR during Teacher inference: {e}")

    # --- Postprocess and Save Outputs ---
    print("\nPostprocessing outputs...")
    # Use the stored add_mean layer for the student model postprocessing
    student_sr_img = postprocess_output(student_output_tensor, student_add_mean)
    teacher_sr_img = postprocess_output(teacher_output_tensor) # Teacher likely doesn't need external add_mean

    # Save individual SR results
    basename = os.path.splitext(os.path.basename(args.lr_image))[0]
    if student_sr_img is not None:
        student_sr_path = os.path.join(args.output_dir, f"{basename}_student_x{args.scale}.png")
        try:
            Image.fromarray(student_sr_img).save(student_sr_path)
            print(f"  Saved Student SR image to: {student_sr_path}")
        except Exception as e: print(f"  ERROR saving student SR image: {e}")
    else: print("  Skipping saving student SR image.")

    if teacher_sr_img is not None:
        teacher_sr_path = os.path.join(args.output_dir, f"{basename}_teacher_x{args.scale}.png")
        try:
            Image.fromarray(teacher_sr_img).save(teacher_sr_path)
            print(f"  Saved Teacher SR image to: {teacher_sr_path}")
        except Exception as e: print(f"  ERROR saving teacher SR image: {e}")
    else: print("  Skipping saving teacher SR image.")

    # --- Comparison Plot --- (Rest of the script remains largely the same)
    print("\nGenerating comparison plot...")
    try:
        lr_img_np_hwc = np.array(lr_img_pil)
        lr_bicubic_h = lr_img_np_hwc.shape[0] * args.scale
        lr_bicubic_w = lr_img_np_hwc.shape[1] * args.scale
        lr_bicubic = cv2.resize(lr_img_np_hwc, (lr_bicubic_w, lr_bicubic_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"  Error creating bicubic image: {e}")
        lr_bicubic = None

    images_to_plot = []
    titles_to_plot = []

    # Prepare HR Ground Truth
    if hr_img_gt is not None:
        ref_shape = None
        if student_sr_img is not None: ref_shape = student_sr_img.shape
        elif teacher_sr_img is not None: ref_shape = teacher_sr_img.shape

        if ref_shape and hr_img_gt.shape[:2] != ref_shape[:2]:
             print(f"Warning: HR image shape {hr_img_gt.shape[:2]} differs from SR shape {ref_shape[:2]}. Resizing HR for metrics/plot.")
             try:
                hr_img_gt = cv2.resize(hr_img_gt, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
                images_to_plot.append(hr_img_gt)
                titles_to_plot.append('Ground Truth (HR)')
             except Exception as e:
                print(f"  Error resizing HR image: {e}")
                hr_img_gt = None # Can't use if resize fails
        elif hr_img_gt is not None: # If shapes already match or no ref_shape
             images_to_plot.append(hr_img_gt)
             titles_to_plot.append('Ground Truth (HR)')

    # Add other images if they exist
    if lr_bicubic is not None:
        images_to_plot.append(lr_bicubic)
        titles_to_plot.append('Bicubic')
    if student_sr_img is not None:
        images_to_plot.append(student_sr_img)
        titles_to_plot.append(f'Student\n{student_time:.3f}s')
    if teacher_sr_img is not None:
        images_to_plot.append(teacher_sr_img)
        titles_to_plot.append(f'Teacher\n{teacher_time:.3f}s')

    # Generate plot if images are available
    if not images_to_plot:
        print("  No images available to generate comparison plot.")
    else:
        num_cols = len(images_to_plot)
        fig_width = max(15, 5 * num_cols) # Ensure minimum width
        fig_height = fig_width / num_cols * 0.8 # Adjust height proportionally
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))
        if num_cols == 1: axes = [axes]

        for ax, img, title in zip(axes, images_to_plot, titles_to_plot):
            ax.imshow(img)
            ax.set_title(title, fontsize=10) # Adjust fontsize if needed
            ax.axis('off')

        plt.tight_layout(pad=0.5) # Add some padding
        comparison_path = os.path.join(args.output_dir, f"{basename}_comparison.png")
        try:
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight') # Use bbox_inches
            print(f"  Saved Comparison plot to: {comparison_path}")
        except Exception as e:
            print(f"  Error saving comparison plot: {e}")
        # plt.show()

    # --- Calculate Metrics ---
    if hr_img_gt is not None:
        print("\nCalculating metrics against Ground Truth...")
        data_range = 255

        if student_sr_img is not None:
            try:
                psnr_student = psnr(hr_img_gt, student_sr_img, data_range=data_range)
                win_size = min(7, hr_img_gt.shape[0]//2 * 2 + 1, hr_img_gt.shape[1]//2 * 2 + 1) # Ensure odd and <= image dim/2
                if win_size < 3: win_size = 3
                ssim_student = ssim(hr_img_gt, student_sr_img, data_range=data_range, channel_axis=-1, win_size=win_size, gaussian_weights=True)

                print(f"  Metrics for Student:")
                print(f"    PSNR: {psnr_student:.4f} dB")
                print(f"    SSIM: {ssim_student:.4f}")
            except Exception as e: print(f"  ERROR calculating metrics for Student: {e}")
        else: print("  Skipping metrics for Student.")

        if teacher_sr_img is not None:
            try:
                psnr_teacher = psnr(hr_img_gt, teacher_sr_img, data_range=data_range)
                win_size = min(7, hr_img_gt.shape[0]//2 * 2 + 1, hr_img_gt.shape[1]//2 * 2 + 1)
                if win_size < 3: win_size = 3
                ssim_teacher = ssim(hr_img_gt, teacher_sr_img, data_range=data_range, channel_axis=-1, win_size=win_size, gaussian_weights=True)

                print(f"  Metrics for Teacher:")
                print(f"    PSNR: {psnr_teacher:.4f} dB")
                print(f"    SSIM: {ssim_teacher:.4f}")
            except Exception as e: print(f"  ERROR calculating metrics for Teacher: {e}")
        else: print("  Skipping metrics for Teacher.")

    print("\nInference and comparison complete.")