# RealESRGAN_Distillation/train_distill.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms # Use transforms for potential preprocessing
import os
import time
import argparse # For command-line arguments (optional but good practice)

# Import custom components
from archs.rrdbnet_arch import RRDBNet
from dataset import UnpairedLRDataset
from utils import FeatureExtractor

# Function to handle None values in a batch (if dataset.__getitem__ returns None)
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed
    return torch.utils.data.dataloader.default_collate(batch)


def train(args):
    """Main training function"""

    # --- Configuration & Setup ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Student models will be saved to: {output_dir}")

    # --- Teacher Model ---
    print("Loading Teacher Model...")
    # Define architecture matching the pre-trained weights
    # Adjust num_feat, num_block based on the specific RealESRGAN model variant
    teacher_model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=args.teacher_num_feat,
        num_block=args.teacher_num_block,
        num_grow_ch=32,
        scale=args.scale
    )

    # Load the pre-trained weights
    try:
        load_net = torch.load(args.teacher_pth_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
         print(f"ERROR: Teacher model PTH file not found at: {args.teacher_pth_path}")
         print("Make sure the path is correct relative to where you run the script.")
         return
    except Exception as e:
        print(f"Error loading teacher model weights: {e}")
        return


    # Find the generator parameters (adjust keys if needed)
    if 'params_ema' in load_net:
        params_key = 'params_ema'
    elif 'params' in load_net:
        params_key = 'params'
    else:
        # Try loading directly if it's just the state_dict
        try:
            teacher_model.load_state_dict(load_net, strict=True)
            params_key = None # Indicate weights loaded directly
            print("Teacher weights loaded directly from state_dict.")
        except Exception as direct_load_e:
            print(f"ERROR: Cannot find generator parameters (keys like 'params_ema', 'params') in '{args.teacher_pth_path}' and direct loading failed: {direct_load_e}")
            print("Check the structure of your .pth file.")
            return

    if params_key:
        teacher_model.load_state_dict(load_net[params_key], strict=True)
        print(f"Teacher weights loaded from key '{params_key}'.")

    teacher_model.eval()                # Set teacher to evaluation mode
    teacher_model.requires_grad_(False) # Freeze teacher parameters
    teacher_model.to(device)
    print(f"Teacher Model parameters: {sum(p.numel() for p in teacher_model.parameters())}")


    # --- Student Model ---
    print("Initializing Student Model...")
    # Define student architecture (can be same RRDBNet but smaller)
    student_model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=args.student_num_feat,
        num_block=args.student_num_block,
        num_grow_ch=32, # Can also adjust this
        scale=args.scale
    )
    student_model.train() # Set student to training mode
    student_model.to(device)
    print(f"Student Model parameters: {sum(p.numel() for p in student_model.parameters())}")
    print(f"Training student model with num_feat={args.student_num_feat}, num_block={args.student_num_block}")


    # --- Feature Layer Names ---
    # ** CRITICAL: ** Inspect your models (print(teacher_model), print(student_model))
    # or use model.named_modules() to get the exact layer names.
    teacher_layer_names = args.teacher_layers
    student_layer_names = args.student_layers
    if len(teacher_layer_names) != len(student_layer_names):
        print("ERROR: Number of teacher layers and student layers for distillation must match!")
        return
    if not teacher_layer_names:
         print("ERROR: No layers specified for feature distillation.")
         return
    print(f"Distilling from Teacher layers: {teacher_layer_names}")
    print(f"Matching to Student layers: {student_layer_names}")


    # --- Dataset and DataLoader ---
    print(f"Loading dataset from: {args.lr_data_dir}")
    # Add any desired transforms (e.g., normalization, random crop if input size varies)
    # For simplicity, starting with just ToTensor
    train_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.RandomCrop(args.patch_size), # Add if you want fixed size patches
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Example norm
    ])

    dataset = UnpairedLRDataset(
        lr_dir=args.lr_data_dir,
        transform=train_transform,
        max_samples=args.max_samples # Use subset if specified
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, # Can speed up data transfer to GPU
        collate_fn=safe_collate # Handles None returns from dataset
    )
    print(f"Dataset size: {len(dataset)} images.")


    # --- Loss and Optimizer ---
    feature_loss_fn = nn.L1Loss().to(device) # Or nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate)

    # --- Training Loop ---
    print("Starting training loop...")
    start_time = time.time()
    total_iterations = len(dataloader) * args.epochs
    current_iter = 0

    for epoch in range(1, args.epochs + 1):
        student_model.train() # Ensure student is in train mode each epoch
        epoch_loss = 0.0
        batches_processed = 0

        for batch_idx, lr_batch in enumerate(dataloader):

            current_iter += 1
            if lr_batch is None: # Skip batch if collate_fn filtered it
                 print(f"Warning: Skipping empty batch at epoch {epoch}, index {batch_idx}")
                 continue

            lr_batch = lr_batch.to(device)
            optimizer.zero_grad()

            # --- Feature Extraction (using context managers for safety) ---
            teacher_features = {}
            student_features = {}

            # Get Teacher Features (no gradients needed)
            try:
                with torch.no_grad():
                    with FeatureExtractor(teacher_model, teacher_layer_names) as teacher_extractor:
                        _ = teacher_model(lr_batch) # Forward pass to trigger hooks
                        teacher_features = teacher_extractor.get_features_dict()
            except Exception as e:
                 print(f"ERROR during teacher feature extraction: {e}")
                 continue # Skip this batch

            # Get Student Features (gradients needed for student)
            try:
                with FeatureExtractor(student_model, student_layer_names) as student_extractor:
                    # Run student forward pass - retain graph for backprop
                    _ = student_model(lr_batch) # Result might be used for other losses later
                    student_features = student_extractor.get_features_dict()
            except Exception as e:
                 print(f"ERROR during student feature extraction: {e}")
                 continue # Skip this batch

            # --- Calculate Feature Distillation Loss ---
            loss_feat = 0
            valid_pairs = 0
            # Iterate based on the requested layer names to ensure order/matching
            for t_layer_name, s_layer_name in zip(teacher_layer_names, student_layer_names):
                if t_layer_name in teacher_features and s_layer_name in student_features:
                    t_feat = teacher_features[t_layer_name]
                    s_feat = student_features[s_layer_name]

                    # Basic shape check (can be made more sophisticated)
                    if t_feat.shape == s_feat.shape:
                        loss_feat += feature_loss_fn(s_feat, t_feat)
                        valid_pairs += 1
                    else:
                        print(f"Warning: Shape mismatch between teacher layer '{t_layer_name}' ({t_feat.shape})"
                              f" and student layer '{s_layer_name}' ({s_feat.shape}). Skipping pair.")
                else:
                     print(f"Warning: Could not find extracted features for teacher layer '{t_layer_name}'"
                           f" or student layer '{s_layer_name}'. Skipping pair.")


            if valid_pairs == 0:
                 print("Warning: No valid feature pairs found for loss calculation in this batch. Skipping update.")
                 continue

            # Normalize loss by number of valid pairs compared
            loss_feat = loss_feat / valid_pairs

            total_loss = args.lambda_feature * loss_feat
            # --- (Optional) Add other losses here (e.g., pixel loss vs bicubic) ---
            # loss_pix = pixel_loss_fn(...)
            # total_loss += lambda_pix * loss_pix

            # --- Backpropagation ---
            try:
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                batches_processed += 1
            except Exception as e:
                 print(f"ERROR during backward pass or optimizer step: {e}")
                 continue # Skip this batch


            # --- Logging ---
            if (current_iter % args.log_freq) == 0:
                elapsed_time = time.time() - start_time
                iters_left = total_iterations - current_iter
                time_per_iter = elapsed_time / current_iter if current_iter > 0 else 0
                eta = iters_left * time_per_iter
                print(f"Epoch:[{epoch}/{args.epochs}] Iter:[{current_iter}/{total_iterations}] "
                      f"Loss:{total_loss.item():.4f} (Feat:{loss_feat.item():.4f}) "
                      f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                      f"ETA:{time.strftime('%H:%M:%S', time.gmtime(eta))}")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / batches_processed if batches_processed > 0 else 0
        print(f"--- Epoch {epoch} Finished. Average Loss: {avg_epoch_loss:.4f} ---")

        # --- Save Checkpoint ---
        if (epoch % args.save_freq) == 0:
            save_path = os.path.join(output_dir, f'student_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args # Save args used for this training run
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

    # --- Training Finished ---
    elapsed_total_time = time.time() - start_time
    print(f"\nTraining finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_total_time))}.")
    final_save_path = os.path.join(output_dir, 'student_model_final.pth')
    torch.save({
            'epoch': args.epochs,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
             'args': args
        }, final_save_path)
    print(f"Final student model saved to {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Supervised SR via Feature Distillation')

    # --- Paths ---
    parser.add_argument('--teacher_pth_path', type=str, required=True, help='Path to the pre-trained teacher model (.pth file)')
    parser.add_argument('--lr_data_dir', type=str, required=True, help='Directory containing low-resolution training images')
    parser.add_argument('--output_dir', type=str, default='./trained_student_models', help='Directory to save trained student models')

    # --- Model Config ---
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor (usually 4)')
    parser.add_argument('--teacher_num_feat', type=int, default=64, help='Number of features in the teacher model')
    parser.add_argument('--teacher_num_block', type=int, default=23, help='Number of RRDB blocks in the teacher model')
    parser.add_argument('--student_num_feat', type=int, default=32, help='Number of features in the student model')
    parser.add_argument('--student_num_block', type=int, default=10, help='Number of RRDB blocks in the student model')

    # --- Distillation Layers (CRITICAL!) ---
    parser.add_argument('--teacher_layers', nargs='+', required=True, help='List of exact layer names from teacher model for distillation (e.g., body.5.rdb1.conv1)')
    parser.add_argument('--student_layers', nargs='+', required=True, help='List of corresponding layer names from student model')

    # --- Training Params ---
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--lambda_feature', type=float, default=1.0, help='Weight for the feature distillation loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    # parser.add_argument('--patch_size', type=int, default=64, help='Size of patches if using RandomCrop transform')

    # --- Logging & Saving ---
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency (iterations) to print training logs')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency (epochs) to save checkpoints')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size for debugging (e.g., 1000)')


    args = parser.parse_args()

    # --- Run Training ---
    train(args)