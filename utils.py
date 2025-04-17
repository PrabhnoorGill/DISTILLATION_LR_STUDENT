# RealESRGAN_Distillation/utils.py

import torch
import torch.nn as nn
from collections import OrderedDict

class FeatureExtractor:
    """
    Helper class to extract features from intermediate layers of a PyTorch model.

    Usage:
        extractor = FeatureExtractor(model, ['layer1_name', 'layer2_name'])
        # Perform a forward pass (output might be discarded if only features are needed)
        model_output = model(input_tensor)
        # Retrieve the features captured during the forward pass
        features = extractor.get_features()
        # Important: Remove hooks when done to avoid memory leaks
        extractor.remove_hooks()

    Alternatively, manage hook lifetime within a context manager or explicitly
    call __call__ which encapsulates forward pass and feature retrieval.
    """

    def __init__(self, model: nn.Module, layer_names: list[str]):
        """
        Args:
            model (nn.Module): The PyTorch model to extract features from.
            layer_names (list[str]): A list of strings containing the exact names
                                      of the layers from which to extract features.
                                      Use model.named_modules() to find valid names.
        """
        self.model = model
        self.layer_names = set(layer_names) # Use set for faster lookup
        self._features = OrderedDict() # Store features keyed by layer name
        self._hooks = [] # Store hook handles for later removal

        self._register_hooks()

    def _hook_fn(self, name):
        """Closure to create a hook function for a specific layer."""
        def hook(module, input, output):
            # Detach output tensor to prevent gradients from flowing back unnecessarily
            # if the features are only used for loss calculation (not further network layers).
            # If features NEED grads (e.g. complex feature manipulation), remove .detach()
            self._features[name] = output # .detach() if features don't need grads

        return hook

    def _register_hooks(self):
        """Registers forward hooks on the specified layers."""
        # First, remove any existing hooks this instance might have registered
        self.remove_hooks()

        found_layers = 0
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(handle)
                found_layers += 1

        if found_layers != len(self.layer_names):
            print(f"Warning: FeatureExtractor expected {len(self.layer_names)} layers "
                  f"but found and registered hooks for {found_layers} layers.")
            registered = {h.name for h in self._hooks} # Assuming handle has name attr, crude check
            missing = self.layer_names - registered # Check which are potentially missing (crude)
            # A more robust check would need to track names associated with handles directly
            print(f" Ensure layer names match module names from model.named_modules(). Attempting registration for: {list(self.layer_names)}")


    def get_features(self) -> list[torch.Tensor]:
        """Returns the extracted features in the order they were requested (or hook execution order)."""
        # Note: Hook execution order isn't strictly guaranteed to be layer definition order.
        # Returning based on the requested layer_names order might be safer if needed,
        # but requires the initial list `self.layer_names_list` (not currently stored)
        # or sorting the OrderedDict keys based on some known sequence.
        # For now, returns based on internal OrderedDict storage order.
        return list(self._features.values())

    def get_features_dict(self) -> OrderedDict[str, torch.Tensor]:
         """Returns the extracted features as a dictionary keyed by layer name."""
         return self._features

    def remove_hooks(self):
        """Removes all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = [] # Clear the list of handles
        self._features = OrderedDict() # Clear stored features

    def __enter__(self):
        """Allows using the extractor with a 'with' statement."""
        # Reset features just in case
        self._features = OrderedDict()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up hooks when exiting a 'with' block."""
        self.remove_hooks()


# --- Example Usage (Do Not Include in the actual utils.py file, just for understanding) ---
# if __name__ == '__main__':
#     # Assuming RRDBNet is defined and imported
#     from archs.rrdbnet_arch import RRDBNet
#
#     # 1. Create a dummy model
#     dummy_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=2, num_grow_ch=4)
#
#     # 2. Define layer names to extract from (Use print(dummy_model) or dummy_model.named_modules() to find names)
#     layer_names_to_extract = ['body.0.rdb1.conv1', 'body.1.rdb1.conv1', 'conv_last']
#
#     # 3. Create a dummy input tensor
#     dummy_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
#
#     # --- Method 1: Manual hook management ---
#     print("--- Method 1: Manual Hook Management ---")
#     extractor_manual = FeatureExtractor(dummy_model, layer_names_to_extract)
#     output_manual = dummy_model(dummy_input) # Run forward pass
#     features_manual = extractor_manual.get_features_dict()
#     for name, feat in features_manual.items():
#         print(f"Layer '{name}' feature shape: {feat.shape}")
#     extractor_manual.remove_hooks() # IMPORTANT!
#     print("Manual hooks removed.")
#
#     # --- Method 2: Using 'with' statement ---
#     print("\n--- Method 2: Using 'with' Statement ---")
#     with FeatureExtractor(dummy_model, layer_names_to_extract) as extractor_with:
#         output_with = dummy_model(dummy_input) # Run forward pass
#         features_with = extractor_with.get_features_dict()
#         for name, feat in features_with.items():
#             print(f"Layer '{name}' feature shape: {feat.shape}")
#     # Hooks are automatically removed upon exiting the 'with' block
#     print("'with' block exited, hooks removed.")
#
#     # Verify hooks are removed (this check might need adjustment based on hook implementation details)
#     print(f"Extractor 'with' hooks list empty: {len(extractor_with._hooks) == 0}")