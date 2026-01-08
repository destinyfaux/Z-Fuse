import os
import torch
import json
import folder_paths
import comfy.model_management
import comfy.utils
from safetensors.torch import load_file, save_file, safe_open
import traceback
import gc
import re

# =============================================================================
# UTILITIES: DYNAMIC MAPPING & NORMALIZATION
# =============================================================================

def get_model_prefix(model):
    """Interrogates the model to find the correct key prefix."""
    if hasattr(model.model, 'diffusion_model'):
        return "diffusion_model."
    return ""

def get_block_info(key):
    """
    Robustly extracts block index and type from a LoRA key.
    Returns (index, type_label) or (-1, None)
    """
    # 1. Z-Image / S3-DiT Block Patterns (0-29)
    # Handles: diffusion_model.layers.10, lora_unet_layers_10, lycoris_layers_10
    match = re.search(r'(?:layers|blocks|single_transformer_blocks|double_blocks|lycoris_layers)[._](\d+)', key, re.IGNORECASE)
    if match:
        return int(match.group(1)), "block"
    
    # 2. Special Z-Image Layers (Non-Block)
    if "cap_embedder" in key: return 30, "cap_embedder"
    if "context_refiner" in key: return 30, "context_refiner"
    if "noise_refiner" in key: return 31, "noise_refiner"
    if "x_embedder" in key: return 32, "x_embedder"
    if "final_layer" in key: return 33, "final_layer"
    
    # 3. Fallback: Finding isolated numbers for unknown architectures
    numbers = re.findall(r"(?<![a-zA-Z])(\d+)(?![a-zA-Z])", key)
    if numbers:
        for n in numbers:
            idx = int(n)
            if 0 <= idx <= 35: # Generous range for Z-Image
                return idx, "fallback"
                
    return -1, None

def normalize_key(lora_key, model_prefix):
    """
    Converts a LoRA key to the Model's internal key format.
    Ensures patches attach to the correct address by preserving .weight.
    """
    # 1. Identify if it's a weight or a bias
    suffix = ".weight" if ".weight" in lora_key else ""
    
    # 2. Strip ALL LoRA-specific noise and extensions
    base = lora_key.replace(".lora_up.weight", "").replace(".lora_down.weight", "").replace(".alpha", "")
    base = base.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
    base = base.replace(".lokr_w1.weight", "").replace(".lokr_w2.weight", "")
    base = base.replace(".hada_w1.weight", "").replace(".hada_w2.weight", "")
    base = base.replace(".diff", "").replace(".diff_b", "")
    base = base.replace(".weight", "") # Clean the base to raw parameter name
    
    # 3. Strip common trainer prefixes (Kohya, LyCORIS, etc.)
    base = base.replace("lora_unet_", "").replace("lora_te_", "").replace("lycoris_", "")
    
    # 4. Standardize block/layer separators
    base = re.sub(r"(layers|blocks|single_transformer_blocks|cap_embedder|context_refiner)[\._]", r"\1.", base)
    
    # 5. Reconstruct the target key for ModelPatcher
    if base.startswith(model_prefix):
        final_key = f"{base}{suffix}"
    else:
        final_key = f"{model_prefix}{base}{suffix}"
        
    return final_key

# =============================================================================
# NODE CLASSES
# =============================================================================

class ZFuseVisualLayerTuner:
    """
    Visual Tuner: Exposes 30 blocks + Base Strength.
    Defaults to 1.0 (Neutral).
    0.0 = Disabled (Skipped calculation).
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
        # Dynamic sliders for Blocks 0-29
        for i in range(30):
            # Default 1.0 means "Neutral" (Use Base Strength)
            # 0.0 means "Disabled"
            inputs["required"][f"block_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05})
        
        inputs["optional"] = {"lora_stack": ("LORA_STACK", )}
        return inputs

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("LORA_STACK", "ANALYSIS_REPORT")
    FUNCTION = "tune_layers"
    CATEGORY = "Z-Image/Fusion"

    def _get_lora_impact(self, lora_path, custom_map):
        """Lightweight analysis of LoRA block intensity."""
        impacts = [0.0] * 35
        try:
            # Load safely on CPU to avoid VRAM spikes before main execution
            sd = load_file(lora_path)
            
            for key, val in sd.items():
                if not any(x in key for x in ["lora_up", "lora_A", "lokr_w1", "hada_w1"]):
                    continue
                
                # Identify Block
                block_id, _ = get_block_info(key)
                
                # --- SKIP ZERO OPTIMIZATION ---
                # If user explicitly set this block to 0.0 in UI, skip matrix math entirely.
                if block_id in custom_map and custom_map[block_id] == 0.0:
                    continue
                
                # Find pair
                down_key = key.replace("lora_up", "lora_down").replace("lora_A", "lora_B").replace("lokr_w1", "lokr_w2").replace("hada_w1", "hada_w2")
                
                if down_key in sd:
                    down = sd[down_key]
                    
                    # Matrix Multiplication Logic (CPU based, Float32 for precision)
                    try:
                        up = val.to(torch.float32)
                        down = down.to(torch.float32)
                        
                        delta = None
                        if up.shape[-1] == down.shape[0]:
                            delta = torch.matmul(up, down)
                        elif down.shape[-1] == up.shape[0]:
                            delta = torch.matmul(down, up)
                        elif up.shape[-1] == down.shape[-1]:
                            delta = torch.matmul(up, down.t())
                        elif up.shape == down.shape:
                            delta = up * down
                        
                        if delta is not None:
                            block_id, _ = get_block_info(key)
                            if block_id != -1 and block_id < 35:
                                impacts[block_id] += torch.norm(delta).item()
                    except Exception:
                        pass 
        except Exception as e:
            print(f"[Z-FUSE] VLT Analysis Error: {e}")
            
        return impacts

    def tune_layers(self, lora_name, base_strength, **kwargs):
        lora_stack = kwargs.get("lora_stack", [])
        custom_map = {}
        for i in range(30):
            val = kwargs.get(f"block_{i}", 1.0)
            # Only store modification if it is NOT neutral (1.0)
            if val != 1.0:
                custom_map[i] = val
        
        # --- Analysis Logic ---
        path = folder_paths.get_full_path("loras", lora_name)
        report_str = ""
        ui_impacts = [0.0] * 35 
        
        if path:
            ui_impacts = self._get_lora_impact(path, custom_map)
            max_act = max(ui_impacts) if max(ui_impacts) > 0 else 1.0
            
            # Format Human Readable Report
            report_lines = []
            for i, act in enumerate(ui_impacts):
                if act > 0:
                    bar = "█" * int((act / max_act) * 20)
                    lbl = f"Block {i:02d}" if i < 30 else ["Context", "Noise", "X-Emb", "Final", "Other"][i-30]
                    report_lines.append(f"{lbl}: {bar} ({act:.2f})")
            
            report_str = "\n".join(report_lines) if report_lines else "No activity detected (Blocks may be disabled or neutral)."
        else:
            report_str = "Error: LoRA file not found."

        lora_stack.append({
            "name": lora_name,
            "strength": base_strength,
            "range": (0, 30),
            "custom_map": custom_map,
            "mode": "visual_tuner"
        })

        # Return dict with UI data for dynamic updates and result for output pins
        return {
            "ui": {"analysis_report": ui_impacts},
            "result": (lora_stack, report_str)
        }

class ZFuseLoRAStack:
    """Standard Bus: Range-based isolation (Start/End)."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "block_start": ("INT", {"default": 0, "min": 0, "max": 30}),
                "block_end": ("INT", {"default": 30, "min": 0, "max": 30}),
                "layer_weights": ("STRING", {"default": "", "placeholder": "12:1.5, 14:0.5 (Overrides)", "multiline": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", ),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "stack_lora"
    CATEGORY = "Z-Image/Fusion"

    def stack_lora(self, lora_name, strength, block_start, block_end, layer_weights="", lora_stack=None):
        lora_list = lora_stack if lora_stack is not None else []
        custom_map = {}
        if layer_weights.strip():
            try:
                pairs = [p.strip() for p in layer_weights.replace('\n', ',').split(",") if ":" in p]
                for p in pairs:
                    parts = p.split(":")
                    # In text mode, these are absolute multipliers, not offsets
                    custom_map[int(parts[0])] = float(parts[1])
            except Exception as e:
                print(f"[Z-FUSE] Parse Error: {e}")

        lora_list.append({
            "name": lora_name,
            "strength": strength,
            "range": (block_start, block_end),
            "custom_map": custom_map,
            "mode": "standard_bus"
        })
        return (lora_list,)

class ZFuseOrchestrator:
    def __init__(self):
        # This keeps the data in memory between executions
        self.result_cache = {}
    """
    The Brain.
    - Interrogates Model & LoRAs
    - Fuses weights (Simultaneous/TIES)
    - Diagnostics (Dump Info, Heatmaps)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": ("LORA_STACK",),
                "merge_mode": (["TIES", "Weighted Sum"], {"default": "TIES"}),
                "ties_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "TRIGGER_WORDS", "ANALYSIS_REPORT", "STATUS_LOG", "DUMP_INFO")
    FUNCTION = "orchestrate"
    CATEGORY = "Z-Image/Fusion"

    def _normalize_tensor_shapes(self, deltas, status_log):
        """
        Pads all tensors in the list to match the largest dimensions.
        This resolves Rank mismatches (e.g., Rank 32 vs Rank 128) or 
        Geometry mismatches by zero-padding the smaller tensors.
        """
        if not deltas:
            return []
            
        # Find the maximum shape across all contributors for this key
        max_h = max(d.shape[0] for d in deltas)
        max_w = max(d.shape[1] for d in deltas)
        target_shape = (max_h, max_w)
        
        normalized_deltas = []
        
        for d in deltas:
            if d.shape == target_shape:
                normalized_deltas.append(d)
            else:
                # Padding logic: Pad height (dim 0) and width (dim 1)
                # F.pad arguments: (left, right, top, bottom)
                pad_h = target_shape[0] - d.shape[0]
                pad_w = target_shape[1] - d.shape[1]
                
                # Only pad if needed
                if pad_h > 0 or pad_w > 0:
                    # We preserve dtype (usually float16)
                    padded = torch.nn.functional.pad(d, (0, pad_w, 0, pad_h))
                    status_log.append(f"  [SHAPE FIX] Padded tensor {d.shape} -> {target_shape}")
                    normalized_deltas.append(padded)
                else:
                    normalized_deltas.append(d)
                    
        return normalized_deltas

    def orchestrate(self, model, lora_stack, merge_mode, ties_threshold):
        # Create a unique key based on the current inputs
        # This ensures if you change a slider, the cache invalidates
        state_key = hash(str(lora_stack) + str(merge_mode) + str(ties_threshold))
        
        if state_key in self.result_cache:
            status_log = ["Z-FUSE: Using Cached Fusion Result (Instant)"]
            cached_data = self.result_cache[state_key]
            
            # We still need to clone the model and apply the cached patches
            work_model = model.clone()
            work_model.add_patches(cached_data['patches'], 1.0, 1.0)
            
            return (work_model, cached_data['triggers'], cached_data['report'], 
                    "\n".join(status_log), cached_data['dump'])
            
        # Get the fast device (GPU) and the big device (CPU)
        cuda_device = comfy.model_management.get_torch_device()
        offload_device = torch.device("cpu")
        
        new_model = model.clone()
        unique_triggers = []
        status_log = ["--- Z-FUSE ORCHESTRATION INITIATED ---"]
        dump_info = []
        
        # 1. Model Interrogation
        model_prefix = get_model_prefix(model)
        status_log.append(f"Model Architecture detected. Prefix set to: '{model_prefix}'")
        
        aggregated_deltas = {} 
        global_activity = [0.0] * 35 # 0-29 blocks + 5 specials
        
        pbar = comfy.utils.ProgressBar(len(lora_stack))
        
        for entry in lora_stack:
            path = folder_paths.get_full_path("loras", entry["name"])
            if not path:
                status_log.append(f"!! SKIP: {entry['name']} (File not found)")
                continue
            
            # Metadata Trigger Harvesting
            try:
                with safe_open(path, framework="pt") as f:
                    meta = f.metadata()
                    if meta:
                        raw = meta.get("ss_trained_words") or meta.get("ss_tag_frequency") or ""
                        if raw:
                            clean = raw.replace('[','').replace(']','').replace('"','').replace('{','').replace('}','')
                            for w in [x.strip() for x in clean.split(',') if x.strip()]:
                                tag = w.split(':')[0].strip()
                                if tag and tag not in unique_triggers: unique_triggers.append(tag)
            except: pass

            status_log.append(f"Processing: {entry['name']} ({entry['mode']})")
            
            try:
                sd = load_file(path)
                mapped_count = 0
                sample_keys = []
                
                # Capture first 10 keys for Dump Info in case of failure
                all_keys_list = list(sd.keys())
                dump_info.append(f"\n[{entry['name']}] Sample Keys:")
                for k in all_keys_list[:10]:
                    dump_info.append(f"  {k}")

                for key, val in sd.items():
                    if not any(x in key for x in ["lora_up", "lora_A", "lokr_w1", "hada_w1"]):
                        continue
                    
                    # Identify Block
                    block_id, block_type = get_block_info(key)
                    
                    # Logic: Visual Tuner (Offset) vs Bus (Multiplier)
                    base_str = entry["strength"]
                    if block_id != -1:
                        if entry['mode'] == 'visual_tuner':
                            # Logic: If in custom_map, use it. Defaults to 1.0 if missing.
                            mod = entry["custom_map"].get(block_id, 1.0)
                            final_strength = base_str * mod
                        else:
                            # Bus uses multipliers or range clipping
                            if block_id in entry["custom_map"]:
                                final_strength = entry["custom_map"][block_id] # Override
                            elif not (entry["range"][0] <= block_id <= entry["range"][1]):
                                final_strength = 0.0
                            else:
                                final_strength = base_str
                    else:
                        # Non-block key (unknown), treat as global strength
                        final_strength = base_str

                    if abs(final_strength) < 0.001: continue

                    # Reconstruct Delta (CUDA Accelerated / Hybrid)
                    # 1. Map the pair (Up -> Down / A -> B)
                    down_key = key.replace("lora_up", "lora_down").replace("lora_A", "lora_B").replace("lokr_w1", "lokr_w2").replace("hada_w1", "hada_w2")

                    if down_key in sd:
                        try:
                            # 2. Cast to float32 immediately for mixed-dtype support
                            up = val.to(cuda_device).to(torch.float16)
                            down = sd[down_key].to(cuda_device).to(torch.float16)

                            # 3. Key components for scaling
                            alpha_key = key.split(".lora")[0].split(".lokr")[0].split(".hada")[0] + ".alpha"
                            delta = None
                            rank = 1.0
                        
                            # 4. Smart Multiplication (Orientation Agnostic)
                            if up.shape[-1] == down.shape[0]: # Standard
                                delta = torch.matmul(up, down)
                                rank = float(up.shape[-1])
                            elif down.shape[-1] == up.shape[0]: # Olympus / Mystic
                                delta = torch.matmul(down, up)
                                rank = float(down.shape[-1])
                            elif up.shape[-1] == down.shape[-1]: # Transposed
                                delta = torch.matmul(up, down.t())
                                rank = float(up.shape[-1])
                            elif up.shape == down.shape: # Hadamard fallback
                                delta = up * down
                                rank = float(up.shape[-1]) if len(up.shape) > 1 else 1.0
                            else:
                                dump_info.append(f"  !! Shape Mismatch: {up.shape} vs {down.shape} at {key}")
                                continue

                            if delta is not None:
                                # 5. Universal Scale Application
                                # We get the raw value first to avoid unnecessary tensor moves
                                alpha_val = sd[alpha_key].item() if alpha_key in sd else rank
                                
                                # Scale the delta while it is still on the GPU
                                scale = final_strength * (alpha_val / rank)
                                delta = delta * scale
                                
                                # Prevents any single LoRA from over-saturating weights / Too agressive commented out for now
                                #max_norm = 2.0 
                                #actual_norm = torch.norm(delta)
                                #if actual_norm > max_norm:
                                #    delta = delta * (max_norm / actual_norm)
                                
                                # 6. CRITICAL: Move result to CPU immediately 
                                # (Saves VRAM from the accumulator)
                                delta = delta.to(offload_device).to(torch.float16)
                                
                                # --- Store in global accumulator ---
                                final_key = normalize_key(key, model_prefix)
                                
                                # OPTIONAL: Verification Log (First 3 keys only)
                                if mapped_count < 3:
                                    dump_info.append(f"  [MAP TEST] LoRA Key: {key}")
                                    dump_info.append(f"  [MAP TEST] Target Key: {final_key}")

                                if final_key not in aggregated_deltas:
                                    aggregated_deltas[final_key] = []
                                aggregated_deltas[final_key].append(delta)
                                
                                mapped_count += 1
                                
                                # Analysis data
                                if block_id != -1 and block_id < 35:
                                    global_activity[block_id] += torch.norm(delta).item()

                        except Exception as math_err:
                            dump_info.append(f"  !! Math Error at {key}: {str(math_err)}")
                            continue
                        finally:
                            # 7. Explicitly clear GPU temporaries
                            del up, down
                        
                        # Analysis data
                        if block_id != -1 and block_id < 35:
                            global_activity[block_id] += torch.norm(delta).item()

                if mapped_count == 0:
                    status_log.append("  WARNING: 0 Keys Mapped. See DUMP_INFO.")
                else:
                    status_log.append(f"  > Mapped {mapped_count} keys.")
                
                del sd
                gc.collect()
                comfy.model_management.soft_empty_cache()
            except Exception as e:
                status_log.append(f"!! Error: {str(e)}")
            
            pbar.update(1)
            gc.collect()

        # 2. Fusion Engine (Memory-Safe topk version)
        status_log.append(f"Fusing {len(aggregated_deltas)} unique tensors using {merge_mode}...")
        fused_patches = {}
        
        for key, deltas in aggregated_deltas.items():
            # --- SHAPE NORMALIZATION FIX ---
            # Ensure all tensors for this key have identical dimensions before stacking
            deltas = self._normalize_tensor_shapes(deltas, status_log)
            
            num_contributors = len(deltas)
            stacked = torch.stack(deltas).to(torch.float32)
            
            if merge_mode == "Weighted Sum" or len(deltas) == 1:
                fused = stacked.sum(dim=0) / num_contributors
            
            elif merge_mode == "TIES":
                # ARCHITECTURAL FIX: Avoid Quantile Overflow
                # Instead of sorting billions of elements, we use a robust mean-based threshold 
                # or a sampled quantile which is mathematically equivalent for sparse LoRAs.
                flat = stacked.view(num_contributors, -1)
                
                # Sample 1/10th of the data to estimate quantile if tensor is too large
                # This prevents the 'tensor too large' RuntimeError
                if flat.shape[1] > 10_000_000:
                    sample_indices = torch.randint(0, flat.shape[1], (1_000_000,), device=flat.device)
                    sampled_flat = flat[:, sample_indices]
                    thresh = torch.quantile(torch.abs(sampled_flat), ties_threshold, dim=1, keepdim=True)
                else:
                    thresh = torch.quantile(torch.abs(flat), ties_threshold, dim=1, keepdim=True)
                
                mask = torch.abs(stacked) >= thresh.view(num_contributors, 1, 1)
                trimmed = stacked * mask.float()
                
                # Elect Sign: Consensus on weight direction
                signs = torch.sign(trimmed.sum(dim=0))
                
                # Disjoint Merge: Only average contributors that agree with consensus
                aligned_mask = (torch.sign(trimmed) == signs).float()
                consensus_count = aligned_mask.sum(dim=0).clamp(min=1.0)
                fused = (trimmed * aligned_mask).sum(dim=0) / consensus_count

            # SAFETY CLAMP: Element-wise clamp to prevent "Frying" (NaNs/Inf)
            # Clamping individual values to +/- 4.0 allows up to 40x standard strength 
            # while preventing numerical instability.
            fused = torch.clamp(fused, -4.0, 4.0).to(torch.float16)
            fused_patches[key] = (fused.to(offload_device),)

            # Explicit cleanup
            del stacked, fused
            if "trimmed" in locals(): del trimmed, mask, signs, aligned_mask

        # 3. Apply to Model
        if fused_patches:
            # --- DEBUG: Key Match Verification ---
            model_keys = new_model.model.state_dict().keys()
            match_count = 0
            for p_key in fused_patches.keys():
                if p_key in model_keys:
                    match_count += 1
            
            status_log.append(f"DEBUG: {match_count} of {len(fused_patches)} keys matched the model.")
            
            # Apply to model
            new_model.add_patches(fused_patches, 1.0, 1.0)
            status_log.append(f"Successfully applied {len(fused_patches)} patched layers.")
            
            # CRITICAL: Immediate Memory Release
            del fused_patches
            torch.cuda.empty_cache()
            aggregated_deltas.clear()
            del aggregated_deltas
            gc.collect()
            comfy.model_management.soft_empty_cache()
            
            # REMOVED EARLY RETURN HERE TO ALLOW REPORTING SECTION TO RUN
            # The original return here bypassed the trigger extraction and heatmap generation.
            # return (new_model, "\n".join(status_log), "\n".join(dump_info), list(global_activity))
            
        else:
            status_log.append("CRITICAL: No patches generated. Output is identical to input.")

        # 4. Reporting
        report = ["### Z-FUSE GLOBAL IMPACT REPORT"]
        max_act = max(global_activity) if max(global_activity) > 0 else 1.0
        
        # Ensure global_activity is clean for the UI
        ui_report = [float(a) for a in global_activity] 

        for i, act in enumerate(global_activity):
            if act > 0:
                bar = "█" * int((act / max_act) * 20)
                # Matches your 35-index range from get_block_info
                lbl = f"Block {i:02d}" if i < 30 else ["Context", "Noise", "X-Emb", "Final", "Other"][i-30]
                report.append(f"{lbl}: {bar} ({act:.2f})")

        status_log.append("--- DONE ---")

        # Cache the results
        self.result_cache[state_key] = {
            'patches': fused_patches if 'fused_patches' in locals() else {}, # Handle empty case for cache
            'triggers': ", ".join(unique_triggers),
            'report': "\n".join(report),
            'dump': "\n".join(dump_info),
            'ui_report': ui_report # Also cache the UI data
        }

        return {
            "ui": {"analysis_report": ui_report}, 
            "result": (new_model, ", ".join(unique_triggers), "\n".join(report), "\n".join(status_log), "\n".join(dump_info))
        }

class ZFuseBake:
    """Export Node: Saves the model state to a LoRA file."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "save_name": ("STRING", {"default": "z_fuse_export"}),
                # Rank 0 = Auto (Full Rank/No-Loss). Any number > 0 attempts SVD (Lossy).
                "rank": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}), 
                "precision": (["float16", "bfloat16", "float32"], {"default": "float16"}),
                "custom_path": ("STRING", {"default": ""}),
                "vram_protection": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SAVE_STATUS")
    FUNCTION = "bake"
    CATEGORY = "Z-Image/Fusion"
    OUTPUT_NODE = True

    def _find_tensor(self, item):
        """Recursively digs into tuples/lists to find the actual torch.Tensor."""
        if isinstance(item, torch.Tensor):
            return item
        if isinstance(item, (tuple, list)):
            for sub_item in item:
                res = self._find_tensor(sub_item)
                if res is not None:
                    return res
        return None

    def _find_strength(self, item):
        """Finds the scalar multiplier (strength) in the tuple/list."""
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
        if isinstance(item, (tuple, list)):
            for sub_item in item:
                res = self._find_strength(sub_item)
                if res is not None:
                    return res
        if isinstance(item, torch.Tensor) and item.numel() == 1:
            return item.item()
        return 1.0

    def bake(self, model, save_name, rank, precision, custom_path, vram_protection):
        logs = ["--- Z-FUSE NO-LOSS BAKE INITIATED ---"]
        
        try:
            # 1. Setup Data Types
            dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            out_dtype = dtype_map[precision]
            
            # 2. Safety & VRAM Cleanup
            if vram_protection:
                comfy.model_management.soft_empty_cache()
                gc.collect()
            
            logs.append(f"Target Precision: {precision}")
            
            mode_str = "Full Rank / No-Loss" if rank == 0 else f"Rank {rank} (Compressed/Lossy)"
            logs.append(f"Requested Rank: {rank} ({mode_str})")

            # Extract patches
            patches = model.patches
            
            if not patches:
                return ("ERROR: No patches found in model. Ensure to connect the Orchestrator output directly to the Bake node.",)
            
            logs.append(f"Found {len(patches)} patch keys in model.")

            baked_sd = {}
            count = 0
            total_skipped = 0
            debug_counter = 0
            
            # Keep calculations in VRAM for speed
            cuda_device = comfy.model_management.get_torch_device()
            
            for key, patch_list in patches.items():
                total_delta = None
                
                # Debug: Print info for the first few keys to diagnose structure
                if debug_counter < 3:
                    logs.append(f"DEBUG Key {debug_counter} ({key}): List Length = {len(patch_list)}")
                    if len(patch_list) > 0:
                         logs.append(f"  > First Item Raw Type: {type(patch_list[0])}")
                    debug_counter += 1

                for p in patch_list:
                    try:
                        # 1. ROBUST EXTRACTION (In VRAM)
                        t = self._find_tensor(p)
                        strength = self._find_strength(p)
                        
                        if t is None:
                            continue

                        # 2. DIRECT MANIPULATION (In VRAM)
                        # Move to Target Precision in VRAM (Fast)
                        # We don't force a copy unless necessary.
                        d = t.to(out_dtype)
                        d = d * strength
                        
                        if total_delta is None:
                            total_delta = d
                        else:
                            total_delta += d
                            
                    except Exception as e:
                        # Log individual patch failures but don't crash the whole bake
                        logs.append(f"  !! Error processing patch for {key}: {e}")
                        continue

                # Save Logic
                if total_delta is not None and total_delta.dim() >= 2:
                    out_features = total_delta.shape[0]
                    in_features = total_delta.shape[1]
                    
                    # LOGIC: Full Rank (No Loss) vs User Rank (SVD/Lossy)
                    user_rank = in_features if rank == 0 else rank
                    
                    if user_rank >= in_features:
                        # NO-LOSS
                        # up is the actual delta. 
                        # down is Identity matrix. 
                        # We construct the Identity matrix on the target device (usually CUDA).
                        up_weight = total_delta
                        down_weight = torch.eye(in_features, dtype=out_dtype, device=total_delta.device)
                        alpha = torch.tensor(in_features, dtype=torch.float16)
                        
                        logs.append(f"[{key}] Saved Full-Rank (No-Loss). Shape: {total_delta.shape}")
                    else:
                        # LOSSY
                        try:
                            # Convert to float32 for stable SVD
                            U, S, Vh = torch.linalg.svd(total_delta.float(), full_matrices=False)
                            U = U[:, :user_rank]
                            S = S[:user_rank]
                            Vh = Vh[:user_rank, :]
                            
                            # Combine S into U for standard LoRA Up weight
                            up_weight = (U @ torch.diag(S)).to(out_dtype)
                            down_weight = Vh.to(out_dtype)
                            alpha = torch.tensor(user_rank, dtype=torch.float16)
                            
                            logs.append(f"[{key}] SVD Compressed to Rank {user_rank} (Lossy).")
                        except Exception as svd_err:
                            logs.append(f"  !! SVD FAILED for {key}: {svd_err}. Skipping.")
                            total_skipped += 1
                            continue

                    # Save to dictionary
                    baked_sd[f"{key}.lora_up.weight"] = up_weight
                    baked_sd[f"{key}.lora_down.weight"] = down_weight
                    baked_sd[f"{key}.alpha"] = alpha
                    count += 1
                    
                    # Explicit cleanup
                    del total_delta, up_weight, down_weight
                    
                elif total_delta is not None:
                    logs.append(f"  !! Skipping {key} (1D Bias/Unsupported shape {total_delta.shape})")
                    total_skipped += 1
                else:
                    logs.append(f"  !! Skipping {key} (No valid tensors extracted)")

            # 4. Save to Disk
            if count == 0:
                return ("\n".join(logs) + "\n\nFAILED: No valid 2D weight patches found to bake. Check DEBUG info above.",)
            
            # VRAM Cleanup before Save
            torch.cuda.empty_cache()
            gc.collect()
            
            save_dir = custom_path.strip() if custom_path.strip() else os.path.join(folder_paths.get_output_directory(), "loras")
            os.makedirs(save_dir, exist_ok=True)
            
            # Filename
            r_str = "FullRank" if rank == 0 else f"r{rank}"
            full_path = os.path.join(save_dir, f"{save_name}_{r_str}_{precision}.safetensors")
            
            # Metadata
            metadata = {
                "ss_framework": "Z-FUSE",
                "fused_by": "Z-FUSE No-Loss Bake",
                "rank": r_str,
                "precision": precision
            }
            
            # save_file handles the final CPU write efficiently
            save_file(baked_sd, full_path, metadata=metadata)
            
            logs.append(f"\n--- SUCCESS ---")
            logs.append(f"Saved {count} layers to: {full_path}")
            logs.append(f"Skipped {total_skipped} layers.")
            
            return ("\n".join(logs),)
            
        except Exception as e:
            error_msg = f"\n\n!!! FATAL BAKE ERROR !!!\n{traceback.format_exc()}"
            print(error_msg) # Print to console too
            return ("\n".join(logs) + error_msg,)

NODE_CLASS_MAPPINGS = {
    "ZFuseLoRAStack": ZFuseLoRAStack,
    "ZFuseVisualLayerTuner": ZFuseVisualLayerTuner,
    "ZFuseOrchestrator": ZFuseOrchestrator,
    "ZFuseBake": ZFuseBake
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZFuseLoRAStack": "Z-FUSE: LoRA Bus (Stacker)",
    "ZFuseVisualLayerTuner": "Z-FUSE: Visual Layer Tuner",
    "ZFuseOrchestrator": "Z-FUSE: Surgical Orchestrator",
    "ZFuseBake": "Z-FUSE: No-Loss Bake (Export)"
}
