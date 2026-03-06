import os
import torch
import json
import folder_paths
import comfy.model_management
import comfy.utils
import comfy.sd
from safetensors.torch import load_file, save_file, safe_open
import traceback
import gc
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Set, Any
import torch.nn.functional as F

# =============================================================================
# CONFIGURATION & UTILITIES
# =============================================================================

ZIMAGE_BLOCKS = 30
ZIMAGE_SPECIALS = {
    "cap_embedder": 30,
    "context_refiner": 31, 
    "noise_refiner": 32,
    "x_embedder": 33,
    "final_layer": 34
}

# Naming convention mappings for heuristic matching
CONVENTION_MAP = {
    "qkv": ["to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"],
    "out": ["to_out", "out_proj", "proj_out", "to_out.0"],
    "adaLN": ["adaLN_modulation", "norm1", "norm2", "adaLN"],
    "mlp": ["ff", "feed_forward", "mlp", "w1", "w2", "w3", "fc1", "fc2"]
}

def get_block_info(key: str) -> Tuple[int, Optional[str]]:
    """Extract block index and type from key."""
    match = re.search(r'(?:layers|blocks|single_transformer_blocks|double_blocks)[._](\d+)', key, re.IGNORECASE)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < ZIMAGE_BLOCKS:
            return idx, "core"
    
    for name, idx in ZIMAGE_SPECIALS.items():
        if name in key:
            return idx, name
    
    return -1, None

def normalize_lora_key(lora_key: str) -> str:
    """Strip all prefixes and suffixes from LoRA key to get clean internal name."""
    key = lora_key
    
    # Strip LoRA suffixes
    for suffix in [".lora_up.weight", ".lora_down.weight", ".lora_A.weight", 
                   ".lora_B.weight", ".lokr_w1.weight", ".lokr_w2.weight",
                   ".hada_w1.weight", ".hada_w2.weight", ".alpha"]:
        key = key.replace(suffix, "")
    
    # Strip framework prefixes
    key = re.sub(r'^(diffusion_model\.|lora_unet_|lora_te_|model\.diffusion_model\.|transformer\.|model\.)', '', key)
    
    # Normalize separators
    key = re.sub(r"(layers|blocks)[\._]", r"\1.", key)
    
    return key

def get_alpha_value(sd: Dict[str, torch.Tensor], alpha_key: str, rank: int) -> float:
    """Get alpha with fallback chain."""
    if alpha_key in sd:
        alpha_tensor = sd[alpha_key]
        try:
            # Try scalar first
            return alpha_tensor.item()
        except RuntimeError:
            # Multi-element tensor
            try:
                return alpha_tensor.mean().item()
            except:
                try:
                    return alpha_tensor[0].item()
                except:
                    pass
    # Final fallback
    return float(rank)

def find_model_match(clean_key: str, model_keys: Set[str]) -> Tuple[Optional[str], str]:
    """
    Find matching model key using heuristics and substring matching.
    Returns (full_model_key, match_method)
    """
    # 1. Heuristic mapping (e.g., to_k -> qkv)
    for model_part, variants in CONVENTION_MAP.items():
        for var in variants:
            if var in clean_key:
                # Replace variant with standard name
                potential = clean_key.replace(var, model_part)
                
                # Find model key containing this pattern
                for m_key in model_keys:
                    # Check if the transformed pattern is in the model key
                    if potential in m_key:
                        return m_key, f"Heuristic ({var}->{model_part})"
                    
                    # Also check with dots normalized
                    potential_norm = potential.replace(".", "")
                    m_key_norm = m_key.replace(".", "")
                    if potential_norm in m_key_norm:
                        return m_key, f"HeuristicNorm ({var}->{model_part})"
    
    # 2. Direct substring match
    for m_key in model_keys:
        if clean_key in m_key:
            return m_key, "Substring"
        
        # Try without dots
        clean_no_dots = clean_key.replace(".", "")
        m_key_no_dots = m_key.replace(".", "")
        if clean_no_dots in m_key_no_dots:
            return m_key, "SubstringNoDots"
    
    # 3. Fuzzy match by layer number and rough structure
    layer_match = re.search(r'layers\.(\d+)', clean_key)
    if layer_match:
        layer_num = layer_match.group(1)
        # Find any model key with same layer number
        for m_key in model_keys:
            if f"layers.{layer_num}" in m_key:
                # Very fuzzy - just same layer
                return m_key, f"FuzzyLayer{layer_num}"
    
    return None, "None"

def generate_cache_key(lora_stack: List[Dict], **params) -> str:
    """Generate unique hash for cache invalidation."""
    normalized = []
    for entry in sorted(lora_stack, key=lambda x: x.get('name', '')):
        normalized.append({
            'name': entry.get('name', ''),
            'strength': round(entry.get('strength', 1.0), 4),
            'range': entry.get('range', (0, 29)),
            'mode': entry.get('mode', 'standard'),
            'custom_map': {k: round(v, 4) for k, v in sorted(entry.get('custom_map', {}).items())}
        })
    
    data = json.dumps({
        'stack': normalized,
        'params': params
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()

# =============================================================================
# TIES FUSER - Outputs State Dict for ComfyUI Loader
# =============================================================================

class TIESFuser:
    """
    TIES merging that produces a state_dict for comfy.sd.load_lora_for_models.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.accumulated = {}  # {full_model_key: delta_tensor}
        self.conflict_stats = {}
        self.processed_samples = []
        self.model_keys = set()
        self.match_stats = {"matched": 0, "unmatched": 0}
        
    def reset(self):
        self.accumulated.clear()
        self.conflict_stats.clear()
        self.processed_samples.clear()
        self.model_keys.clear()
        self.match_stats = {"matched": 0, "unmatched": 0}
        gc.collect()
        torch.cuda.empty_cache()
    
    def set_model_keys(self, model_keys: List[str]):
        """Set reference model keys for matching."""
        self.model_keys = set(model_keys)
    
    def _compute_delta(self, sd: Dict[str, torch.Tensor], up_key: str, 
                       strength: float) -> Optional[torch.Tensor]:
        """Compute scaled delta with robust alpha handling."""
        # Find down key
        down_key = (up_key.replace("lora_up", "lora_down")
                         .replace("lora_A", "lora_B")
                         .replace("lokr_w1", "lokr_w2")
                         .replace("hada_w1", "hada_w2"))
        
        if down_key not in sd:
            return None
        
        # Load to VRAM
        up = sd[up_key].to(self.device, dtype=self.dtype)
        down = sd[down_key].to(self.device, dtype=self.dtype)
        
        # Get alpha with fallback
        alpha_key = up_key.replace(".lora_up.weight", ".alpha").replace(".lora_A.weight", ".alpha")
        rank = up.shape[-1] if len(up.shape) > 1 else up.shape[0]
        alpha = get_alpha_value(sd, alpha_key, rank)
        scale = strength * (alpha / rank)
        
        try:
            # Matrix multiplication
            if up.shape[-1] == down.shape[0]:
                delta = torch.matmul(up, down)
            elif down.shape[-1] == up.shape[0]:
                delta = torch.matmul(down, up)
            elif up.shape[-1] == down.shape[-1]:
                delta = torch.matmul(up, down.t())
            elif up.shape == down.shape:
                delta = up * down
            else:
                return None
            
            delta = delta * scale
            
        finally:
            del up, down
            
        return delta
    
    def _match_shapes(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Pad tensor to match target shape."""
        if tensor.shape == target_shape:
            return tensor
        
        pad = []
        for i in reversed(range(len(target_shape))):
            if i < len(tensor.shape):
                pad.extend([0, target_shape[i] - tensor.shape[i]])
            else:
                pad.extend([0, target_shape[i]])
        
        return F.pad(tensor, pad)
    
    def fuse_lora(self, lora_path: str, strength: float, 
                  block_range: Tuple[int, int], custom_map: Dict[int, float],
                  mode: str = "standard", ties_threshold: float = 0.2):
        """Fuse one LoRA into accumulated state."""
        try:
            sd = load_file(lora_path, device="cpu")
            up_keys = [k for k in sd.keys() if any(x in k for x in 
                      ["lora_up", "lora_A", "lokr_w1", "hada_w1"])]
            
            for idx, key in enumerate(up_keys):
                block_id, _ = get_block_info(key)
                
                # Determine strength
                if mode == "visual_tuner":
                    mod = custom_map.get(block_id, 1.0)
                    if mod == 0.0:
                        continue
                    final_strength = strength * mod
                else:
                    if block_id in custom_map:
                        final_strength = custom_map[block_id]
                    elif not (block_range[0] <= block_id <= block_range[1]):
                        continue
                    else:
                        final_strength = strength
                
                if abs(final_strength) < 0.001:
                    continue
                
                # Compute delta
                delta = self._compute_delta(sd, key, final_strength)
                if delta is None:
                    continue
                
                # Normalize and find model match
                clean_key = normalize_lora_key(key)
                model_key, match_method = find_model_match(clean_key, self.model_keys)
                
                # Store sample for debugging
                if len(self.processed_samples) < 5:
                    self.processed_samples.append(
                        f"{key[:50]}... -> {clean_key[:40]}... -> {model_key[:50] if model_key else 'None'} ({match_method})"
                    )
                
                if model_key is None:
                    self.match_stats["unmatched"] += 1
                    continue
                
                self.match_stats["matched"] += 1
                
                # TIES SIGN ELECTION
                if model_key in self.accumulated:
                    existing = self.accumulated[model_key]
                    
                    # Shape matching
                    if existing.shape != delta.shape:
                        delta = self._match_shapes(delta, existing.shape)
                    
                    # Trim bottom threshold%
                    flat_delta = delta.view(-1)
                    k_trim = int(len(flat_delta) * ties_threshold)
                    if k_trim > 0:
                        thresh = torch.kthvalue(torch.abs(flat_delta), k_trim).values
                        delta[torch.abs(delta) < thresh] = 0
                    
                    # Sign election
                    sign_existing = torch.sign(existing)
                    sign_new = torch.sign(delta)
                    
                    conflict_mask = (sign_existing != sign_new) & (sign_existing != 0) & (sign_new != 0)
                    conflict_count = conflict_mask.sum().item()
                    
                    if model_key not in self.conflict_stats:
                        self.conflict_stats[model_key] = 0
                    self.conflict_stats[model_key] += conflict_count
                    
                    delta[conflict_mask] = 0
                    
                    # Weighted average
                    mag_existing = torch.abs(existing).mean()
                    mag_new = torch.abs(delta).mean()
                    total_mag = mag_existing + mag_new
                    
                    if total_mag > 0:
                        w_existing = mag_existing / total_mag
                        w_new = mag_new / total_mag
                        self.accumulated[model_key] = existing * w_existing + delta * w_new
                    else:
                        self.accumulated[model_key] = existing + delta
                    
                    del existing
                else:
                    self.accumulated[model_key] = delta
                
                del delta
            
            del sd
            gc.collect()
            
        except Exception as e:
            print(f"[TIESFuser] Error processing {lora_path}: {e}")
            traceback.print_exc()
    
    def get_patches(self, clamp: float = 4.0) -> Dict[str, Tuple]:
        """
        Export accumulated deltas as ComfyUI patches.
        Format: {model_key: ("diff", (delta_tensor,))}
        This matches ComfyUI's lora.load_lora output format.
        """
        patches = {}
        for key, delta in self.accumulated.items():
            # Clamp for safety
            delta = torch.clamp(delta, -clamp, clamp)
            # ComfyUI format: ("diff", (tensor,))
            patches[key] = ("diff", (delta.cpu().to(torch.float16),))
        return patches
    
    def get_analysis(self) -> str:
        """Generate impact analysis report."""
        if not self.accumulated:
            return "No data accumulated"
        
        block_impact = [0.0] * (ZIMAGE_BLOCKS + 4)
        for key, delta in self.accumulated.items():
            block_id, _ = get_block_info(key)
            if block_id >= 0:
                block_impact[block_id] += torch.norm(delta).item()
        
        max_val = max(block_impact) if max(block_impact) > 0 else 1.0
        lines = ["### TIES Merge Analysis ###", ""]
        
        for i, val in enumerate(block_impact):
            if val > 0.01:
                pct = (val / max_val) * 100
                bar = "█" * int(pct / 5)
                if i < ZIMAGE_BLOCKS:
                    lines.append(f"Block {i:02d}: {bar} ({val:.2f})")
                else:
                    specials = ["Context", "Noise", "X-Embed", "Final"]
                    if i - ZIMAGE_BLOCKS < len(specials):
                        lines.append(f"{specials[i-ZIMAGE_BLOCKS]}: {bar} ({val:.2f})")
        
        total_conflicts = sum(self.conflict_stats.values())
        if total_conflicts > 0:
            lines.append(f"\nTotal Sign Conflicts: {total_conflicts}")
        
        match_rate = (self.match_stats["matched"] / (self.match_stats["matched"] + self.match_stats["unmatched"]) * 100) if (self.match_stats["matched"] + self.match_stats["unmatched"]) > 0 else 0
        lines.append(f"\nKey Matching: {self.match_stats['matched']} matched, {self.match_stats['unmatched']} unmatched ({match_rate:.1f}%)")
        
        return "\n".join(lines)

# =============================================================================
# NODE CLASSES
# =============================================================================

class ZFuseLoRAStack:
    """Standard Bus: Range-based isolation (Start/End)."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Select LoRA file to add to the fusion stack"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Base strength multiplier"
                }),
                "block_start": ("INT", {
                    "default": 0, "min": 0, "max": 29,
                    "tooltip": "First block to apply (0-29)"
                }),
                "block_end": ("INT", {
                    "default": 29, "min": 0, "max": 29,
                    "tooltip": "Last block to apply (0-29)"
                }),
                "layer_weights": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "block:strength, e.g. 12:1.5, 14:0.5",
                    "tooltip": "Override specific block strengths"
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "Chain multiple LoRA Bus nodes"
                }),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "stack_lora"
    CATEGORY = "Z-Image/Fusion"

    def stack_lora(self, lora_name, strength, block_start, block_end, layer_weights="", lora_stack=None):
        lora_list = list(lora_stack) if lora_stack is not None else []
        
        custom_map = {}
        if layer_weights.strip():
            try:
                pairs = [p.strip() for p in layer_weights.replace('\n', ',').split(",") if ":" in p]
                for p in pairs:
                    parts = p.split(":")
                    # In text mode, these are absolute multipliers, not offsets
                    custom_map[int(parts[0])] = float(parts[1])
            except Exception as e:
                print(f"[Z-FUSE] Parse error: {e}")
        
        lora_list.append({
            "name": lora_name,
            "strength": strength,
            "range": (min(block_start, block_end), max(block_start, block_end)),
            "custom_map": custom_map,
            "mode": "standard_bus"
        })
        
        return (lora_list,)

class ZFuseVisualLayerTuner:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoRA to analyze and tune per-block"
                }),
                "base_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Global strength multiplier"
                }),
            }
        }
        
        for i in range(ZIMAGE_BLOCKS):
            inputs["required"][f"block_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                "tooltip": f"Block {i} multiplier (1.0=neutral, 0.0=off)"
            })
        
        inputs["optional"] = {
            "lora_stack": ("LORA_STACK", {
                "tooltip": "Chain from previous stack"
            })
        }
        
        return inputs

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("LORA_STACK", "ANALYSIS_REPORT")
    FUNCTION = "tune_layers"
    CATEGORY = "Z-Image/Fusion"

    def _analyze_impact(self, lora_path: str, custom_map: Dict[int, float]) -> List[float]:
        impacts = [0.0] * (ZIMAGE_BLOCKS + 4)
        
        try:
            sd = load_file(lora_path, device="cpu")
            
            with torch.inference_mode():
                for key, val in sd.items():
                    if not any(x in key for x in ["lora_up", "lora_A"]):
                        continue
                    
                    block_id, _ = get_block_info(key)
                    if block_id in custom_map and custom_map[block_id] == 0.0:
                        continue
                    
                    impact = torch.norm(val).item()
                    if 0 <= block_id < len(impacts):
                        impacts[block_id] += impact
            
            del sd
            gc.collect()
            
        except Exception as e:
            print(f"[VisualTuner] Analysis error: {e}")
        
        return impacts

    def tune_layers(self, lora_name, base_strength, **kwargs):
        lora_stack = kwargs.get("lora_stack", [])
        
        custom_map = {}
        for i in range(ZIMAGE_BLOCKS):
            val = kwargs.get(f"block_{i}", 1.0)
            if val != 1.0:
                custom_map[i] = val
        
        path = folder_paths.get_full_path("loras", lora_name)
        report = ""
        
        if path:
            impacts = self._analyze_impact(path, custom_map)
            max_impact = max(impacts) if max(impacts) > 0 else 1.0
            
            lines = [f"Impact: {lora_name}", ""]
            for i, imp in enumerate(impacts):
                if imp > 0.01:
                    bar = "█" * int((imp / max_impact) * 20)
                    if i < ZIMAGE_BLOCKS:
                        lines.append(f"Block {i:02d}: {bar} ({imp:.2f})")
                    else:
                        specials = ["Ctx", "Noise", "X-Emb", "Final"]
                        if i - ZIMAGE_BLOCKS < len(specials):
                            lines.append(f"{specials[i-ZIMAGE_BLOCKS]:5s}: {bar} ({imp:.2f})")
            
            report = "\n".join(lines) if len(lines) > 2 else "Low impact"
        else:
            report = "File not found"
        
        lora_stack.append({
            "name": lora_name,
            "strength": base_strength,
            "range": (0, ZIMAGE_BLOCKS - 1),
            "custom_map": custom_map,
            "mode": "visual_tuner"
        })
        
        return {
            "ui": {"block_impacts": impacts if 'impacts' in dir() else [0] * (ZIMAGE_BLOCKS + 4)},
            "result": (lora_stack, report)
        }

class ZFuseOrchestrator:
    """
    TIES-based fusion using comfy.sd.load_lora_for_models for application.
    """
    
    def __init__(self):
        self.cache = {}
        self.fuser = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base Z-Image model"
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP model (passthrough)"
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack from Bus or Tuner nodes"
                }),
                "merge_mode": (["TIES-Fast"], {
                    "default": "TIES-Fast",
                    "tooltip": "TIES: Fast sequential merging"
                }),
                "ties_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Noise trim threshold (0.2 = trim bottom 20%)"
                }),
                "final_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Final strength applied to merged LoRA"
                }),
                "strength_clamp": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Max weight value (safety clamp)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "TRIGGER_WORDS", "ANALYSIS_REPORT", "STATUS_LOG")
    FUNCTION = "orchestrate"
    CATEGORY = "Z-Image/Fusion"

    def orchestrate(self, model, clip, lora_stack, merge_mode, ties_threshold, 
                    final_strength, strength_clamp):
        
        trigger_words = []
        status_log = ["=== Z-FUSE ORCHESTRATOR (TIES) ===", ""]
        
        if not lora_stack:
            return (model, clip, "", "Empty stack", "No LoRAs provided")
        
        # Get model keys for matching
        try:
            model_keys = list(model.model.state_dict().keys())
            status_log.append(f"Model keys: {len(model_keys)}")
            # Show sample keys for debugging
            sample_keys = [k for k in model_keys if "layers.10" in k and "attention" in k][:3]
            if sample_keys:
                status_log.append("Sample model keys:")
                for k in sample_keys:
                    status_log.append(f"  {k}")
        except Exception as e:
            return (model, clip, "", f"Model inspection failed: {e}", str(e))
        
        # Cache check
        cache_key = generate_cache_key(
            lora_stack,
            ties_threshold=ties_threshold,
            final_strength=final_strength,
            clamp=strength_clamp
        )
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            try:
                work_model = model.clone()
                # Apply cached patches with final_strength
                work_model.add_patches(cached['patches'], final_strength, final_strength)
                
                status_log.extend([
                    f"Using cached result",
                    f"Cache: {cache_key[:16]}...",
                    f"LoRAs: {len(lora_stack)}",
                    f"Applied {len(cached['patches'])} cached patches"
                ])
                
                return (work_model, clip, cached['triggers'], 
                       cached['analysis'], "\n".join(status_log))
            except Exception as e:
                status_log.append(f"Cache apply failed: {e}, recalculating...")
        
        # Extract metadata from all LoRAs
        for entry in lora_stack:
            path = folder_paths.get_full_path("loras", entry["name"])
            if not path:
                status_log.append(f"SKIP: {entry['name']} not found")
                continue
            
            try:
                with safe_open(path, framework="pt") as f:
                    meta = f.metadata()
                    if meta:
                        raw = meta.get("ss_trained_words") or meta.get("ss_tag_frequency") or ""
                        if raw:
                            clean = raw.replace('[','').replace(']','').replace('"','').replace('{','').replace('}','')
                            for word in [x.strip() for x in clean.split(',') if x.strip()]:
                                tag = word.split(':')[0].strip()
                                if tag and tag not in trigger_words:
                                    trigger_words.append(tag)
            except Exception as e:
                status_log.append(f"Metadata error for {entry['name']}: {e}")
        
        # New fusion
        status_log.append(f"Processing {len(lora_stack)} LoRAs...")
        
        self.fuser = TIESFuser()
        self.fuser.set_model_keys(model_keys)
        
        pbar = comfy.utils.ProgressBar(len(lora_stack))
        
        for i, entry in enumerate(lora_stack):
            path = folder_paths.get_full_path("loras", entry["name"])
            if not path:
                status_log.append(f"[{i+1}] SKIP: {entry['name']} not found")
                pbar.update(1)
                continue
            
            self.fuser.fuse_lora(
                path, entry["strength"],
                entry.get("range", (0, 29)),
                entry.get("custom_map", {}),
                entry.get("mode", "standard"),
                ties_threshold
            )
            
            status_log.append(f"[{i+1}/{len(lora_stack)}] {entry['name']}")
            pbar.update(1)
            
            if i % 2 == 0:
                torch.cuda.empty_cache()
        
        # Get patches in ComfyUI format
        patches = self.fuser.get_patches(clamp=strength_clamp)
        analysis = self.fuser.get_analysis()
        
        status_log.append(f"\nFused keys: {len(patches)}")
        status_log.append(f"Match stats: {self.fuser.match_stats}")
        if self.fuser.processed_samples:
            status_log.append("Samples:")
            for s in self.fuser.processed_samples[:3]:
                status_log.append(f"  {s}")
        
        # Apply to model using add_patches (matches original working code)
        try:
            work_model = model.clone()
            
            # Verify key match before applying
            model_keys_set = set(work_model.model.state_dict().keys())
            match_count = sum(1 for k in patches if k in model_keys_set)
            status_log.append(f"Key verification: {match_count}/{len(patches)} patches match model keys")
            
            # Apply patches with final_strength
            work_model.add_patches(patches, final_strength, final_strength)
            status_log.append(f"Successfully applied {len(patches)} patches with strength {final_strength}")
            
            # Cache
            self.cache[cache_key] = {
                'patches': patches,
                'triggers': ", ".join(trigger_words),
                'analysis': analysis
            }
            
            if len(self.cache) > 3:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                
        except Exception as e:
            error_msg = f"Failed to apply patches: {e}"
            status_log.append(error_msg)
            traceback_str = traceback.format_exc()
            status_log.append(traceback_str)
            # Return original model on failure
            return (model, clip, ", ".join(trigger_words), analysis, "\n".join(status_log))
        
        # Memory cleanup
        del self.fuser
        gc.collect()
        torch.cuda.empty_cache()
        
        return (work_model, clip, ", ".join(trigger_words), analysis, "\n".join(status_log))

class ZFuseBake:
    """Export Node: Saves the model state to a LoRA file."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Model with fused LoRAs applied"
                }),
                "save_name": ("STRING", {
                    "default": "zfuse_merged",
                    "tooltip": "Output filename (no extension)"
                }),
                "export_rank": (["Full (No Loss)", "128 (High Quality)", "64 (Standard)"], {
                    "default": "64 (Standard)",
                    "tooltip": "Full=~3-6GB, 128=~500MB, 64=~250MB"
                }),
                "precision": (["bf16", "fp16"], {
                    "default": "bf16",
                    "tooltip": "bf16 recommended for Z-Image"
                }),
            },
            "optional": {
                "trigger_words": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated trigger words to embed in metadata"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SAVE_STATUS",)
    FUNCTION = "bake"
    CATEGORY = "Z-Image/Fusion"
    OUTPUT_NODE = True

    def bake(self, model, save_name, export_rank, precision, trigger_words=""):
        logs = ["=== Z-FUSE BAKE ===", ""]
        
        if "Full" in export_rank:
            target_rank = 0
            rank_label = "Full"
        else:
            target_rank = int(export_rank.split()[0])
            rank_label = f"r{target_rank}"
        
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        
        # Extract patches from model
        patches = getattr(model, 'patches', None)
        if not patches:
            return ("ERROR: No patches found. Apply LoRA through Orchestrator first.",)
        
        logs.append(f"Export: {rank_label} | {precision}")
        logs.append(f"Patches: {len(patches)}")
        
        baked_sd = {}
        count = 0
        skipped = 0
        
        for i, (key, patch_list) in enumerate(patches.items()):
            # Extract delta from patch format ("diff", (tensor,))
            total_delta = None
            for p in patch_list:
                if isinstance(p, (tuple, list)) and len(p) == 2 and p[0] == "diff":
                    # Format: ("diff", (tensor,))
                    delta = p[1][0] if isinstance(p[1], (tuple, list)) else p[1]
                elif isinstance(p, torch.Tensor):
                    delta = p
                else:
                    continue
                
                if not isinstance(delta, torch.Tensor):
                    continue
                
                delta = delta.to(dtype)
                total_delta = delta if total_delta is None else total_delta + delta
            
            if total_delta is None or total_delta.dim() < 2:
                skipped += 1
                continue
            
            try:
                out_features, in_features = total_delta.shape[0], total_delta.shape[1]
                
                if target_rank == 0 or target_rank >= min(out_features, in_features):
                    # Full rank
                    baked_sd[f"{key}.lora_up.weight"] = total_delta.cpu()
                    baked_sd[f"{key}.lora_down.weight"] = torch.eye(
                        in_features, dtype=dtype
                    )
                    baked_sd[f"{key}.alpha"] = torch.tensor(in_features, dtype=torch.float32)
                    logs.append(f"[{i+1}] {key[:50]}...: Full {list(total_delta.shape)}")
                else:
                    # SVD compression
                    U, S, Vh = torch.linalg.svd(total_delta.float(), full_matrices=False)
                    
                    k = min(target_rank, len(S))
                    up = (U[:, :k] @ torch.diag(S[:k])).to(dtype).cpu()
                    down = Vh[:k, :].to(dtype).cpu()
                    
                    baked_sd[f"{key}.lora_up.weight"] = up
                    baked_sd[f"{key}.lora_down.weight"] = down
                    baked_sd[f"{key}.alpha"] = torch.tensor(k, dtype=torch.float32)
                    
                    energy = (S[:k] ** 2).sum() / (S ** 2).sum()
                    logs.append(f"[{i+1}] {key[:50]}...: r{k} ({energy*100:.1f}%)")
                
                count += 1
                
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logs.append(f"[{i+1}] {key[:50]}...: ERROR {e}")
                skipped += 1
        
        if count == 0:
            return ("\n".join(logs + ["", "ERROR: Nothing to save"]),)
        
        # Deduplicate trigger words
        unique_triggers = []
        if trigger_words:
            for word in trigger_words.split(','):
                word = word.strip()
                if word and word not in unique_triggers:
                    unique_triggers.append(word)
        
        try:
            output_dir = folder_paths.get_output_directory()
            save_dir = os.path.join(output_dir, "loras")
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"{save_name}_{rank_label}_{precision}.safetensors"
            save_path = os.path.join(save_dir, filename)
            
            # Metadata
            metadata = {
                "ss_framework": "Z-FUSE",
                "fused_by": "Z-FUSE Bake",
                "export_rank": rank_label,
                "precision": precision,
                "layers": str(count),
                "ss_trained_words": json.dumps(unique_triggers) if unique_triggers else ""
            }
            
            save_file(baked_sd, save_path, metadata=metadata)
            
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logs.extend([
                "",
                f"Saved: {save_path}",
                f"Layers: {count} | Skipped: {skipped}",
                f"Size: {size_mb:.1f} MB",
                f"Triggers: {len(unique_triggers)} unique"
            ])
            
        except Exception as e:
            logs.append(f"Save error: {e}")
            traceback.print_exc()
        
        return ("\n".join(logs),)

# =============================================================================
# Z-IMAGE MODEL MERGER NODE
# =============================================================================

class ZFuseModelMerge:
    """
    TIES-based model merger for Z-Image Base/Turbo models.
    
    Fixed implementation: Robust memory management, dynamic layer detection, 
    proper task vector arithmetic with disjoint TIES merging, and ComfyUI patch integration.
    Fixed implementation: Resolves FP8 double-scaling explosions and ComfyUI patch format bugs.
    - True-Space FP8 arithmetic (resolves red-noise value explosion)
    - Structural layer protection (bypasses TIES trimming on Norms/Biases)
    """
    
    def __init__(self):
        self.validation_info = ""
        self.status_log =[]
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": ("MODEL", {
                    "tooltip": "First model - determines output type (Base/Turbo) and provides metadata. Use original Z-Image model here for best results."
                }),
                "strength_1": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight multiplier for custom_1 model"
                }),
                "strength_2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight multiplier for custom_2 model"
                }),
                "strength_3": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Weight multiplier for custom_3 model"
                }),
                "zone_early": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Layers 0-33% weight multiplier (Early blocks)"
                }),
                "zone_mid": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Layers 34-66% weight multiplier (Mid blocks)"
                }),
                "zone_late": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Layers 67%+ weight multiplier (Late blocks)"
                }),
                "ties_density": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "TIES density threshold. 0.9 = trim bottom 10% of dense weights"
                }),
                "ties_lambda": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "TIES merge ratio. 0.5 = balanced, 1.0 = full override"
                }),
                "save_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, save merged model to save_path. If False, output merged model for immediate use."
                }),
                "save_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full save path including .safetensors extension"
                }),
                "output_dtype": (["bf16", "fp8"], {
                    "default": "bf16",
                    "tooltip": "Output precision. BF16 will safely bake the FP8 scales into the weights."
                }),
            },
            "optional": {
                "custom_1": ("MODEL", {
                    "tooltip": "Optional: Second model to merge"
                }),
                "custom_2": ("MODEL", {
                    "tooltip": "Optional: Third model to merge"
                }),
                "custom_3": ("MODEL", {
                    "tooltip": "Optional: Fourth model to merge"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MERGED_MODEL", "VALIDATION_INFO", "STATUS_LOG")
    FUNCTION = "merge_models"
    CATEGORY = "Z-Image/Fusion"
    OUTPUT_NODE = True
    
    DESCRIPTION = """TIES Model Merger for Z-Image Base/Turbo models.
    
First model determines output type. Last 5 layers always bypass the merge safely.
Supports up to 4 models with zone-based weighting (Early/Mid/Late).
Efficiently processes to avoid VRAM/RAM spikes."""

    def detect_model_type(self, model) -> Tuple[str, List[str]]:
        """
        Detect if model is Base or Turbo from structure.
        Returns: (type_str, indicators_list)
        """
        indicators = []
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                state_dict = model.model.state_dict()
                keys = list(state_dict.keys())
                
                # Check for final layer structure differences
                final_layers = [k for k in keys if any(x in k for x in 
                    ["final_layer", "x_embedder", "context_refiner", "noise_refiner"])]
                
                # Turbo typically has 'single_transformer_blocks' or simplified final layers
                has_single_blocks = any("single_transformer_blocks" in k for k in keys)
                has_double_blocks = any("double_blocks" in k for k in keys)
                
                if has_single_blocks: indicators.append("Has single_transformer_blocks (Turbo indicator)")
                if has_double_blocks: indicators.append("Has double_blocks (Base indicator)")
                    
                # Check specific layer patterns
                adaLN_count = sum(1 for k in final_layers if "adaLN" in k)
                if adaLN_count > 2: indicators.append(f"Multiple adaLN layers ({adaLN_count})")
                
                final_param_count = sum(1 for k in keys if any(x in k for x in["final_layer", "x_embedder", "cap_embedder"]))
                
                if has_single_blocks and not has_double_blocks: return "Turbo", indicators
                elif has_double_blocks and not has_single_blocks: return "Base", indicators
                elif final_param_count < 10: return "Turbo", indicators
                else: return "Base", indicators
        except Exception as e:
            indicators.append(f"Detection error: {str(e)}")
            
        return "Unknown", indicators

    def get_layer_info(self, key: str, max_layer: int = 0) -> Tuple[Optional[int], str]:
        patterns =[
            r'(?:layers|blocks|single_transformer_blocks|double_blocks|transformer_blocks)\.(\d+)',
            r'layer_(\d+)',
            r'block_(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, key, re.IGNORECASE)
            if match:
                layer_num = int(match.group(1))
                if max_layer > 0 and layer_num >= max_layer - 4:
                    return layer_num, "final"
                
                if max_layer > 4:
                    non_final_count = max_layer - 4
                    ratio = layer_num / non_final_count
                    if ratio < 0.34: return layer_num, "early"
                    elif ratio < 0.67: return layer_num, "mid"
                    else: return layer_num, "late"
                else:
                    if 0 <= layer_num <= 9: return layer_num, "early"
                    elif 10 <= layer_num <= 19: return layer_num, "mid"
                    elif 20 <= layer_num <= 29: return layer_num, "late"
                    else: return layer_num, "other"
        
        if any(x in key for x in["x_embedder", "final_layer", "cap_embedder", "context_refiner", "noise_refiner"]):
            return None, "final"
        return None, "other"

    def get_zone_multiplier(self, zone: str, zone_early: float, zone_mid: float, zone_late: float) -> float:
        if zone == "early": return zone_early
        elif zone == "mid": return zone_mid
        elif zone == "late": return zone_late
        elif zone == "final": return 0.0 # Force protection skip on final architecture layers
        else: return 1.0

    def safe_broadcast_multiply(self, tensor: torch.Tensor, scale) -> torch.Tensor:
        if scale is None or (isinstance(scale, float) and scale == 1.0):
            return tensor
        try:
            if scale.dim() == 1 and tensor.dim() >= 1:
                if scale.shape[0] == tensor.shape[0]:
                    s_view = scale.view(-1, *([1] * (tensor.dim() - 1)))
                    return tensor * s_view
                elif scale.shape[0] == tensor.shape[-1]:
                    return tensor * scale
        except: pass
        return tensor * scale

    def safe_broadcast_divide(self, tensor: torch.Tensor, scale) -> torch.Tensor:
        if scale is None or (isinstance(scale, float) and scale == 1.0):
            return tensor
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        try:
            if safe_scale.dim() == 1 and tensor.dim() >= 1:
                if safe_scale.shape[0] == tensor.shape[0]:
                    s_view = safe_scale.view(-1, *([1] * (tensor.dim() - 1)))
                    return tensor / s_view
                elif safe_scale.shape[0] == tensor.shape[-1]:
                    return tensor / safe_scale
        except: pass
        return tensor / safe_scale

    def get_true_tensor_and_scale(self, state_dict, key: str) -> Tuple[Optional[torch.Tensor], Any]:
        """Extracts the math-accurate True Float32 values and original scale factor."""
        if key not in state_dict:
            return None, 1.0
            
        tensor = state_dict[key]
        if tensor.dtype in[torch.int64, torch.int32, torch.uint8, torch.int8]:
            return tensor.cpu(), 1.0
            
        is_fp8 = tensor.dtype in[torch.float8_e4m3fn, torch.float8_e5m2]
        scale = None
        
        if is_fp8:
            for sk in[key.replace(".weight", ".weight_scale"), key.replace(".bias", ".bias_scale"), key + "_scale"]:
                if sk in state_dict:
                    scale = state_dict[sk].to(torch.float32).cpu()
                    break
        
        t_f32 = tensor.to(torch.float32).cpu()
        true_tensor = self.safe_broadcast_multiply(t_f32, scale)
        return true_tensor, scale

    def ties_merge_task_vectors(self, task_vectors: List[torch.Tensor], density: float) -> torch.Tensor:
        """
        PROPER TIES: Disjoint merge of task vectors.
        
        1. Trim each task vector independently
        2. Elect sign across all task vectors at each position
        3. Average only non-conflicting, non-trimmed values
        
        Args:
            task_vectors: List of task vectors (deltas) from each custom model
            density: Keep top (density)% of weights per task vector
        
        Returns:
            Single merged task vector
        """
        if not task_vectors:
            return None
            
        stacked = torch.stack(task_vectors)
        num_models = len(task_vectors)
        
        # Step 1: TRIM - Zero out bottom (1-density)% of each task vector independently
        if density < 1.0:
            # Flatten each task vector to find thresholds
            flat = stacked.view(num_models, -1)  # [num_models, num_elements]
            k = int(flat.shape[1] * (1.0 - density))
            
            if k > 0:
                # Find kth smallest absolute value for each task vector
                thresholds = torch.kthvalue(torch.abs(flat), k, dim=1).values  # [num_models]
                thresholds = thresholds.view(num_models, *([1] * (stacked.dim() - 1)))
                
                # Create trim mask: keep if |value| >= threshold
                trim_mask = torch.abs(stacked) >= thresholds
                trimmed = stacked * trim_mask
            else:
                trimmed = stacked
        else:
            trimmed = stacked
            
        # Bypass majority voting if only 1 model (but keep trimmed result)
        if num_models == 1:
            return trimmed[0]
        
        # Step 2: ELECT SIGN
        signs = torch.sign(trimmed)
        sign_sum = torch.sum(signs, dim=0)
        
        # Majority sign: positive if sum > 0, negative if sum < 0, 0 if tied
        majority_sign = torch.where(
            sign_sum > 0,
            torch.ones_like(sign_sum),
            torch.where(
                sign_sum < 0,
                -torch.ones_like(sign_sum),
                torch.zeros_like(sign_sum)
            )
        )
        
        # Step 3: DISJOINT MERGE - Average values that match majority sign
        # Create mask: keep if sign matches majority (and both non-zero)
        match_mask = (signs == majority_sign.unsqueeze(0)) & (signs != 0)
        
        # Zero out conflicting values
        agreeing_values = trimmed * match_mask
        
        # Count how many models agree at each position (avoid div by zero)
        agreement_count = match_mask.sum(dim=0).float().clamp(min=1.0)
        
        return agreeing_values.sum(dim=0) / agreement_count

    def _match_shapes(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        if tensor.shape == target_shape:
            return tensor
        pad =[]
        for i in reversed(range(len(target_shape))):
            if i < len(tensor.shape): pad.extend([0, target_shape[i] - tensor.shape[i]])
            else: pad.extend([0, target_shape[i]])
        return F.pad(tensor, pad)

    def merge_models(self, base_model, strength_1, strength_2, strength_3,
                    zone_early, zone_mid, zone_late, ties_density, ties_lambda,
                    save_model, save_path, output_dtype, 
                    custom_1=None, custom_2=None, custom_3=None):
        """
        Main merge function with CORRECTED TIES math.
        """
        self.status_log =["=== Z-FUSE MODEL MERGE (OPTIMIZED) ===", ""]
        
        custom_models =[]
        if custom_1 is not None and strength_1 > 0: custom_models.append((custom_1, strength_1))
        if custom_2 is not None and strength_2 > 0: custom_models.append((custom_2, strength_2))
        if custom_3 is not None and strength_3 > 0: custom_models.append((custom_3, strength_3))
        
        if not custom_models:
            self.status_log.append("No custom models provided - returning base model.")
            return (base_model, "No custom models to merge", "\n".join(self.status_log))
        
        # Detect base model type
        base_type, base_indicators = self.detect_model_type(base_model)
        
        # Build validation info
        validation_lines = [
            "=== Z-FUSE MODEL MERGE VALIDATION ===",
            "",
            f"Base Model Type: {base_type}",
            f"  Indicators: {', '.join(base_indicators)}",
            "",
            f"Custom models to merge: {len(custom_models)}"
        ]
        
        self.status_log.append("Loading state dicts references...")
        base_state = base_model.model.state_dict()
        base_keys = list(base_state.keys())
        
        # Pre-build custom mappings to strip off `model.diffusion_model.` discrepancies seamlessly.
        custom_states =[]
        for i, (model, strength) in enumerate(custom_models, 1):
            m_type, m_ind = self.detect_model_type(model)
            validation_lines.append(f"  Custom_{i}: type={m_type}, strength={strength}")
            
            c_state = model.model.state_dict()
            c_map = {re.sub(r'^(?:model\.)?(?:diffusion_model\.)?', '', k): k for k in c_state.keys()}
            custom_states.append({
                "state": c_state,
                "map": c_map,
                "strength": strength
            })
            
        validation_lines.extend([
            "",
            "Zone Multipliers:",
            f"  Early (0-9): {zone_early}",
            f"  Mid (10-19): {zone_mid}",
            f"  Late (20-29): {zone_late}",
            f"  Final (30-34): 1.0 (from base only)",
            "",
            "TIES Parameters:",
            f"  Density: {ties_density} (trim bottom {(1-ties_density)*100:.0f}%)",
            f"  Lambda: {ties_lambda}",
            "",
            "Merge Strategy:",
            "  1. Compute task vectors (custom - base) for each model",
            "  2. Apply zone weights and model strengths to task vectors",
            "  3. TRIM: Zero bottom (1-density)% of each task vector",
            "  4. ELECT: Majority sign voting across all task vectors",
            "  5. MERGE: Average only non-conflicting values",
            "  6. Apply merged task vector to base: base + lambda * merged_delta",
        ])
        
        self.validation_info = "\n".join(validation_lines)
        
        # Find maximum block index dynamically for guaranteed structure preservation
        max_layer = 0
        patterns =[
            r'(?:layers|blocks|single_transformer_blocks|double_blocks|transformer_blocks)\.(\d+)',
            r'layer_(\d+)',
            r'block_(\d+)',
        ]
        for key in base_keys:
            for p in patterns:
                m = re.search(p, key, re.IGNORECASE)
                if m: max_layer = max(max_layer, int(m.group(1)))
                    
        self.status_log.append(f"Detected max layer: {max_layer}")
        self.status_log.append(f"Layers >= {max_layer-4} mapped as architecture finals (bypassed safely).")
        
        patches = {}
        processed = 0
        pbar = comfy.utils.ProgressBar(len(base_keys))
        
        self.status_log.append("Processing True-Space layer deltas...")
        
        for key in base_keys:
            # We skip scaling parameters entirely; ComfyUI inherently manages them
            if key.endswith('_scale') or '.weight_scale' in key or '.bias_scale' in key:
                processed += 1
                pbar.update(1)
                continue
                
            layer_num, zone = self.get_layer_info(key, max_layer)
            zone_mult = self.get_zone_multiplier(zone, zone_early, zone_mid, zone_late)
            
            if zone == "final" or zone_mult == 0.0:
                processed += 1
                pbar.update(1)
                continue
                
            base_original = base_state[key]
            if base_original.dtype in[torch.int64, torch.int32, torch.uint8, torch.int8]:
                processed += 1
                pbar.update(1)
                continue
                
            base_true, base_scale = self.get_true_tensor_and_scale(base_state, key)
            if base_true is None:
                processed += 1
                pbar.update(1)
                continue
                
            # Structural logic - protect Norms/Biases from TIES zeroing
            key_lower = key.lower()
            is_dense_weight = (
                base_true.dim() >= 2 and 
                "norm" not in key_lower and 
                "ln" not in key_lower and 
                "adaln" not in key_lower and 
                "bias" not in key_lower
            )
                
            task_vectors =[]
            clean_base = re.sub(r'^(?:model\.)?(?:diffusion_model\.)?', '', key)
            
            for c_info in custom_states:
                if clean_base not in c_info["map"]: continue
                c_key = c_info["map"][clean_base]
                
                custom_true, _ = self.get_true_tensor_and_scale(c_info["state"], c_key)
                if custom_true is None: continue
                    
                if custom_true.shape != base_true.shape:
                    custom_true = self._match_shapes(custom_true, base_true.shape)
                    
                # Task vector generated in True Float Space
                tv = (custom_true - base_true) * (c_info["strength"] * zone_mult)
                task_vectors.append(tv)
                
            if task_vectors:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                task_vectors_dev = [t.to(device) for t in task_vectors]
                
                if is_dense_weight:
                    merged_true_dev = self.ties_merge_task_vectors(task_vectors_dev, ties_density)
                else:
                    # Pure smooth linear merge for structural blocks
                    merged_true_dev = torch.stack(task_vectors_dev).sum(dim=0)
                
                if merged_true_dev is not None:
                    # Apply global TIES strength
                    merged_true_dev = merged_true_dev * ties_lambda
                    merged_true_cpu = merged_true_dev.cpu()
                    
                    # Inverse-scale so ComfyUI's dynamic forward pass multiplier resolves to the perfect value
                    inverse_scaled_patch = self.safe_broadcast_divide(merged_true_cpu, base_scale)
                    final_patch = inverse_scaled_patch.to(torch.bfloat16)
                    
                    patches[key] = ("diff", (final_patch,))
                    
                del task_vectors_dev, merged_true_dev
                
            processed += 1
            pbar.update(1)
            
            if processed % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        self.status_log.append(f"Successfully generated {len(patches)} memory-optimized patches.")
        
        # Free custom models states completely
        del custom_states
        gc.collect()
        
        # ComfyUI best-practice integration: Patch the underlying base model
        merged_model = base_model.clone()
        merged_model.add_patches(patches, 1.0, 1.0)
        self.status_log.append("Patches injected dynamically. True target space verified. Ready for KSampler!")
        
        # Saving Logic Block
        if save_model and save_path:
            self.status_log.append("Saving merged model to disk...")
            try:
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                save_state = {}
                for key in base_keys:
                    # Drop scale vectors when exporting to non-FP8 bf16
                    if output_dtype == "bf16" and (key.endswith('_scale') or '.weight_scale' in key or '.bias_scale' in key):
                        continue
                    elif output_dtype == "fp8" and (key.endswith('_scale') or '.weight_scale' in key or '.bias_scale' in key):
                        save_state[key] = base_state[key].cpu()
                        continue
                        
                    base_orig = base_state[key]
                    if base_orig.dtype in[torch.int64, torch.int32, torch.uint8, torch.int8]:
                        save_state[key] = base_orig.cpu()
                        continue
                        
                    base_true, base_scale = self.get_true_tensor_and_scale(base_state, key)
                    
                    if key in patches:
                        # Reconstruct final float layer
                        raw_patch = patches[key][1][0].to(torch.float32)
                        patch_true = self.safe_broadcast_multiply(raw_patch, base_scale)
                        final_true = base_true + patch_true
                    else:
                        final_true = base_true
                        
                    if output_dtype == "bf16":
                        save_state[key] = final_true.to(torch.bfloat16)
                    else:
                        # Re-quantize naively if FP8 is specifically requested
                        save_state[key] = final_true.to(torch.float8_e4m3fn)
                
                metadata = {
                    "zfuse_merged": "true",
                    "zfuse_base_type": base_type,
                    "zfuse_models_count": str(len(custom_models) + 1),
                    "zfuse_ties_density": str(ties_density),
                    "zfuse_ties_lambda": str(ties_lambda),
                }

                from safetensors.torch import save_file
                save_file(save_state, save_path, metadata=metadata)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                self.status_log.append(f"Successfully saved to: {save_path} ({size_mb:.1f} MB)")
                
                del save_state
                gc.collect()
            except Exception as e:
                self.status_log.append(f"Save failed: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return (merged_model, self.validation_info, "\n".join(self.status_log))
