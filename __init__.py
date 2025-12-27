from .z_fuse_node import ZFuseLoRAStack, ZFuseVisualLayerTuner, ZFuseOrchestrator, ZFuseBake

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]