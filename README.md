# Z-Fuse
Z-FUSE is a hi-fi LoRA orchestrator for Z-Image in ComfyUI. Standard loaders suffer from activation saturation &amp; "bad gens" from sequential application, Z-FUSE utilizes a Simultaneous Bus Architecture &amp; TIES-Merging logic. Fusing multi LoRAs into a single math delta before application, preventing structural collapse in single-stream architectures.

Z-FUSE: Surgical LoRA Orchestrator
A ComfyUI node suite designed for high-fidelity merging and manipulation of LoRAs for the Z-Image (S3-DiT) architecture.

Z-FUSE moves beyond sequential loading, utilizing simultaneous TIES-fusion to prevent "blurry soup" and structural collapse in single-stream diffusion models.

Features
🚀 Simultaneous Fusion Engine
Replaces destructive sequential patching with a unified mathematical fusion.

TIES-Merging: Resolves "sign conflicts" between LoRAs, pruning noise and ensuring clean output.
Weighted Sum: Standard additive merging for simple combinations.
Architecture Aware: Specifically optimized for Z-Image's Scalable Single-Stream Diffusion Transformer (S3-DiT).
🧬 Visual Layer Tuner (VLT)
Gain surgical control over the Transformer Spine.

30-Block Isolation: Target specific regions (Early, Mid, Late) of the generative process.
Granular Weighting: Apply modifiers from -5.0 to +5.0 per block.
Heatmap Analysis: Visualize exactly which blocks are driving the visual impact.
Zero-Skip Optimization: Blocks set to 0.0 are skipped entirely to save system resources.
🛠️ Z-FUSE Bus (Stacker)
Stack an unlimited number of LoRAs efficiently.

Range Masking: Define start and end blocks to limit LoRA influence.
Text Overrides: Surgical overrides via string input (e.g., 12:1.5).
📊 Transparency & Diagnostics
Stop working blind.

Analysis Reports: Get a human-readable breakdown of block intensity.
Trigger Harvesting: Automatically extracts trigger words from LoRA metadata.
Status Logs: Detailed reporting on mapping success, key matching, and errors.
🔒 No-Loss Bake (Export)
Export your optimized stack to a standalone .safetensors file.

Full-Rank Preservation: Avoids the quality loss of SVD compression.
Precision Control: Export in FP16, BF16, or FP32.
Installation
Navigate to your ComfyUI custom_nodes directory.
Clone this repository:
git clone https://github.com/destinyfaux/Z-Fuse.git
Restart ComfyUI.
Quick Start
Load Model: Use the standard CheckpointLoaderSimple.
Add LoRAs: Use Z-FUSE: Visual Layer Tuner or Z-FUSE: LoRA Bus (Stacker) to load your files. Connect them in a chain to build your stack.
Orchestrate: Connect the stack output to Z-FUSE: Surgical Orchestrator. Connect the model to the model input.
Execute: Run the queue. The Orchestrator will fuse the weights and output the patched model.
Analyze: Check the ANALYSIS_REPORT output to see which blocks are "hot".
Refine: Adjust block sliders in the Tuner nodes to temper or amplify specific layers.
Bake (Optional): Pass the patched model to Z-FUSE: No-Loss Bake to save your fusion as a new LoRA.
Architecture Notes
Z-FUSE is specifically tuned for Z-Image / S3-DiT.

Blocks 0-8 (Input Head): Composition and global framing.
Blocks 9-20 (Transformer Spine): Anatomical structure and spatial relationships.
Blocks 21-30 (Output Head): Texture, skin details, and lighting.
Recommended Settings
Max Clamp: The node uses an element-wise clamp of +/- 4.0. This allows for significant impact (up to 40x standard strength) while preventing "frying" or NaN errors.
Negative Weights: Supported. Use negative values (e.g., -0.5) to subtract features or invert a LoRA's influence on specific blocks.
Memory: The node aggressively offloads tensors to System RAM during the stacking process to save VRAM. Ensure you have sufficient System RAM for large stacks.
Contributing
Contributions, bug reports, and feature requests are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Credits
Developed for the Z-Image community.
