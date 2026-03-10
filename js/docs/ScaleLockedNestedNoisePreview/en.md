# Scale-Locked Nested Noise Preview

Debug/utility node for inspecting the nested-noise construction used by Scale-Locked Residual Diffusion.

## Inputs

- `latent_image`: target high-resolution latent
- `seed`: base seed used to derive the low-res and high-res noise fields
- `target_megapixels`: planner resolution in pixel-space MP
- `nested_noise_strength`: strength of the extra high-frequency residual noise

## Outputs

- `nested_noise_latent`: preview latent containing the constructed nested noise field
- `lowres_reference_latent`: resized low-resolution reference latent used for the planner branch

## Notes

- This node is for inspection and debugging.
- The coarse structure should match the low-res reference branch, while high-frequency noise remains free to add detail.
