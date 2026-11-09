# FidelityFX SDK (subset)

The FSR 3.1 upscaler's GPU code from AMD's FidelityFX SDK, used in place of a game's DLSS
evaluation (`src/video_core/renderer_vulkan/vk_fsr3_upscaler.cpp`), and the FSR 3 frame
interpolation and optical flow code, used by the AMD FSR 3 frame generation
(`src/video_core/renderer_vulkan/present/amd_fsr3_frame_gen.cpp`).

- Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v1.1.4`
  (commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`), FSR 3.1.4 upscaler.
- Licence: MIT, `LICENSE.txt` (the SDK's `sdk/LICENSE.txt`).
- `include/FidelityFX/gpu/`: from `sdk/include/FidelityFX/gpu/` (the core headers, `fsr3upscaler/`
  without the HLSL callbacks and the debug view, `spd/ffx_spd.h`, `fsr1/ffx_fsr1.h`, `frameinterpolation/`, `opticalflow/`).
- `shaders/fsr3upscaler/`: the Vulkan pass files from `sdk/src/backends/vk/shaders/fsr3upscaler/`
  (the eight upscaling passes; not the autogen-reactive or debug-view pass).
- `shaders/frameinterpolation/`, `shaders/opticalflow/`: the Vulkan pass files from the same
  directory (frame interpolation without the debug view; optical flow v5).

The files are unmodified. They are compiled to SPIR-V by `src/video_core/host_shaders`; no
prebuilt binary or AMD tool from the SDK is used.
