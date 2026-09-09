# water-surface

Real-time liquid water and water ray-tracing on an **infinite lake surface**, rendered with WebGPU compute shaders via [`euv`](https://crates.io/crates/euv) and [`euv-engine`](https://crates.io/crates/euv-engine).

## Demo

Wind blows over a procedurally tiled ocean. The surface lifts into Gerstner-style ripples, the sky's sun and clouds reflect across the plane (with the reflection breaking up as the wave height grows), and a Fresnel-style mix between refracted deep-water color and sky reflection gives the surface its real-world look.

- **Single finger drag** — orbit the camera around the lake
- **Two finger pinch** — zoom toward / away from the lake surface
- The plane is mathematically infinite — only the visible tile around the camera is allocated as a 256×256 mesh, and the shader computes sky / wave contribution for any world-space XZ coordinate.

## Build

```bash
# dev
euv build

# release
euv build --release

# serve locally on :8080
python3 -m http.server --directory www 8080
```

## Technical notes

- **Water simulation** — WebGPU compute shader. Two storage textures (current height + previous height) are ping-ponged each tick. Each cell integrates the 2D wave equation with damping. Touch impulses on the lake surface inject a Gaussian height bump that propagates outward as a ripple.
- **Wave surface shading** — fragment shader reads the displacement texture, samples a procedural Gerstner-style normal, then computes Fresnel + sky reflection + sun specular + procedural clouds + foam.
- **Sky** — procedural in the same fragment shader. Rayleigh-style horizon, sun disc with halo, procedural cumulus cloud field driven by FBM noise.
- **Performance** — 256×256 vertex grid around the camera. Wave simulation only updates cells within a sliding window centered on the camera so the lake is effectively infinite without unbounded GPU memory.

## Stack

- Rust 2024 + WebAssembly
- euv 0.20.x (`crates.io`)
- euv-engine 0.20.x (`crates.io`)
- WebGPU (WGSL compute + render pipelines)

## Status

Render path is implemented end-to-end with the euv-engine 0.20.6+
low-level WebGPU API surface (`create_command_encoder` /
`begin_render_pass` / `set_pipeline` / `set_bind_group` /
`set_vertex_buffer` / `set_index_buffer` / `draw_indexed` /
`end_render_pass` / `finish_command_encoder` / `submit`). `cargo build`
is zero-warning, the compiled wasm is 362 KB, and a headless Chromium
launched with `--enable-unsafe-webgpu` reports `WebGPU Active`,
canvas context = `webgpu`, and a stable RAF loop.
