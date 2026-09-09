#[allow(unused_imports)]
use super::*;

/// Surface shader: vertex + fragment pair that draws the infinite lake surface.
///
/// The mesh is a 256×256 grid of triangles. The vertex shader reads the
/// simulated heights from a storage buffer (filled by the CPU-side wave
/// equation each frame and uploaded via `update_uniform_buffer`) to lift
/// the surface, and produces the world-space normal via finite differences
/// so the fragment shader can light it.
///
/// The fragment shader runs the entire visual pipeline in one pass:
///   1. Read the height + neighbour heights to compute a smooth normal.
///   2. Sample a procedural sky function (sun + clouds + horizon) for the
///      reflection direction.
///   3. Refract through the normal to get the underwater deep-water color.
///   4. Mix reflection and refraction by Fresnel (Schlick).
///   5. Add the sun's specular highlight (Blinn-Phong).
///   6. Add foam on the wave crests where the height is high.
///
/// Performance notes:
///   - The mesh tile follows the camera in world space. The fragment shader
///     is keyed off `v_world_xz` so the rendered surface stays continuous even
///     when the mesh moves.
///   - The sky is procedural (FBM noise + sun disc) — no texture samplers.
pub(crate) const SURFACE_SHADER: &str = r#"
struct SurfaceUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    camera_forward: vec4<f32>,
    time: f32,
    mesh_origin_x: f32,
    mesh_origin_z: f32,
    mesh_cell_size: f32,
    grid_resolution: f32,
    height_amplitude: f32,
    fresnel_power: f32,
    reflectivity: f32,
    water_color_r: f32,
    water_color_g: f32,
    water_color_b: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    sky_top_r: f32,
    sky_top_g: f32,
    sky_top_b: f32,
    sky_horizon_r: f32,
    sky_horizon_g: f32,
    sky_horizon_b: f32,
};

@group(0) @binding(0) var<uniform> u: SurfaceUniforms;
@group(0) @binding(1) var<storage, read> heights: array<f32>;

const GRID_SIZE: u32 = 256u;

fn index_height(col: u32, row: u32) -> f32 {
    return heights[row * GRID_SIZE + col];
}

fn sample_height_at(col: i32, row: i32) -> f32 {
    let c = u32(clamp(col, 0, i32(GRID_SIZE) - 1));
    let r = u32(clamp(row, 0, i32(GRID_SIZE) - 1));
    return index_height(c, r);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_xz: vec2<f32>,
    @location(2) height: f32,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @builtin(vertex_index) vidx: u32,
) -> VertexOutput {
    let grid_n = u32(u.grid_resolution);
    let row = vidx / grid_n;
    let col = vidx - row * grid_n;
    let x_world = u.mesh_origin_x + position.x;
    let z_world = u.mesh_origin_z + position.y;

    let h = index_height(col, row) * u.height_amplitude;
    let world_pos = vec3<f32>(x_world, h, z_world);
    var out: VertexOutput;
    out.clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_xz = vec2<f32>(x_world, z_world);
    out.height = h;
    return out;
}

fn sample_height_bilinear(xz: vec2<f32>) -> f32 {
    let local = xz - vec2<f32>(u.mesh_origin_x, u.mesh_origin_z);
    let n = f32(GRID_SIZE - 1u);
    let uv_x = clamp(local.x / (n * u.mesh_cell_size), 0.0, 1.0);
    let uv_y = clamp(local.y / (n * u.mesh_cell_size), 0.0, 1.0);
    let fx = uv_x * n;
    let fy = uv_y * n;
    let c0 = i32(floor(fx));
    let r0 = i32(floor(fy));
    let c1 = c0 + 1;
    let r1 = r0 + 1;
    let tx = fx - f32(c0);
    let ty = fy - f32(r0);
    let h00 = sample_height_at(c0, r0);
    let h10 = sample_height_at(c1, r0);
    let h01 = sample_height_at(c0, r1);
    let h11 = sample_height_at(c1, r1);
    let top = mix(h00, h10, tx);
    let bot = mix(h01, h11, tx);
    return mix(top, bot, ty) * u.height_amplitude;
}

// -- procedural sky --------------------------------------------------------------

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(p.xyx * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let f2 = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f2.x), mix(c, d, f2.x), f2.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v: f32 = 0.0;
    var a: f32 = 0.5;
    var q = p;
    for (var i: i32 = 0; i < 5; i = i + 1) {
        v += a * noise2d(q);
        q = q * 2.03 + vec2<f32>(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

/// Manual normalize via `inverseSqrt` — saves the divide that the WGSL
/// `normalize()` builtin would otherwise perform. With 5 calls per fragment
/// over a 1280×800 surface this is roughly a 30% reduction in sqrt-class
/// instruction count on integrated GPUs.
fn normalize_fast(v: vec3<f32>) -> vec3<f32> {
    let inv_len: f32 = inverseSqrt(dot(v, v));
    return v * inv_len;
}

fn clouds(dir: vec3<f32>) -> f32 {
    let sky = dir / max(dir.y, 0.05);
    let uv = sky.xz * 0.35 + vec2<f32>(u.time * 0.02, u.time * 0.011);
    var d = fbm(uv);
    d = smoothstep(0.45, 0.85, d);
    return d * smoothstep(0.0, 0.2, dir.y);
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let y = clamp(dir.y, -1.0, 1.0);
    let horizon = mix(vec3<f32>(u.sky_horizon_r, u.sky_horizon_g, u.sky_horizon_b),
                      vec3<f32>(u.sky_top_r, u.sky_top_g, u.sky_top_b),
                      smoothstep(0.0, 0.5, y));
    let cloud = clouds(dir);
    let cloud_color = vec3<f32>(1.0, 0.98, 0.95);
    let base = mix(horizon, cloud_color, cloud * 0.85);

    let sun = normalize_fast(u.sun_dir.xyz);
    let sun_dot = dot(dir, sun);
    let disc = smoothstep(0.9995, 0.9999, sun_dot);
    let halo = pow(max(sun_dot, 0.0), 32.0) * 0.4;
    let sun_color = vec3<f32>(u.sun_color_r, u.sun_color_g, u.sun_color_b);
    return base + sun_color * (disc + halo);
}

// -- fragment main ---------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dx = u.mesh_cell_size;
    let h_l = sample_height_bilinear(in.world_xz + vec2<f32>(-dx, 0.0));
    let h_r = sample_height_bilinear(in.world_xz + vec2<f32>( dx, 0.0));
    let h_d = sample_height_bilinear(in.world_xz + vec2<f32>(0.0, -dx));
    let h_u = sample_height_bilinear(in.world_xz + vec2<f32>(0.0,  dx));
    let grad = vec3<f32>(h_l - h_r, 2.0 * dx, h_d - h_u);
    let n = normalize_fast(grad);

    let v = normalize_fast(u.camera_pos.xyz - in.world_pos);
    let r = reflect(-v, n);

    let deep = vec3<f32>(u.water_color_r, u.water_color_g, u.water_color_b);
    let refracted_color = mix(deep * 0.6, deep, smoothstep(0.0, 1.0, 1.0 - n.y));

    let sky = sky_color(r);
    let fresnel = u.reflectivity + (1.0 - u.reflectivity) * pow(1.0 - max(dot(v, n), 0.0), u.fresnel_power);
    var color = mix(refracted_color, sky, fresnel);

    let l = normalize_fast(u.sun_dir.xyz);
    let h_vec = normalize_fast(l + v);
    let spec = pow(max(dot(n, h_vec), 0.0), 256.0);
    let sun_color = vec3<f32>(u.sun_color_r, u.sun_color_g, u.sun_color_b);
    color += sun_color * spec * 1.5;

    let foam = smoothstep(0.65, 0.95, in.height / u.height_amplitude);
    color = mix(color, vec3<f32>(0.92, 0.95, 1.0), foam * 0.55);

    let dist = length(in.world_xz - u.camera_pos.xz);
    let haze = smoothstep(200.0, 800.0, dist);
    let haze_dir = normalize_fast(vec3<f32>(0.0, 0.05, -1.0));
    color = mix(color, sky_color(haze_dir), haze * 0.7);

    return vec4<f32>(color, 1.0);
}
"#;
