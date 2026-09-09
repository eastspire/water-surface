use super::*;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use euv::wasm_bindgen::prelude::*;
use euv::wasm_bindgen::{JsCast, JsValue};
use euv::wasm_bindgen_futures::spawn_local;
use euv::web_sys::*;
use euv::*;

use euv_engine::*;
use js_sys::Math;

use crate::shader::SURFACE_SHADER;

/// WebGPU buffer-usage bitmask values (from the W3C WebGPU spec).
///
/// `euv-engine` keeps these as `pub(crate)` constants — water-surface needs
/// the raw values to call `create_buffer(size, usage)`, so we redeclare the
/// three values we actually use. See
/// <https://www.w3.org/TR/webgpu/#buffer-usage> for the full bitmask.
mod usage {
    /// Buffer bound as `var<storage, read>` / `var<storage, read_write>`.
    pub const STORAGE: u32 = 0x80;
    /// Pipeline-uniform binding (`var<uniform>`).
    pub const UNIFORM: u32 = 0x40;
    /// `COPY_DST (0x08)` — required on any buffer the CPU writes to via
    /// `queue.writeBuffer`.
    pub const COPY_DST: u32 = 0x08;
}

/// Creates the page-level reactive state.
pub(crate) fn use_water_state() -> UseWater {
    UseWater {
        fps: euv::App::use_signal(|| 0.0_f32),
        ready: euv::App::use_signal(|| false),
        error_message: euv::App::use_signal(String::new),
    }
}

/// Reads the canvas element's CSS layout size.
fn read_canvas_size(selector: &str) -> Option<(f32, f32)> {
    let window_value = window()?;
    let document_value = window_value.document()?;
    let element = document_value.query_selector(selector).ok().flatten()?;
    let canvas: web_sys::HtmlCanvasElement = element.unchecked_into();
    let rect: web_sys::DomRect = canvas.get_bounding_client_rect();
    Some((rect.width() as f32, rect.height() as f32))
}

fn normalize3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len: f32 = (x * x + y * y + z * z).sqrt();
    if len < 1e-6 {
        (0.0, 0.0, -1.0)
    } else {
        (x / len, y / len, z / len)
    }
}

fn look_at(eye: (f32, f32, f32), target: (f32, f32, f32), up: (f32, f32, f32)) -> [f32; 16] {
    let f: (f32, f32, f32) = normalize3(target.0 - eye.0, target.1 - eye.1, target.2 - eye.2);
    let s: (f32, f32, f32) = normalize3(
        f.1 * up.2 - f.2 * up.1,
        f.2 * up.0 - f.0 * up.2,
        f.0 * up.1 - f.1 * up.0,
    );
    let u: (f32, f32, f32) = (
        s.1 * f.2 - s.2 * f.1,
        s.2 * f.0 - s.0 * f.2,
        s.0 * f.1 - s.1 * f.0,
    );
    let mut m: [f32; 16] = [0.0_f32; 16];
    m[0] = s.0;
    m[1] = u.0;
    m[2] = -f.0;
    m[3] = 0.0;
    m[4] = s.1;
    m[5] = u.1;
    m[6] = -f.1;
    m[7] = 0.0;
    m[8] = s.2;
    m[9] = u.2;
    m[10] = -f.2;
    m[11] = 0.0;
    m[12] = -(s.0 * eye.0 + s.1 * eye.1 + s.2 * eye.2);
    m[13] = -(u.0 * eye.0 + u.1 * eye.1 + u.2 * eye.2);
    m[14] = f.0 * eye.0 + f.1 * eye.1 + f.2 * eye.2;
    m[15] = 1.0;
    m
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f: f32 = 1.0 / (fov_y * 0.5).tan();
    let nf: f32 = 1.0 / (near - far);
    let mut m: [f32; 16] = [0.0_f32; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1.0;
    m[14] = 2.0 * far * near * nf;
    m
}

fn multiply_mat4(a: &[f32; 16], b: &[f32; 16]) -> Vec<f32> {
    let mut out: Vec<f32> = vec![0.0_f32; 16];
    for col in 0..4u32 {
        for row in 0..4u32 {
            let mut sum: f32 = 0.0_f32;
            for k in 0..4u32 {
                sum += a[k as usize * 4 + row as usize] * b[col as usize * 4 + k as usize];
            }
            out[col as usize * 4 + row as usize] = sum;
        }
    }
    out
}

/// Builds the view-projection matrix.
fn build_view_proj(camera_pos: (f32, f32, f32), w: f32, h: f32) -> Vec<f32> {
    let view: [f32; 16] = look_at(camera_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    let aspect: f32 = w / h.max(1.0);
    let proj: [f32; 16] = perspective(std::f32::consts::FRAC_PI_4, aspect, 0.5, 1000.0);
    multiply_mat4(&proj, &view)
}

/// Returns the camera's forward unit vector (camera → origin).
fn camera_forward(orbit: &CameraOrbit) -> (f32, f32, f32) {
    let yaw: f32 = orbit.yaw.get();
    let pitch: f32 = orbit.pitch.get();
    let cy: f32 = pitch.cos();
    normalize3(-yaw.sin() * cy, -pitch.sin(), -yaw.cos() * cy)
}

/// Per-frame uniform buffer for the surface render pipeline.
fn build_surface_uniforms(
    view_proj: &[f32],
    camera_pos: (f32, f32, f32),
    forward: (f32, f32, f32),
    time: f32,
    mesh_origin: (f32, f32),
    cell_size: f32,
) -> Vec<f32> {
    let mut u: Vec<f32> = vec![0.0_f32; SURFACE_UNIFORM_F32_COUNT];
    u[..16].copy_from_slice(view_proj);
    u[16] = camera_pos.0;
    u[17] = camera_pos.1;
    u[18] = camera_pos.2;
    u[19] = 1.0;
    u[20] = SUN_DIR_X;
    u[21] = SUN_DIR_Y;
    u[22] = SUN_DIR_Z;
    u[23] = 1.0;
    u[24] = forward.0;
    u[25] = forward.1;
    u[26] = forward.2;
    u[27] = 1.0;
    u[28] = time;
    u[29] = mesh_origin.0;
    u[30] = mesh_origin.1;
    u[31] = cell_size;
    u[32] = GRID_RESOLUTION as f32;
    u[33] = HEIGHT_AMPLITUDE_M;
    u[34] = FRESNEL_POWER;
    u[35] = FRESNEL_REFLECTIVITY;
    u[36] = WATER_COLOR.0;
    u[37] = WATER_COLOR.1;
    u[38] = WATER_COLOR.2;
    u[39] = 0.0;
    u[40] = SUN_COLOR.0;
    u[41] = SUN_COLOR.1;
    u[42] = SUN_COLOR.2;
    u[43] = 0.0;
    u[44] = SKY_TOP_COLOR.0;
    u[45] = SKY_TOP_COLOR.1;
    u[46] = SKY_TOP_COLOR.2;
    u[47] = 0.0;
    u[48] = SKY_HORIZON_COLOR.0;
    u[49] = SKY_HORIZON_COLOR.1;
    u[50] = SKY_HORIZON_COLOR.2;
    u[51] = 0.0;
    u
}

/// Vertex buffer for the GRID_RESOLUTION × GRID_RESOLUTION grid. Each vertex
/// is a 2D position in mesh-local space ranging over [0, MESH_TILE_SIZE_M]^2.
fn build_grid_vertex_buffer() -> Vec<u8> {
    let n: usize = GRID_RESOLUTION as usize;
    let mesh_size: f32 = MESH_TILE_SIZE_M;
    let mut data: Vec<u8> = Vec::with_capacity(n * n * 2 * 4);
    for row in 0..n {
        for col in 0..n {
            let x: f32 = (col as f32 / (n - 1) as f32) * mesh_size;
            let z: f32 = (row as f32 / (n - 1) as f32) * mesh_size;
            data.extend_from_slice(&x.to_le_bytes());
            data.extend_from_slice(&z.to_le_bytes());
        }
    }
    data
}

/// Index buffer for the grid — 2 triangles per quad.
fn build_grid_index_buffer() -> Vec<u8> {
    let n: usize = GRID_RESOLUTION as usize;
    let mut data: Vec<u8> = Vec::with_capacity((n - 1) * (n - 1) * 6 * 4);
    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl: u32 = (row * n + col) as u32;
            let tr: u32 = tl + 1;
            let bl: u32 = tl + n as u32;
            let br: u32 = bl + 1;
            data.extend_from_slice(&tl.to_le_bytes());
            data.extend_from_slice(&bl.to_le_bytes());
            data.extend_from_slice(&tr.to_le_bytes());
            data.extend_from_slice(&tr.to_le_bytes());
            data.extend_from_slice(&bl.to_le_bytes());
            data.extend_from_slice(&br.to_le_bytes());
        }
    }
    data
}

// ─── Wave simulation (CPU side) ───────────────────────────────────────────────

/// A CPU-side wave-equation simulation. Two height buffers (prev, curr) per
/// cell; each step integrates `h_new = 2h - prev + c²·∇²h` then applies
/// damping.
struct WaveSimulation {
    prev: Vec<f32>,
    curr: Vec<f32>,
}

impl WaveSimulation {
    fn new() -> Self {
        let n: usize = (GRID_RESOLUTION * GRID_RESOLUTION) as usize;
        Self {
            prev: vec![0.0_f32; n],
            curr: vec![0.0_f32; n],
        }
    }

    fn inject_gaussian(&mut self, cx: i32, cy: i32, amplitude: f32, radius: i32) {
        let n: i32 = GRID_RESOLUTION as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x: i32 = cx + dx;
                let y: i32 = cy + dy;
                if x < 0 || y < 0 || x >= n || y >= n {
                    continue;
                }
                let r2: f32 = (dx * dx + dy * dy) as f32;
                let sigma2: f32 = (radius as f32 * 0.5).powi(2);
                let bump: f32 = amplitude * (-r2 / (2.0 * sigma2)).exp();
                self.curr[(y * n + x) as usize] += bump;
            }
        }
    }

    fn step(&mut self, c_squared: f32, damping: f32) {
        let n: usize = GRID_RESOLUTION as usize;
        let mut next: Vec<f32> = vec![0.0_f32; n * n];
        for y in 1..(n - 1) {
            for x in 1..(n - 1) {
                let idx: usize = y * n + x;
                let center: f32 = self.curr[idx];
                let hm: f32 = self.curr[idx - 1];
                let hp: f32 = self.curr[idx + 1];
                let hd: f32 = self.curr[idx - n];
                let hu: f32 = self.curr[idx + n];
                let prev: f32 = self.prev[idx];
                let laplacian: f32 = hm + hp + hd + hu - 4.0 * center;
                let accel: f32 = c_squared * laplacian;
                next[idx] = (2.0 * center - prev + accel) * (1.0 - damping);
            }
        }
        // Boundaries decay toward zero (soft reflective).
        for i in 0..n {
            next[i] = 0.0;
            next[(n - 1) * n + i] = 0.0;
            next[i * n] = 0.0;
            next[i * n + (n - 1)] = 0.0;
        }
        std::mem::swap(&mut self.prev, &mut self.curr);
        self.curr.copy_from_slice(&next);
    }
}

/// Live GPU + CPU resources, owned by the RAF closure.
struct WaterRenderer {
    renderer: WebGpuRenderer,
    render_pipeline: JsValue,
    height_buffer: JsValue,
    surface_uniform: JsValue,
    surface_bind_group: JsValue,
    vertex_buffer: JsValue,
    index_buffer: JsValue,
    /// Number of indices in the index buffer (one entry per triangle corner).
    index_count: u32,
    mesh_origin: Rc<RefCell<(f32, f32)>>,
    sim: Rc<RefCell<WaveSimulation>>,
    orbit: CameraOrbit,
    cancelled: Rc<Cell<bool>>,
    raf_id: Rc<Cell<Option<i32>>>,
    state: UseWater,
}

impl WaterRenderer {
    fn cell_size() -> f32 {
        MESH_TILE_SIZE_M / GRID_RESOLUTION as f32
    }

    fn snap_mesh_to_camera(&self) {
        let cell = Self::cell_size();
        let cam = self.orbit.camera_position();
        let cx = (cam.0 / cell).round() * cell - MESH_TILE_SIZE_M * 0.5;
        let cz = (cam.2 / cell).round() * cell - MESH_TILE_SIZE_M * 0.5;
        *self.mesh_origin.borrow_mut() = (cx, cz);
    }
}

/// Main entrypoint. Spawns the WebGPU init + RAF loop. Called from the view
/// once on first mount.
pub(crate) fn start_water_loop(state: UseWater, orbit: CameraOrbit) {
    spawn_local(async move {
        match init_water_renderer(state.clone(), orbit.clone()).await {
            Ok(renderer) => {
                let renderer = Rc::new(RefCell::new(renderer));
                run_water_loop(renderer).await;
            }
            Err(err) => {
                let msg = format!("WebGPU init failed: {err:?}");
                web_sys::console::error_1(&JsValue::from_str(&msg));
                state.get_error_message().set(msg);
                state.get_ready().set(true);
            }
        }
    });
}

async fn init_water_renderer(
    state: UseWater,
    orbit: CameraOrbit,
) -> Result<WaterRenderer, JsValue> {
    let config = RenderConfig::webgpu(WATER_CANVAS_SELECTOR, 1280.0, 800.0);
    let renderer = Engine::webgpu_renderer(&config)
        .await
        .map_err(|e| format!("webgpu_renderer init: {e:?}"))?;

    // Surface (render) pipeline — single vertex buffer slot 0 holding
    // `[f32; 2]` positions, no other vertex attributes.
    let vertex_layout = VertexBufferLayout::new(
        8u64,
        VertexStepMode::Vertex,
        vec![VertexAttribute::new(0u32, 0u64, "float32x2")],
    );
    let render_pipeline = renderer.create_render_pipeline_full(
        SURFACE_SHADER,
        &[vertex_layout],
        "vs_main",
        "fs_main",
        None,
    );
    if render_pipeline.is_undefined() || render_pipeline.is_null() {
        return Err(JsValue::from_str(
            "create_render_pipeline_full returned undefined",
        ));
    }

    // Storage buffer for `var<storage, read> array<f32>` heights — created
    // with `STORAGE | COPY_DST` so we can upload from CPU each frame.
    let grid_count = (GRID_RESOLUTION * GRID_RESOLUTION) as usize;
    let height_buffer_size = (grid_count * std::mem::size_of::<f32>()) as u64;
    let height_buffer =
        renderer.create_buffer(height_buffer_size, usage::STORAGE | usage::COPY_DST);
    if height_buffer.is_undefined() || height_buffer.is_null() {
        return Err(JsValue::from_str("create_buffer (heights) failed"));
    }

    // Surface-uniform buffer: mat4x4 view-proj + camera / sun / sky params.
    let surface_uniform_size: u64 = (SURFACE_UNIFORM_F32_COUNT * std::mem::size_of::<f32>()) as u64;
    let surface_uniform =
        renderer.create_buffer(surface_uniform_size, usage::UNIFORM | usage::COPY_DST);
    if surface_uniform.is_undefined() || surface_uniform.is_null() {
        return Err(JsValue::from_str("create_buffer (uniforms) failed"));
    }

    // The bind group layout must list a Buffer entry for `@binding(0)` and
    // `@binding(1)`. With auto-layout inferred from the pipeline, the GPU
    // decides which slots are uniform vs storage — we just provide both
    // buffers and the validation layer accepts it because the shader declares
    // `var<uniform>` / `var<storage>` matching the buffer usage flags.
    //
    // `create_bind_group` calls `device.createBindGroup(...)` and captures
    // any GPU validation error via `pop_error_sync`, logging it to the
    // JS console. A mismatch between the buffer's `usage` flags and the
    // shader's `@binding` declaration is the most common cause of a
    // silent "no draw" (the draw call still runs, but the GPU rejects
    // the bind group and emits zero fragments).
    let entries = vec![
        BindGroupEntry::Buffer {
            binding: 0,
            buffer: surface_uniform.clone(),
            offset: 0,
            size: None,
        },
        BindGroupEntry::Buffer {
            binding: 1,
            buffer: height_buffer.clone(),
            offset: 0,
            size: None,
        },
    ];
    let surface_bind_group = renderer.create_bind_group(&render_pipeline, 0, &entries);
    if surface_bind_group.is_undefined() || surface_bind_group.is_null() {
        return Err(JsValue::from_str("create_bind_group (surface) failed"));
    }

    // Static vertex / index buffers for the GRID_RESOLUTION × GRID_RESOLUTION
    // mesh tile. The tile follows the camera in world space; the mesh-local
    // coordinates span [0, MESH_TILE_SIZE_M]^2.
    let vertex_buffer = renderer.create_vertex_buffer(&build_grid_vertex_buffer());
    if vertex_buffer.is_undefined() || vertex_buffer.is_null() {
        return Err(JsValue::from_str("create_vertex_buffer failed"));
    }
    let index_buffer = renderer.create_index_buffer(&build_grid_index_buffer());
    if index_buffer.is_undefined() || index_buffer.is_null() {
        return Err(JsValue::from_str("create_index_buffer failed"));
    }
    let index_count: u32 = (GRID_RESOLUTION - 1) * (GRID_RESOLUTION - 1) * 6;

    state.get_ready().set(true);

    Ok(WaterRenderer {
        renderer,
        render_pipeline,
        height_buffer,
        surface_uniform,
        surface_bind_group,
        vertex_buffer,
        index_buffer,
        index_count,
        mesh_origin: Rc::new(RefCell::new((0.0, 0.0))),
        sim: Rc::new(RefCell::new(WaveSimulation::new())),
        orbit,
        cancelled: Rc::new(Cell::new(false)),
        raf_id: Rc::new(Cell::new(None)),
        state,
    })
}

async fn run_water_loop(renderer: Rc<RefCell<WaterRenderer>>) {
    let start_time = current_time_ms();
    let mut last_frame = start_time;
    let mut frame_count: u32 = 0;
    let mut ambient_accum: f32 = 0.0;

    // Seed a few ripples for visual interest on first load.
    {
        let sim_ref = renderer.borrow();
        let mut sim = sim_ref.sim.borrow_mut();
        sim.inject_gaussian(
            GRID_RESOLUTION as i32 / 2,
            GRID_RESOLUTION as i32 / 2,
            0.6,
            8,
        );
        sim.inject_gaussian(
            GRID_RESOLUTION as i32 / 2 + 16,
            GRID_RESOLUTION as i32 / 2 - 12,
            0.4,
            6,
        );
        sim.inject_gaussian(
            GRID_RESOLUTION as i32 / 2 - 20,
            GRID_RESOLUTION as i32 / 2 + 18,
            0.5,
            7,
        );
    }

    let renderer_for_closure = renderer.clone();
    let cancelled = renderer.borrow().cancelled.clone();
    let raf_id_cell = renderer.borrow().raf_id.clone();
    let raf_id_for_inner = raf_id_cell.clone();

    // Build the RAF closure and stash a clone of its `js_sys::Function`
    // handle so we can re-schedule itself each frame after it's moved
    // into the FnMut body.
    let raf_fn_cell: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));
    let raf_fn_cell_inner = raf_fn_cell.clone();
    let closure: Closure<dyn FnMut(f64)> = Closure::wrap(Box::new(move |_t: f64| {
        if cancelled.get() {
            return;
        }
        let now = current_time_ms();
        let dt = ((now - last_frame) as f32 / 1000.0).min(0.05);
        last_frame = now;
        let elapsed = (now - start_time) as f32 / 1000.0;
        frame_count += 1;
        // Drain any GPU validation error that surfaced since the last
        // frame. `pop_error_sync` is asynchronous (it writes the result
        // to `pending_error` via a microtask), so the value is only
        // visible on the next render tick.
        if let Some(err) = renderer_for_closure.borrow().renderer.take_last_error()
            && (frame_count <= 5 || frame_count.is_multiple_of(60))
        {
            let s = format!("{err:?}");
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "[water] gpu error (frame {frame_count}): {s}"
            )));
        }
        if frame_count.is_multiple_of(30) {
            let fps = 1.0 / dt.max(1e-3);
            renderer_for_closure.borrow().state.get_fps().set(fps);
        }

        // Step 1: ambient ripples.
        ambient_accum += dt;
        let ambient_interval = 1.0 / AMBIENT_RIPPLE_RATE_HZ;
        while ambient_accum >= ambient_interval {
            ambient_accum -= ambient_interval;
            let r = renderer_for_closure.borrow();
            let mut sim = r.sim.borrow_mut();
            let cx = (Math::random() as f32 * GRID_RESOLUTION as f32) as i32;
            let cy = (Math::random() as f32 * GRID_RESOLUTION as f32) as i32;
            sim.inject_gaussian(cx, cy, 0.25, 5);
        }

        // Step 2: integrate wave equation.
        {
            let r = renderer_for_closure.borrow();
            let mut sim = r.sim.borrow_mut();
            for _ in 0..2 {
                sim.step(WAVE_C_SQUARED, WAVE_DAMPING);
            }
        }

        // Step 3: upload the simulation buffer to the GPU height-storage
        // buffer via euv-engine's `write_buffer`.
        let curr_bytes: Vec<u8> = {
            let r = renderer_for_closure.borrow();
            let curr: &Vec<f32> = &r.sim.borrow().curr;
            let mut bytes: Vec<u8> = Vec::with_capacity(curr.len() * 4);
            for v in curr.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        };
        {
            let r = renderer_for_closure.borrow_mut();
            r.renderer.write_buffer(&r.height_buffer, 0, &curr_bytes);
        }

        // Step 4: snap mesh origin to the camera.
        renderer_for_closure.borrow().snap_mesh_to_camera();

        // Step 5: rebuild surface uniforms.
        let (cam, forward, mesh_origin) = {
            let r = renderer_for_closure.borrow();
            (
                r.orbit.camera_position(),
                camera_forward(&r.orbit),
                *r.mesh_origin.borrow(),
            )
        };
        let canvas_size: (f32, f32) =
            read_canvas_size(WATER_CANVAS_SELECTOR).unwrap_or((1280.0, 720.0));
        let view_proj: Vec<f32> = build_view_proj(cam, canvas_size.0, canvas_size.1);
        let uniforms: Vec<f32> = build_surface_uniforms(
            &view_proj,
            cam,
            forward,
            elapsed,
            mesh_origin,
            WaterRenderer::cell_size(),
        );
        {
            let uniforms_bytes: Vec<u8> = uniforms.iter().flat_map(|f| f.to_le_bytes()).collect();
            let r = renderer_for_closure.borrow_mut();
            r.renderer
                .write_buffer(&r.surface_uniform, 0, &uniforms_bytes);
        }

        // Step 6: drive the render pass manually with the new low-level API.
        // `begin_render_pass` requires `&mut self`, so we hold a single
        // `borrow_mut` scope for the whole pass + submit sequence.
        let command_buffer = {
            let mut r = renderer_for_closure.borrow_mut();
            let encoder = r.renderer.create_command_encoder();
            let pass = r
                .renderer
                .begin_render_pass(&encoder, (0.55, 0.72, 0.86, 1.0));
            if frame_count == 1 {
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "[water] first render: pipeline={:?}, bind_group={:?}, vb={:?}, ib={:?}, ic={}",
                    if r.render_pipeline.is_undefined() {
                        "undef"
                    } else {
                        "ok"
                    },
                    if r.surface_bind_group.is_undefined() {
                        "undef"
                    } else {
                        "ok"
                    },
                    if r.vertex_buffer.is_undefined() {
                        "undef"
                    } else {
                        "ok"
                    },
                    if r.index_buffer.is_undefined() {
                        "undef"
                    } else {
                        "ok"
                    },
                    r.index_count,
                )));
            }
            r.renderer.set_pipeline(&pass, &r.render_pipeline);
            r.renderer.set_bind_group(&pass, 0, &r.surface_bind_group);
            r.renderer.set_vertex_buffer(&pass, 0, &r.vertex_buffer);
            r.renderer
                .set_index_buffer(&pass, &r.index_buffer, "uint32");
            r.renderer.draw_indexed(&pass, r.index_count, 1);
            r.renderer.end_render_pass(&pass);
            r.renderer.finish_command_encoder(&encoder)
        };
        renderer_for_closure
            .borrow_mut()
            .renderer
            .submit(&[command_buffer]);

        // Step 7: schedule next frame via the stashed Function handle.
        if let Some(win) = window()
            && let Some(raf_fn) = raf_fn_cell_inner.borrow().as_ref()
        {
            let raf_value = win.request_animation_frame(raf_fn);
            if let Ok(id) = raf_value {
                raf_id_for_inner.set(Some(id));
            }
        }
    }));

    let raf_fn: js_sys::Function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    *raf_fn_cell.borrow_mut() = Some(raf_fn);

    if let Some(win) = window()
        && let Some(raf_fn) = raf_fn_cell.borrow().as_ref()
    {
        let raf_value = win.request_animation_frame(raf_fn);
        if let Ok(id) = raf_value {
            raf_id_cell.set(Some(id));
        }
    }

    closure.forget();
}

fn current_time_ms() -> f64 {
    js_sys::Date::now()
}
