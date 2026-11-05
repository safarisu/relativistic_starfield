import numpy as np
import pandas as pd
import pyglet
from pyglet.window import key
import moderngl
from pyrr import Matrix44

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FOV = 60
SPEED_STEP = 0.002
MAX_SPEED = 0.99999
MOVE_SPEED = 0.02
NUMBER_OF_STARS = 9110
MIN_POINT_SIZE = 0.5
MAX_POINT_SIZE = 2.3
MAX_PITCH = 89.0
LINE_SUBDIVISIONS = 24
CONSTELLATION_LINE_COLOR = (0.18, 0.26, 0.42)


def kelvin_to_rgb(temp_k, brightness):
    """Approximate blackbody color for arrays of temperatures, scaled by brightness. Returns (N, 3) float32."""
    temp = np.asarray(temp_k, dtype=np.float64) / 100
    hot = temp > 66
    with np.errstate(invalid="ignore", divide="ignore"):
        red = np.where(hot, np.clip(329.70 * np.maximum(temp - 60, 1e-6) ** -0.1332, 0, 255), 255.0)
        green = np.where(hot,
                         np.clip(288.12 * np.maximum(temp - 60, 1e-6) ** -0.0755, 0, 255),
                         np.clip(99.47 * np.log(temp) - 161.12, 0, 255))
        blue = np.where(hot, 255.0,
                        np.where(temp <= 19, 0.0, np.clip(138.52 * np.log(temp - 10) - 305.04, 0, 255)))
    rgb = np.stack([red, green, blue], axis=1) / 255.0
    return (rgb * np.asarray(brightness)[:, None]).astype(np.float32)


def lorentz_gamma(beta):
    return 1.0 / np.sqrt(1.0 - beta ** 2)


def relativistic_transform(unit_dirs, v_hat, beta):
    """Aberrate rest-frame star directions for an observer moving along v_hat at speed beta (in units of c).

    Returns the observed unit directions (N, 3) and the per-star Doppler factor (N,).
    """
    gamma = lorentz_gamma(beta)
    dot = np.sum(unit_dirs * v_hat, axis=1, keepdims=True)

    factor = (1 - beta ** 2) / (1 + beta * dot)
    aberrated = factor * (unit_dirs + (gamma - 1) * dot * v_hat + gamma * beta * v_hat)
    observed = aberrated / np.linalg.norm(aberrated, axis=1, keepdims=True)

    doppler = gamma * (1 + beta * dot)
    return observed, doppler.flatten()


def great_circle_points(starts, ends, subdivisions):
    """Evenly spaced points along the great-circle arc from each start to end unit vector. Returns (N, subdivisions + 1, 3)."""
    starts, ends = starts.astype(np.float64), ends.astype(np.float64)
    omega = np.arccos(np.clip(np.sum(starts * ends, axis=1), -1.0, 1.0))[:, None, None]
    t = np.linspace(0.0, 1.0, subdivisions + 1)[None, :, None]
    return (np.sin((1 - t) * omega) * starts[:, None, :] + np.sin(t * omega) * ends[:, None, :]) / np.sin(omega)


def camera_direction(yaw_deg, pitch_deg):
    """Unit view vector for yaw (around +Z, from +X) and pitch (towards +Z), in degrees."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)


class StarfieldWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "Relativistic Starfield", resizable=True)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        # Shaders for stars
        self.program = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 mvp;
            in vec3 in_position;
            in vec3 in_color;
            in float in_brightness;
            out vec3 color;
            out float brightness;

            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);

                brightness = in_brightness;
                float size = mix(""" + f"{MIN_POINT_SIZE}" + """, """ + f"{MAX_POINT_SIZE}" + """, brightness);
                gl_PointSize = size;

                color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 color;
            in float brightness;
            out vec4 fragColor;

            void main() {
                // Calculate distance from center of point (0.5, 0.5 is center)
                float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
                
                // Discard pixels outside the circle (radius = 0.5)
                if (dist > 0.5) {
                    discard;
                }
                
                // Create smooth circular falloff
                float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

                // Core star color
                vec3 core_color = color * (0.8 + brightness * 0.7);
                
                // Glow color
                vec3 glow_color = color * 0.9;
                
                // Combine core and glow
                vec3 final_color = mix(glow_color, core_color, alpha);
                
                fragColor = vec4(final_color, alpha * (0.8 + brightness * 0.4));
            }
            """
        )

        # Shaders for velocity markers
        self.marker_program = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 mvp;
            in vec3 in_position;
            in vec3 in_color;
            out vec3 color;

            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                gl_PointSize = 10.0;
                color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 color;
            out vec4 fragColor;

            void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);

            // Solid center dot
            float center_dot = smoothstep(0.0, 0.1, 0.1 - dist);

            // Bold crosshair lines
            float cross = 0.0;
            if (abs(coord.x) < 0.1 || abs(coord.y) < 0.1) {
                cross = 1.0;
            }

            float alpha = max(center_dot, cross);
            if (alpha < 0.05) discard;

            fragColor = vec4(color, alpha);
            }
            """
        )

        # Shaders for constellation lines
        self.line_program = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 mvp;
            in vec3 in_position;

            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            out vec4 fragColor;

            void main() {
                fragColor = vec4(""" + ", ".join(f"{c}" for c in CONSTELLATION_LINE_COLOR) + """, 1.0);
            }
            """
        )

        self.yaw = 0.0
        self.pitch = 0.0
        self.camera_front = camera_direction(self.yaw, self.pitch)
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.velocity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.speed_fraction = 0.0
        self.speed_control = 0.0
        self.speed_exponent = 4.0
        self.velocity_locked = False
        
        # New toggles
        self.show_velocity_markers = False
        self.show_constellations = True

        self.set_exclusive_mouse(True)
        self.set_mouse_visible(False)
        self.left_mouse_held = False

        # Load star data (X, Y, Z, Temp_K, Mag)
        df = pd.read_csv("bright_stars.csv")
        df = df.sort_values("MAG").head(NUMBER_OF_STARS).reset_index(drop=True)

        positions = df[['x', 'y', 'z']].values.astype(np.float32)
        unit_vectors = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        self.star_positions = unit_vectors

        df['Temp_K'] = pd.to_numeric(df['Temp_K'], errors='coerce')
        self.star_temperatures = df['Temp_K'].values
        self.star_magnitudes = df['MAG'].values
        
        mag_min = np.min(self.star_magnitudes)
        
        # Convert magnitudes to linear brightness (brighter stars = higher value)
        linear_brightness = 2.512 ** (mag_min - self.star_magnitudes)
        
        # Apply perceptual adjustment to better distribute visible brightness (Weber-Fechner law)
        perceptual_factor = 0.1
        perceptual_brightness = np.power(linear_brightness, perceptual_factor)
        
        # Normalize and boost the overall brightness
        brightness_min = np.min(perceptual_brightness)
        brightness_max = np.max(perceptual_brightness)
        
        # Boost factor increases the minimum brightness significantly
        boost_factor = 0.1
        self.star_brightness = boost_factor + (1.0 - boost_factor) * (perceptual_brightness - brightness_min) / (brightness_max - brightness_min)
        
        # Apply additional global brightness multiplier
        global_brightness = 1.8  
        self.star_brightness = np.clip(self.star_brightness * global_brightness, 0.0, 1.0)
        
        self.star_brightness = self.star_brightness.astype(np.float32)

        self.star_colors = kelvin_to_rgb(self.star_temperatures, self.star_brightness)

        brightness_column = self.star_brightness.reshape(-1, 1)
        vertex_data = np.hstack((self.star_positions, self.star_colors, brightness_column)).astype(np.float32)

        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f 1f', 'in_position', 'in_color', 'in_brightness')
            ]
        )

        # Setup velocity marker VAO
        self.setup_velocity_markers()

        self.setup_constellation_lines(df)

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

        # Initialize labels in on_resize for correct positioning
        self.speed_label = pyglet.text.Label(
            text="Speed: 0.0000c", font_name="Consolas", font_size=14,
            x=10, y=self.height - 20, anchor_x="left", anchor_y="top", color=(255, 255, 255, 255)
        )
        self.gamma_label = pyglet.text.Label(
            text="γ: 1.0000", font_name="Consolas", font_size=14,
            x=10, y=self.height - 40, anchor_x="left", anchor_y="top", color=(255, 255, 255, 255)
        )
        self.fps_display = pyglet.window.FPSDisplay(self)

        self.ra_dec_label = pyglet.text.Label(
            text="RA: --h --m, DEC: --° --'",
            font_name="Consolas",
            font_size=14,
            x=self.width - 10,
            y=10,
            anchor_x="right",
            anchor_y="bottom",
            color=(255, 255, 255, 255)
        )

        # # Add control instructions label
        # self.controls_label = pyglet.text.Label(
        #     text="Controls: M - Toggle velocity markers | R - Reset | SPACE - Lock velocity",
        #     font_name="Consolas",
        #     font_size=12,
        #     x=10,
        #     y=10,
        #     anchor_x="left",
        #     anchor_y="bottom",
        #     color=(255, 255, 255, 200)
        # )

    def setup_velocity_markers(self):
        """Setup VAO for velocity direction markers"""
        # Create marker positions (forward and backward)
        marker_positions = np.array([self.velocity_dir, -self.velocity_dir], dtype=np.float32)
        # Colors: Green for forward direction, Red for backward
        marker_colors = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        
        marker_data = np.hstack([marker_positions, marker_colors]).astype(np.float32)
        
        self.marker_vbo = self.ctx.buffer(marker_data.tobytes())
        self.marker_vao = self.ctx.vertex_array(
            self.marker_program,
            [
                (self.marker_vbo, '3f 3f', 'in_position', 'in_color')
            ]
        )

    def setup_constellation_lines(self, stars_df):
        """Build constellation line geometry from constellation_lines.csv (pairs of HR numbers)."""
        lines = pd.read_csv("constellation_lines.csv")
        hr_to_index = pd.Series(stars_df.index, index=stars_df["HR"])
        # Skip lines whose stars aren't loaded (NUMBER_OF_STARS smaller than the catalog)
        lines = lines[lines["HR1"].isin(hr_to_index.index) & lines["HR2"].isin(hr_to_index.index)]

        starts = self.star_positions[hr_to_index[lines["HR1"]].values]
        ends = self.star_positions[hr_to_index[lines["HR2"]].values]
        # Each line is subdivided along its great-circle arc, because aberration turns those arcs into curves
        points = great_circle_points(starts, ends, LINE_SUBDIVISIONS)
        self.line_positions = points.reshape(-1, 3).astype(np.float32)

        # Index pairs (i, i + 1) along each subdivided line, rendered as GL_LINES
        offsets = np.arange(len(lines))[:, None] * (LINE_SUBDIVISIONS + 1)
        steps = np.arange(LINE_SUBDIVISIONS)
        indices = np.stack([offsets + steps, offsets + steps + 1], axis=-1).astype('i4')

        self.line_vbo = self.ctx.buffer(self.line_positions.tobytes())
        self.line_ibo = self.ctx.buffer(indices.tobytes())
        self.line_vao = self.ctx.vertex_array(
            self.line_program,
            [
                (self.line_vbo, '3f', 'in_position')
            ],
            index_buffer=self.line_ibo,
            index_element_size=4
        )

    def update_velocity_markers(self):
        """Update velocity marker positions"""
        if self.velocity_locked:
            marker_positions = np.array([self.velocity_dir, -self.velocity_dir], dtype=np.float32)
            marker_colors = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
            marker_data = np.hstack([marker_positions, marker_colors]).astype(np.float32)
            self.marker_vbo.write(marker_data.tobytes())

    def reset_simulation(self):
        """Reset velocity and speed to zero"""
        self.speed_fraction = 0.0
        self.speed_control = 0.0
        self.velocity_locked = False
        self.velocity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.update_velocity_markers()

    def vector_to_ra_dec(self, vec):
        x, y, z = vec
        ra_rad = np.arctan2(y, x)
        if ra_rad < 0:
            ra_rad += 2 * np.pi
        ra_deg = np.degrees(ra_rad)
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_m = int((ra_hours - ra_h) * 60)
        ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

        dec_rad = np.arcsin(z)
        dec_deg = np.degrees(dec_rad)
        dec_d = int(dec_deg)
        dec_m = int(abs(dec_deg - dec_d) * 60)
        dec_s = (abs(dec_deg - dec_d) * 60 - dec_m) * 60

        ra_str = f"{ra_h:02d}h{ra_m:02d}m{ra_s:04.1f}s"
        dec_sign = '+' if dec_d >= 0 else '-'
        dec_str = f"{dec_sign}{abs(dec_d):02d}°{dec_m:02d}'{dec_s:04.1f}\""
        return f"RA: {ra_str}  DEC: {dec_str}"

    def update(self, dt):
        # WASD rotates the camera: A/D yaw left/right, W/S pitch up/down
        step = np.degrees(MOVE_SPEED)
        d_yaw = step * (self.keys[key.A] - self.keys[key.D])
        d_pitch = step * (self.keys[key.W] - self.keys[key.S])
        if d_yaw or d_pitch:
            self.rotate_camera(d_yaw, d_pitch)

        # Speed control with Up/Down arrows
        if self.keys[key.UP]:
            self.speed_control = min(self.speed_control + SPEED_STEP, 1.0)
        if self.keys[key.DOWN]:
            self.speed_control = max(self.speed_control - SPEED_STEP, 0.0)
            if self.speed_control == 0.0:
                self.velocity_locked = False
        self.speed_fraction = MAX_SPEED * (1 - pow(1 - self.speed_control, self.speed_exponent))

    def on_draw(self):
        self.clear()
        self.ctx.clear(0.0, 0.0, 0.0)

        v_hat = self.velocity_dir / np.linalg.norm(self.velocity_dir)
        if self.speed_fraction > 0:
            positions, doppler = relativistic_transform(self.star_positions, v_hat, self.speed_fraction)

            # Doppler shift and brightness boost
            shifted_temperatures = self.star_temperatures * doppler
            final_brightness = self.star_brightness * doppler ** 0.4
            shifted_colors = kelvin_to_rgb(shifted_temperatures, final_brightness)
        else:
            positions = self.star_positions
            shifted_colors = self.star_colors
            final_brightness = self.star_brightness

        # VBO layout: pos (3f), color (3f), brightness (1f).
        # Sorted so brighter stars draw last: pyglet's labels leave GL blending disabled, so overlapping points overwrite.
        interleaved_data = np.hstack([positions, shifted_colors, final_brightness.reshape(-1, 1)]).astype(np.float32)
        interleaved_data = interleaved_data[np.argsort(interleaved_data[:, -1])]
        self.vbo.write(interleaved_data.tobytes())

        # View-projection matrix
        view = Matrix44.look_at(np.zeros(3), self.camera_front, self.camera_up)
        proj = Matrix44.perspective_projection(FOV, self.width / self.height, 0.1, 2000.0)
        mvp = proj * view

        # Draw constellation lines beneath the stars. They are not light, so only aberration applies, not Doppler.
        if self.show_constellations:
            line_positions = self.line_positions
            if self.speed_fraction > 0:
                line_positions, _ = relativistic_transform(line_positions, v_hat, self.speed_fraction)
            self.line_vbo.write(line_positions.astype(np.float32).tobytes())
            self.line_program['mvp'].write(mvp.astype('f4').tobytes())
            # Opaque, so lines crowded together at high speed don't add up to a bright blob
            self.ctx.disable(moderngl.BLEND)
            self.line_vao.render(mode=moderngl.LINES)

        # Draw stars
        self.program['mvp'].write(mvp.astype('f4').tobytes())
        self.vao.render(mode=moderngl.POINTS)

        # Draw velocity markers if enabled and velocity is locked
        if self.show_velocity_markers and self.velocity_locked:
            self.marker_program['mvp'].write(mvp.astype('f4').tobytes())
            self.marker_vao.render(mode=moderngl.POINTS)

        # HUD: Speed, gamma, RA/Dec, FPS, Controls
        gamma_val = lorentz_gamma(self.speed_fraction)
        self.speed_label.text = f"Speed: {self.speed_fraction:.4f}c"
        self.gamma_label.text = f"γ: {gamma_val:.4f}"
        self.speed_label.draw()
        self.gamma_label.draw()

        ra_dec_text = self.vector_to_ra_dec(self.camera_front)
        self.ra_dec_label.text = ra_dec_text
        self.ra_dec_label.draw()

        # self.controls_label.draw()
        self.fps_display.draw()

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.left_mouse_held = True

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.left_mouse_held = False

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.left_mouse_held:
            return

        sensitivity = 0.1
        self.rotate_camera(dx * sensitivity, -dy * sensitivity)

    def rotate_camera(self, d_yaw, d_pitch):
        """Shared by mouse and keyboard: adjust yaw/pitch in degrees and recompute the view direction."""
        self.yaw = (self.yaw + d_yaw) % 360.0
        self.pitch = float(np.clip(self.pitch + d_pitch, -MAX_PITCH, MAX_PITCH))
        self.camera_front = camera_direction(self.yaw, self.pitch)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.velocity_dir = self.camera_front.copy()
            self.velocity_locked = True
            self.update_velocity_markers()
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.F:
            self.set_fullscreen(not self.fullscreen)
        elif symbol == key.M:
            # Toggle velocity markers
            self.show_velocity_markers = not self.show_velocity_markers
        elif symbol == key.C:
            self.show_constellations = not self.show_constellations
        elif symbol == key.R:
            # Reset simulation
            self.reset_simulation()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.ctx.viewport = (0, 0, width, height)
        # Update label positions on resize
        self.speed_label.y = height - 20
        self.gamma_label.y = height - 40
        self.ra_dec_label.x = width - 10


if __name__ == "__main__":
    window = StarfieldWindow()
    pyglet.app.run()