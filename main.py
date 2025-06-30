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
STAR_DISTANCE = 1
MOVE_SPEED = 0.02
NUMBER_OF_STARS = 3500
MIN_POINT_SIZE = 0.5
MAX_POINT_SIZE = 2.3


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
                
                // Create glow effect
                float glow = .0 - smoothstep(0.0, 0.4, dist);
                float enhanced_glow = glow * (0.6 + brightness * 0.8);
                
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

        self.camera_front = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.camera_right = np.cross(self.camera_front, self.camera_up)

        self.velocity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.speed_fraction = 0.0
        self.speed_control = 0.0
        self.speed_exponent = 4.0
        self.velocity_locked = False
        
        # New toggles
        self.show_velocity_markers = False

        self.yaw = -90.0
        self.pitch = 0.0
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
        
        # Get magnitude range
        mag_min = np.min(self.star_magnitudes)
        mag_max = np.max(self.star_magnitudes)
        
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
        
        # Store original brightness values for reference
        self.original_max_brightness = np.max(self.star_brightness)
        self.base_star_brightness = self.star_brightness.copy()

        def kelvin_to_rgb(temp_k, brightness=1.0):
            temp = temp_k / 100
            if temp <= 66:
                red = 255
                green = np.clip(99.47 * np.log(temp) - 161.12, 0, 255)
                blue = 0 if temp <= 19 else np.clip(138.52 * np.log(temp - 10) - 305.04, 0, 255)
            else:
                red = np.clip(329.70 * ((temp - 60) ** -0.1332), 0, 255)
                green = np.clip(288.12 * ((temp - 60) ** -0.0755), 0, 255)
                blue = 255
                
            return np.array([red, green, blue]) / 255.0 * brightness

        self.kelvin_to_rgb = kelvin_to_rgb
        self.star_colors = np.array([
            kelvin_to_rgb(t, b) for t, b in zip(self.star_temperatures, self.star_brightness)
        ], dtype=np.float32)

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
        # WASD controls for rotating camera_front
        # Calculate right vector to move laterally
        self.camera_right = np.cross(self.camera_front, self.camera_up)
        self.camera_right /= np.linalg.norm(self.camera_right)

        rotated = False

        # Yaw rotation (around camera_up)
        if self.keys[key.A]:
            # Rotate left
            angle = MOVE_SPEED
            c, s = np.cos(angle), np.sin(angle)
            front = self.camera_front
            right = self.camera_right
            self.camera_front = front * c - right * s
            rotated = True
        if self.keys[key.D]:
            # Rotate right
            angle = -MOVE_SPEED
            c, s = np.cos(angle), np.sin(angle)
            front = self.camera_front
            right = self.camera_right
            self.camera_front = front * c - right * s
            rotated = True

        # Pitch rotation (around camera_right)
        if self.keys[key.W]:
            angle = MOVE_SPEED
            c, s = np.cos(angle), np.sin(angle)
            front = self.camera_front
            up = self.camera_up
            right = self.camera_right
            self.camera_front = front * c + up * s
            rotated = True
        if self.keys[key.S]:
            angle = -MOVE_SPEED
            c, s = np.cos(angle), np.sin(angle)
            front = self.camera_front
            up = self.camera_up
            self.camera_front = front * c + up * s
            rotated = True

        if rotated:
            self.camera_front /= np.linalg.norm(self.camera_front)

            camera_up = np.array([0, 0, 1], dtype=np.float32)

            # pitch is angle between camera_front and XY plane (around right vector)
            # pitch = arcsin(z component in rotated coordinate system)

            # Project front onto XY plane
            front_xy = np.array([self.camera_front[0], self.camera_front[1], 0])
            front_xy_norm = np.linalg.norm(front_xy)
            if front_xy_norm > 1e-6:
                front_xy /= front_xy_norm
            else:
                front_xy = np.array([1.0, 0.0, 0.0])  # fallback

            self.pitch = np.degrees(np.arcsin(self.camera_front[2]))  # z component

            # yaw is angle of front_xy vector in XY plane:
            self.yaw = np.degrees(np.arctan2(self.camera_front[1], self.camera_front[0]))

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

        unit_dirs = self.star_positions 

        if self.speed_fraction > 0:
            beta = self.speed_fraction
            gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
            v_hat = self.velocity_dir / np.linalg.norm(self.velocity_dir)
            dot = np.sum(unit_dirs * v_hat, axis=1, keepdims=True)

            # Relativistic aberration
            factor = (1 - beta ** 2) / (1 + beta * dot)
            aberrated = factor * (unit_dirs + (gamma - 1) * dot * v_hat + gamma * beta * v_hat)
            unit_dirs = aberrated / np.linalg.norm(aberrated, axis=1, keepdims=True)

            # Doppler shift
            doppler = gamma * (1 + beta * dot)
            shifted_temperatures = self.star_temperatures * doppler.flatten()

            # Brightness boost and gamma correction
            brightness_boost = doppler.flatten() ** 0.4
            boosted_brightness = self.star_brightness * brightness_boost

            shifted_colors = np.array([
                self.kelvin_to_rgb(t, b)
                for t, b in zip(shifted_temperatures, boosted_brightness)
            ], dtype=np.float32)

            final_brightness = boosted_brightness
        else:
            shifted_colors = self.star_colors
            final_brightness = self.star_brightness

        # Convert unit vectors to 3D positions
        positions = unit_dirs
        brightness_column = final_brightness.reshape(-1, 1)

        # Prepare VBO data: pos (3f), color (3f), brightness (1f)
        interleaved_data = np.hstack([positions, shifted_colors, brightness_column]).astype(np.float32)
        sorted_indices = np.argsort(interleaved_data[:, -1])  # sort by brightness column
        interleaved_data = interleaved_data[sorted_indices]
        self.vbo.write(interleaved_data.tobytes())

        # View-projection matrix
        view = Matrix44.look_at(np.zeros(3), self.camera_front, self.camera_up)
        proj = Matrix44.perspective_projection(FOV, self.width / self.height, 0.1, 2000.0)
        mvp = proj * view

        # Draw stars
        self.program['mvp'].write(mvp.astype('f4').tobytes())
        self.vao.render(mode=moderngl.POINTS)

        # Draw velocity markers if enabled and velocity is locked
        if self.show_velocity_markers and self.velocity_locked:
            self.marker_program['mvp'].write(mvp.astype('f4').tobytes())
            self.marker_vao.render(mode=moderngl.POINTS)

        # HUD: Speed, gamma, RA/Dec, FPS, Controls
        gamma_val = 1.0 / np.sqrt(1.0 - self.speed_fraction ** 2) if self.speed_fraction > 0 else 1.0
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
        self.yaw += dx * sensitivity
        self.pitch += -dy * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

        # Here:
        # Yaw rotates around Z (camera_up)
        # Pitch rotates around right axis (which is cross(camera_front, camera_up))

        # Calculate camera right vector
        camera_up = np.array([0, 0, 1], dtype=np.float32)
        # We'll calculate front from yaw and pitch:

        # Step 1: get horizontal direction from yaw (in XY plane)
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

        # Horizontal direction in XY plane (no pitch yet)
        horizontal = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0], dtype=np.float32)

        # Now apply pitch by rotating horizontal vector towards up (Z)
        # Pitch rotates camera front around right vector = cross(horizontal, up)
        right = np.cross(horizontal, camera_up)
        right /= np.linalg.norm(right)

        # Rotate horizontal vector by pitch around right axis
        def rotate_vector(vec, axis, angle_rad):
            axis = axis / np.linalg.norm(axis)
            cos_ang = np.cos(angle_rad)
            sin_ang = np.sin(angle_rad)
            return vec * cos_ang + np.cross(axis, vec) * sin_ang + axis * np.dot(axis, vec) * (1 - cos_ang)

        front = rotate_vector(horizontal, right, pitch_rad)
        self.camera_front = front / np.linalg.norm(front)

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