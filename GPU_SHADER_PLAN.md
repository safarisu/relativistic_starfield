# Plan: relativistic effects on the GPU

Status: planned, not started. Do this before switching to a larger star catalog.

## Why

`on_draw` currently computes aberration, Doppler shift and star colors in NumPy every frame,
then re-uploads the whole vertex buffer. With the Bright Star Catalogue (~9,100 stars) this is
fast enough, but the cost grows linearly with star count. A catalog like Hipparcos (~118,000 stars)
would likely exceed the 16 ms frame budget, and Gaia-scale data is out of reach on the CPU.
In the vertex shader these effects are essentially free.

Not worth doing for the current catalog alone. It is the prerequisite for a denser sky.

## Design

**Static vertex buffer, uploaded once at startup.** Per star:
- `in_direction` (3f): rest-frame unit direction (`self.star_positions`)
- `in_temperature` (1f): rest-frame temperature in K
- `in_brightness` (1f): base brightness

**Uniforms, set by Python each frame:**
- `mvp` (mat4), unchanged
- `v_hat` (vec3): unit direction of travel
- `beta`, `one_minus_beta`, `k` (floats): see precision notes. Compute these in Python in
  float64, then pass them to the shader.

**Vertex shader:** aberrate the direction, compute the Doppler factor `D`, then
`temperature * D` → blackbody color (port of `kelvin_to_rgb`) and brightness `* D^0.4`.
Pass color and brightness to the fragment shader as now. The fragment shader is unchanged.

**Python side after the change:**
- `on_draw` no longer builds or writes the VBO. It only sets uniforms and renders.
- With `beta = 0` the shader must reproduce the rest-frame view exactly, which removes the
  `if self.speed_fraction > 0` branch.
- Velocity markers and the HUD are unaffected.

**Constellation lines:** the line VBO (rest-frame points, pre-subdivided along great circles)
is also static. The line vertex shader applies the same aberration function as the star shader
(no Doppler, since lines aren't light). Share the GLSL function between the two programs.

**Draw order / blending:** pyglet's text labels call `glDisable(GL_BLEND)` after drawing, so in
practice stars render without blending, and `on_draw` sorts them by brightness so the brightest
are drawn last. A static VBO can't be re-sorted each frame. Before doing the GPU move, decide
between re-enabling additive blending explicitly every frame (which makes order irrelevant, but
changes how dense regions look), or using a depth test keyed on brightness.

## Precision (the important part)

GPUs compute in float32. At `MAX_SPEED = 0.99999`, a straight port of the current formulas
loses most of its precision for stars behind the observer: `1 + beta * dot` approaches
0 there (≈1e-5), so stars near the anti-velocity direction would jitter or land in the wrong place.
NumPy hides this today because it uses float64.

Use numerically stable forms instead. Let θ be the angle between the star and `v_hat`:

- **Angle:** `theta = atan(length(cross(n, v_hat)), dot(n, v_hat))`. This is stable at
  any angle, unlike `acos(dot)`.
- **Aberration (half-angle form):** `tan(θ'/2) = k · tan(θ/2)`, with
  `k = sqrt((1 - β) / (1 + β))`. Evaluate as
  `theta_obs = 2.0 * atan(k * sin(theta / 2.0), cos(theta / 2.0))` to avoid infinities at θ = π.
- **New direction:** `cos(θ') · v_hat + sin(θ') · perp`, where
  `perp = normalize(n - dot(n, v_hat) · v_hat)`. Guard the case where the star lies exactly on
  the velocity axis (`perp` undefined): the direction is unchanged there.
- **Doppler factor, from the observed angle:** `D = 1 / (γ · (1 - β cos θ'))`, with the
  denominator rewritten as `one_minus_beta + 2β · sin²(θ'/2)`. This stays accurate both ahead
  (θ' → 0) and behind. It is mathematically identical to the current `γ(1 + β · dot)`.

## Verification

The NumPy functions in `main.py` (`relativistic_transform`, `kelvin_to_rgb`) are the reference
implementation. Keep them.

1. Port the stable formulas to NumPy first, using float32 throughout. Check them against
   the float64 reference on random directions at β = 0.1, 0.9 and 0.99999. This tests the math
   before any GLSL is involved.
2. Write the shader. Use transform feedback, or read back a small test render, to compare the
   shader's output with the reference for a known set of stars.
3. Visual checks: at rest the view matches the current build. At high speed, stars bunch up
   ahead and turn blue, the sky behind turns dark red, and nothing jitters while turning.

## Cleanup that comes with it

- Remove the per-frame `np.hstack` and `self.vbo.write(...)` from `on_draw`.
- Consider replacing the f-string-spliced `MIN_POINT_SIZE` / `MAX_POINT_SIZE` in the shader
  source with uniforms.
