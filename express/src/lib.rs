//! Expression: the outputs that show how the robot is, and what each should be doing.
//!
//! Not part of Pollen's robot. Two outputs today, both visible:
//!
//! - **The eye**, one WS2812-type RGB LED in the head. It shows what the robot is doing at a
//!   glance, and is the "LED under software control" that `docs/project/roadmap.md` (the
//!   visible camera indicator) and `docs/design/app-path-design.md` (`identify`) found missing.
//! - **The fan**, a 2-wire 5 V axial fan on a PWM-driven MOSFET. Its first job is cooling the
//!   compute module; its second is expression (a burst of excitement, a slow swell).
//!
//! This file is the part with no hardware in it, so it is the part with tests: colours,
//! patterns, the layer stack that decides who wins, the mapping from `robot.state` to a look,
//! the fan's thermal curve and stall handling, and the SPI encoding of the LED's single-wire
//! protocol. `main.rs` owns the devices, the sockets and the clock.
//!
//! **Layers, highest first.** The eye and the fan each have a [`Stack`] of these; each layer
//! holds at most one look per output, and the highest non-empty one is shown.
//!
//! | Layer | Set by | For |
//! |---|---|---|
//! | `privacy` | a client (`express.set`) | the camera is being watched. Nothing below can hide it |
//! | `fault` | robotd's state | fallen |
//! | `identify` | a client (`express.identify`) | "this one is mine" |
//! | `mood` | a client (`express.set`) | whatever a behaviour layer wants to say |
//! | `ambient` | robotd's state | asleep, limp, holding, standing, walking |
//!
//! `fault` and `ambient` are derived, never set: a client that could paint over "fallen" or
//! fake "walking" would make the eye a liar about the one thing it is for.
//!
//! **The fan has a floor.** Cooling is not a layer a client can outrank: the speed is the
//! larger of what the winning layer asks for and what the SoC temperature needs
//! ([`FanCurve`]). A mood can spin the fan up or animate it above that floor, never below it.

use std::time::Duration;

use duck_ipc_proto::RobotState;
use serde::{Deserialize, Serialize};

/// An RGB colour at full scale. Brightness is applied at output, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const OFF: Rgb = Rgb(0, 0, 0);

    fn scale(self, k: f32) -> Rgb {
        let f = |c: u8| (f32::from(c) * k.clamp(0.0, 1.0)).round() as u8;
        Rgb(f(self.0), f(self.1), f(self.2))
    }
}

/// How a colour moves over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pattern {
    /// Steady.
    #[default]
    Solid,
    /// A slow sine between dim and full: alive, not asking for anything.
    Breathe,
    /// Half the period on, half off: wants attention.
    Blink,
    /// A short flash at the start of each period: present, quietly.
    Pulse,
}

impl Pattern {
    /// The pattern's level, 0 to 1, at `t` seconds into a cycle of `period_s`.
    pub fn factor(self, t: f32, period_s: f32) -> f32 {
        let phase = (t / period_s.max(0.05)).fract();
        match self {
            Pattern::Solid => 1.0,
            // 0.1 to 1.0, so the trough is dim rather than off: off reads as "dead".
            Pattern::Breathe => 0.1 + 0.9 * (0.5 - 0.5 * (std::f32::consts::TAU * phase).cos()),
            Pattern::Blink => {
                if phase < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Pattern::Pulse => {
                if phase < 0.12 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// What a layer asks the eye to show.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Look {
    pub colour: Rgb,
    #[serde(default)]
    pub pattern: Pattern,
    /// One cycle of the pattern, seconds. Ignored for `solid`.
    #[serde(default = "default_period")]
    pub period_s: f32,
}

fn default_period() -> f32 {
    2.0
}

impl Look {
    pub const fn new(colour: Rgb, pattern: Pattern, period_s: f32) -> Self {
        Self {
            colour,
            pattern,
            period_s,
        }
    }

    /// The colour at `t` seconds since the daemon started, before brightness.
    pub fn at(&self, t: f32) -> Rgb {
        self.colour.scale(self.pattern.factor(t, self.period_s))
    }

    /// Whether the output changes with time, so the render loop knows it may idle.
    pub fn animated(&self) -> bool {
        self.pattern != Pattern::Solid
    }
}

/// The layer stack, highest first. See the module docs for who may set which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    Privacy,
    Fault,
    Identify,
    Mood,
    Ambient,
}

impl Layer {
    pub const ALL: [Layer; 5] = [
        Layer::Privacy,
        Layer::Fault,
        Layer::Identify,
        Layer::Mood,
        Layer::Ambient,
    ];

    /// May a client write this layer? `fault` and `ambient` come from robotd and nowhere else.
    pub fn client_settable(self) -> bool {
        matches!(self, Layer::Privacy | Layer::Identify | Layer::Mood)
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One layer's request, with an optional expiry in daemon seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry<T> {
    look: T,
    until: Option<f32>,
}

/// The stack itself, one per output. Time is passed in as seconds since start, so tests need
/// no clock.
#[derive(Debug, Clone)]
pub struct Stack<T> {
    layers: [Option<Entry<T>>; 5],
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self {
            layers: [None, None, None, None, None],
        }
    }
}

impl<T: Copy> Stack<T> {
    pub fn set(&mut self, layer: Layer, look: T, now: f32, ttl: Option<Duration>) {
        let until = ttl.map(|d| now + d.as_secs_f32());
        self.layers[layer.index()] = Some(Entry { look, until });
    }

    pub fn clear(&mut self, layer: Layer) {
        self.layers[layer.index()] = None;
    }

    /// The winning layer and its look, dropping anything that has expired.
    pub fn top(&mut self, now: f32) -> Option<(Layer, T)> {
        for layer in Layer::ALL {
            let slot = &mut self.layers[layer.index()];
            if let Some(entry) = slot {
                if entry.until.is_some_and(|until| now >= until) {
                    *slot = None;
                    continue;
                }
                return Some((layer, entry.look));
            }
        }
        None
    }

    /// Every layer that is set, highest first, for `express.status`.
    pub fn active(&self) -> Vec<(Layer, T)> {
        Layer::ALL
            .into_iter()
            .filter_map(|l| self.layers[l.index()].map(|e| (l, e.look)))
            .collect()
    }
}

// ── What robotd's state looks like ───────────────────────────────────────────

/// robotd is not running, or not answering: the body has no brain. Dim, slow, white.
pub const ASLEEP: Look = Look::new(Rgb(255, 255, 255), Pattern::Breathe, 4.0);
/// Torque off. Soft violet, breathing.
pub const LIMP: Look = Look::new(Rgb(140, 0, 255), Pattern::Breathe, 3.0);
/// Running, no policy driving (`held`). Blue, breathing.
pub const HOLDING: Look = Look::new(Rgb(0, 60, 255), Pattern::Breathe, 3.0);
/// The stand policy, or the walk policy with nothing asked of it. Steady blue.
pub const STANDING: Look = Look::new(Rgb(0, 60, 255), Pattern::Solid, 2.0);
/// Walking. Steady cyan.
pub const WALKING: Look = Look::new(Rgb(0, 220, 200), Pattern::Solid, 2.0);
/// Fallen. Red, blinking at 2 Hz.
pub const FALLEN: Look = Look::new(Rgb(255, 0, 0), Pattern::Blink, 0.5);
/// `express.identify`. White, blinking fast enough not to be mistaken for anything else.
pub const IDENTIFY: Look = Look::new(Rgb(255, 255, 255), Pattern::Blink, 0.25);

/// Below this commanded speed (m/s or rad/s, any axis) the robot counts as standing.
const MOVING: f64 = 0.02;

/// The `fault` and `ambient` layers for one `robot.state` frame.
pub fn from_state(state: &RobotState) -> (Option<Look>, Look) {
    let fault = state.safety.fallen.then_some(FALLEN);
    let moving = state.movement.applied.iter().any(|v| v.abs() > MOVING);
    let ambient = if state.safety.limp {
        LIMP
    } else {
        match state.policy.as_str() {
            "walk" if moving => WALKING,
            "walk" | "stand" => STANDING,
            _ => HOLDING,
        }
    };
    (fault, ambient)
}

// ── The fan ──────────────────────────────────────────────────────────────────

/// What a layer asks the fan to do: a speed, 0 to 1, moved by a pattern.
///
/// The same patterns as the eye, and they read the same way on a fan: `breathe` swells and
/// settles, `blink` is bursts, `pulse` is a short puff each period. The fan's own inertia
/// smooths them, so periods under a couple of seconds mostly come out as a lower steady speed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FanLook {
    pub level: f32,
    #[serde(default)]
    pub pattern: Pattern,
    #[serde(default = "default_fan_period")]
    pub period_s: f32,
}

fn default_fan_period() -> f32 {
    4.0
}

impl FanLook {
    /// The requested speed at `t`, 0 to 1.
    pub fn at(&self, t: f32) -> f32 {
        self.level.clamp(0.0, 1.0) * self.pattern.factor(t, self.period_s)
    }

    pub fn animated(&self) -> bool {
        self.pattern != Pattern::Solid && self.level > 0.0
    }
}

/// The cooling floor: off below `off_c`, and once the SoC has passed `on_c`, a ramp from
/// `min` at `on_c` to full at `full_c`. Between `off_c` and `on_c` it holds whatever it was
/// doing, so a SoC sitting near one threshold does not cycle the fan.
#[derive(Debug, Clone, Copy)]
pub struct FanCurve {
    pub off_c: f32,
    pub on_c: f32,
    pub full_c: f32,
    pub min: f32,
    running: bool,
}

impl FanCurve {
    pub fn new(off_c: f32, on_c: f32, full_c: f32, min: f32) -> Self {
        Self {
            off_c,
            on_c,
            full_c,
            min,
            running: false,
        }
    }

    /// The floor for this temperature. `None` is an unreadable temperature, which is full
    /// speed: not knowing is not a reason to let the SoC cook.
    pub fn floor(&mut self, temp_c: Option<f32>) -> f32 {
        let Some(t) = temp_c else {
            return 1.0;
        };
        if t >= self.on_c {
            self.running = true;
        } else if t < self.off_c {
            self.running = false;
        }
        if !self.running {
            return 0.0;
        }
        let span = (self.full_c - self.on_c).max(0.1);
        (self.min + (1.0 - self.min) * ((t - self.on_c) / span)).clamp(self.min, 1.0)
    }
}

/// From a wanted speed to a PWM duty, for a 2-wire fan on a low-side switch.
///
/// Such a fan stalls below some duty (its own electronics brown out), so anything above zero
/// is raised to `min`; and it may not start from rest at `min`, so a start from stopped runs at
/// full for `kick_s` first.
#[derive(Debug, Clone, Copy)]
pub struct FanDrive {
    pub min: f32,
    pub kick_s: f32,
    kick_until: Option<f32>,
    running: bool,
}

impl FanDrive {
    pub fn new(min: f32, kick_s: f32) -> Self {
        Self {
            min,
            kick_s,
            kick_until: None,
            running: false,
        }
    }

    /// The duty for `want` (0 to 1) at `now` seconds.
    pub fn duty(&mut self, want: f32, now: f32) -> f32 {
        if want <= 0.0 {
            self.running = false;
            self.kick_until = None;
            return 0.0;
        }
        if !self.running {
            self.running = true;
            self.kick_until = Some(now + self.kick_s);
        }
        if self.kick_until.is_some_and(|until| now < until) {
            return 1.0;
        }
        want.clamp(self.min, 1.0)
    }
}

// ── Output ───────────────────────────────────────────────────────────────────

/// Byte order of the three colours on the wire. WS2812-family parts are GRB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Order {
    #[default]
    Grb,
    Rgb,
}

/// SPI clock for [`encode`]: three SPI bits per LED bit gives 1.25 µs per LED bit, and highs of
/// 0.42 µs (a 0) and 0.83 µs (a 1), inside the WS2812's 0.4 and 0.8 µs ± 0.15 µs.
pub const SPI_HZ: u32 = 2_400_000;

/// Zero bytes on each side of a frame. 96 bytes at 2.4 MHz is 320 µs of low line, past the
/// 280 µs reset of current WS2812B parts, so a frame always starts and latches cleanly.
const PAD: usize = 96;

/// Gamma for the brightness curve: linear PWM looks bright almost at once and then flat, so a
/// breathe would spend most of its cycle looking full.
const GAMMA: f32 = 2.2;

/// One frame for one LED, ready for a single SPI write.
pub fn encode(rgb: Rgb, brightness: f32, order: Order) -> Vec<u8> {
    let level = |c: u8| -> u8 {
        let x = f32::from(c) / 255.0;
        (x.powf(GAMMA) * brightness.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    let (r, g, b) = (level(rgb.0), level(rgb.1), level(rgb.2));
    let bytes = match order {
        Order::Grb => [g, r, b],
        Order::Rgb => [r, g, b],
    };
    let mut bits: u128 = 0;
    for byte in bytes {
        for i in (0..8).rev() {
            bits = (bits << 3) | if byte >> i & 1 == 1 { 0b110 } else { 0b100 };
        }
    }
    let mut frame = vec![0u8; PAD];
    frame.extend_from_slice(&bits.to_be_bytes()[16 - 9..]);
    frame.extend(std::iter::repeat_n(0u8, PAD));
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Look = Look::new(Rgb(255, 0, 0), Pattern::Solid, 1.0);
    const GREEN: Look = Look::new(Rgb(0, 255, 0), Pattern::Solid, 1.0);

    /// The higher layer wins whatever order the two were set in.
    #[test]
    fn the_highest_layer_wins() {
        let mut s = Stack::default();
        s.set(Layer::Privacy, RED, 0.0, None);
        s.set(Layer::Mood, GREEN, 0.0, None);
        assert_eq!(s.top(0.0), Some((Layer::Privacy, RED)));
        s.clear(Layer::Privacy);
        assert_eq!(s.top(0.0), Some((Layer::Mood, GREEN)));
    }

    /// A timed layer drops out on its own and uncovers the one beneath.
    #[test]
    fn an_expired_layer_uncovers_the_next() {
        let mut s = Stack::default();
        s.set(Layer::Ambient, GREEN, 0.0, None);
        s.set(Layer::Identify, RED, 0.0, Some(Duration::from_secs(5)));
        assert_eq!(s.top(4.9).map(|(l, _)| l), Some(Layer::Identify));
        assert_eq!(s.top(5.0).map(|(l, _)| l), Some(Layer::Ambient));
        assert_eq!(
            s.active().len(),
            1,
            "expired entries are removed, not just skipped"
        );
    }

    /// Only privacy, identify and mood are a client's to write.
    #[test]
    fn derived_layers_are_not_client_settable() {
        assert!(!Layer::Fault.client_settable());
        assert!(!Layer::Ambient.client_settable());
        assert!(Layer::Privacy.client_settable());
    }

    #[test]
    fn patterns_move_as_described() {
        let blink = Look::new(Rgb(10, 20, 30), Pattern::Blink, 1.0);
        assert_eq!(blink.at(0.25), Rgb(10, 20, 30));
        assert_eq!(blink.at(0.75), Rgb::OFF);
        let breathe = Look::new(Rgb(200, 200, 200), Pattern::Breathe, 2.0);
        assert_eq!(breathe.at(0.0), Rgb(20, 20, 20), "trough is dim, not off");
        assert_eq!(breathe.at(1.0), Rgb(200, 200, 200));
        assert!(!RED.animated() && blink.animated());
    }

    /// GRB on the wire, three SPI bits per LED bit, framed by low padding.
    #[test]
    fn encoding_is_grb_three_bits_per_bit() {
        let f = encode(Rgb(255, 0, 0), 1.0, Order::Grb);
        assert_eq!(f.len(), PAD + 9 + PAD);
        assert!(f[..PAD].iter().chain(&f[PAD + 9..]).all(|&b| b == 0));
        let data = &f[PAD..PAD + 9];
        // Green 0x00 is eight 100s: 100100100100100100100100 = 0x92 0x49 0x24.
        assert_eq!(&data[..3], &[0x92, 0x49, 0x24]);
        // Red 0xff is eight 110s: 110110110110110110110110 = 0xdb 0x6d 0xb6.
        assert_eq!(&data[3..6], &[0xdb, 0x6d, 0xb6]);
        assert_eq!(&data[6..], &[0x92, 0x49, 0x24]);
    }

    #[test]
    fn brightness_scales_through_gamma() {
        let full = encode(Rgb(255, 255, 255), 0.5, Order::Rgb);
        let off = encode(Rgb::OFF, 1.0, Order::Rgb);
        // 255 * 0.5 = 127.5 -> 128 = 0b1000_0000: one 110 then seven 100s.
        assert_eq!(&full[PAD..PAD + 3], &[0xd2, 0x49, 0x24]);
        assert_eq!(&off[PAD..PAD + 3], &[0x92, 0x49, 0x24]);
    }

    #[test]
    fn state_maps_to_a_look() {
        let mut state: RobotState = serde_json::from_value(serde_json::json!({
            "t": 0.0,
            "move": {"requested": [0.0, 0.0, 0.0], "applied": [0.0, 0.0, 0.0], "limited_by": []},
            "head": [0.0, 0.0, 0.0, 0.0],
            "policy": "walk",
            "safety": {"fallen": false, "limp": false, "gravity": [0.0, 0.0, -1.0]},
            "loop": {"hz": 50.0, "missed": 0},
            "joints": [], "targets": []
        }))
        .expect("a minimal robot.state parses");
        assert_eq!(from_state(&state), (None, STANDING));
        state.movement.applied = [0.1, 0.0, 0.0];
        assert_eq!(from_state(&state), (None, WALKING));
        state.policy = "held".to_owned();
        assert_eq!(from_state(&state).1, HOLDING);
        state.safety.limp = true;
        assert_eq!(from_state(&state).1, LIMP);
        state.safety.fallen = true;
        assert_eq!(from_state(&state).0, Some(FALLEN));
    }

    #[test]
    fn the_cooling_floor_has_hysteresis_and_a_ramp() {
        let mut c = FanCurve::new(50.0, 60.0, 75.0, 0.3);
        assert_eq!(c.floor(Some(55.0)), 0.0, "below on_c from cold: off");
        assert_eq!(c.floor(Some(60.0)), 0.3);
        assert_eq!(c.floor(Some(55.0)), 0.3, "between thresholds: holds");
        assert!((c.floor(Some(67.5)) - 0.65).abs() < 1e-4);
        assert_eq!(c.floor(Some(90.0)), 1.0);
        assert_eq!(c.floor(Some(49.9)), 0.0, "below off_c: off again");
    }

    /// Not knowing the temperature is not a reason to let the SoC cook.
    #[test]
    fn an_unreadable_temperature_is_full_speed() {
        assert_eq!(FanCurve::new(50.0, 60.0, 75.0, 0.3).floor(None), 1.0);
    }

    /// A 2-wire fan stalls below its minimum and may not start from rest at it.
    #[test]
    fn the_drive_kicks_from_rest_and_never_asks_for_a_stall() {
        let mut d = FanDrive::new(0.3, 0.5);
        assert_eq!(d.duty(0.0, 0.0), 0.0);
        assert_eq!(d.duty(0.1, 1.0), 1.0, "start from rest: kick");
        assert_eq!(d.duty(0.1, 1.4), 1.0);
        assert_eq!(
            d.duty(0.1, 1.5),
            0.3,
            "after the kick: clamped to min, not 0.1"
        );
        assert_eq!(d.duty(0.8, 2.0), 0.8, "already running: no second kick");
        assert_eq!(d.duty(0.0, 3.0), 0.0);
        assert_eq!(d.duty(0.5, 3.1), 1.0, "stopped again, so kicks again");
    }

    #[test]
    fn fan_patterns_scale_the_level() {
        let bursts = FanLook {
            level: 0.6,
            pattern: Pattern::Blink,
            period_s: 2.0,
        };
        assert!((bursts.at(0.5) - 0.6).abs() < 1e-6);
        assert_eq!(bursts.at(1.5), 0.0);
        assert!(bursts.animated());
        assert!(
            !FanLook {
                level: 0.0,
                pattern: Pattern::Blink,
                period_s: 2.0
            }
            .animated()
        );
    }
}
