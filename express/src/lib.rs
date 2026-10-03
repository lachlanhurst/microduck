//! The eye: one WS2812-type RGB LED in the head, and what it should be showing.
//!
//! Not part of Pollen's robot. The eye shows what the robot is doing at a glance, and is the
//! "LED under software control" that `docs/project/roadmap.md` (the visible camera indicator)
//! and `docs/design/app-path-design.md` (`identify`) both found missing.
//!
//! This file is the part with no hardware in it, so it is the part with tests: colours,
//! patterns, the layer stack that decides who wins, the mapping from `robot.state` to a look,
//! and the SPI encoding of the LED's single-wire protocol. `main.rs` owns the device, the
//! sockets and the clock.
//!
//! **Layers, highest first.** Each layer holds at most one [`Look`]; the highest non-empty one
//! is shown.
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
        let period = self.period_s.max(0.05);
        let phase = (t / period).fract();
        match self.pattern {
            Pattern::Solid => self.colour,
            Pattern::Breathe => {
                // 0.1 to 1.0, so the trough is dim rather than off: off reads as "dead".
                let s = 0.5 - 0.5 * (std::f32::consts::TAU * phase).cos();
                self.colour.scale(0.1 + 0.9 * s)
            }
            Pattern::Blink => {
                if phase < 0.5 {
                    self.colour
                } else {
                    Rgb::OFF
                }
            }
            Pattern::Pulse => {
                if phase < 0.12 {
                    self.colour
                } else {
                    Rgb::OFF
                }
            }
        }
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
struct Entry {
    look: Look,
    until: Option<f32>,
}

/// The stack itself. Time is passed in as seconds since start, so tests need no clock.
#[derive(Debug, Clone, Default)]
pub struct Stack {
    layers: [Option<Entry>; 5],
}

impl Stack {
    pub fn set(&mut self, layer: Layer, look: Look, now: f32, ttl: Option<Duration>) {
        let until = ttl.map(|d| now + d.as_secs_f32());
        self.layers[layer.index()] = Some(Entry { look, until });
    }

    pub fn clear(&mut self, layer: Layer) {
        self.layers[layer.index()] = None;
    }

    /// The winning layer and its look, dropping anything that has expired.
    pub fn top(&mut self, now: f32) -> Option<(Layer, Look)> {
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
    pub fn active(&self) -> Vec<(Layer, Look)> {
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
}
