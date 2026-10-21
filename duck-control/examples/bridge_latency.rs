//! Measure the control loop's delay through the servo bridge, the delay the RL simulation's
//! command lag stands in for: from the moment a servo was sampled for the state a tick reads, to
//! the first reply in which that servo acts on the command the tick then sent.
//!
//!   bridge_latency [port] [kp] [step_rad] [events_per_servo] [compute_ms] [csv] [move_rad]
//!   bridge_latency /dev/ttyS7 1.0 0.3 10 0 /root/bridge_latency.csv 0.3
//!
//! With `move_rad` above 0, every servo but the one being stepped follows its own sine of that
//! amplitude at [`MOVE_HZ`], from a random phase, as a robot's joints all move at once; the
//! stepped servo holds, so its torque change stays clean to detect.
//!
//! Every servo holds where it is at `kp` and kd 0.05, ticking at 50 Hz as robotd does: a sample
//! request, `compute_ms` standing in for inference, then the targets. At a random tick every
//! half to one second, one servo's target toggles by `step_rad` (out, then back the next time),
//! servo after servo. The joints move, so they must be free to.
//!
//! Times are on the bridge's clock, so no clock sync is needed: a servo's sample time is the
//! sample's `bridge_time_us` less that servo's `age_us`, and the command's arrival is a state
//! frame's `bridge_time_us` less `command_age_us`. Between ticks the full state is polled as fast
//! as the link answers (about every 4 ms) to see the servo's replies; it replies about every 2 ms, so the
//! response lies between the last reply seen without it and the first one with it, and both are
//! reported. Torque responds first, the position a little later.
//!
//! Those bounds follow the polling, which is tied to the tick: a poll sent right after the
//! command queues behind it on the link and tends to see nothing yet, and the next comes a
//! round trip later. So after each step the polling starts a random 0 to 4 ms late, and every
//! reply seen counts as a sample of whether the servo had responded that long after its sample.
//! The share that had, by 0.5 ms bins, is the delay's distribution; its 5, 50 and 95% points
//! are printed.

use std::fs::File;
use std::io::Write as _;
use std::time::{Duration, Instant};

use duck_bridge_proto as proto;
use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::{JointTargets, NUM_JOINTS, RobotIo};

const HZ: f64 = 50.0;
/// The other servos' motion with `move_rad`: about a gait's frequency.
const MOVE_HZ: f64 = 1.5;
const KD: f64 = 0.05;
/// Ticks before the first step, for the hold to settle.
const SETTLE_TICKS: u64 = 50;
/// Stop watching a step after this long; it is then counted as not seen.
const WATCH: Duration = Duration::from_millis(100);
/// Leave this much of each tick free so the next state request starts on time.
const POLL_MARGIN: Duration = Duration::from_micros(4500);
/// Polling after a step starts up to this much late (see the module notes).
const POLL_JITTER_US: u64 = 4000;
/// Bins for the delay's distribution.
const BIN_US: i64 = 500;
const BINS: usize = 80;

/// A response is a change of this share of `kp * step` in torque, or of `step` in position.
const TORQUE_SHARE: f32 = 0.5;
const MOTION_SHARE: f32 = 0.1;

/// Small, fast and seeded from the clock: the step times only need to be spread over the tick.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

/// A servo reply, by when it happened on the bridge's clock.
fn reply_us(state: &proto::State, j: usize) -> u64 {
    state.bridge_time_us - state.servos[j].age_us as u64
}

/// The same for a sample.
fn sampled_us(sample: &proto::Sample, j: usize) -> u64 {
    sample.bridge_time_us - sample.servos[j].age_us as u64
}

/// One step being watched.
struct Event {
    servo: usize,
    segment: u8,
    out: bool,
    seq: u32,
    rtt_us: u64,
    host_us: u64,
    state_age_us: u64,
    sample_us: u64,
    torque_before: f32,
    position_before: f32,
    started: Instant,
    /// The reply counter of the last state frame seen, to count each reply once.
    last_reply: Option<u16>,
    /// Every reply seen after the step: µs since the sample, and whether torque had responded.
    seen: Vec<(i64, bool)>,
    last_quiet_torque_us: u64,
    last_quiet_motion_us: u64,
    receipt_us: Option<u64>,
    torque: Option<(u64, u64)>,
    motion: Option<(u64, u64)>,
}

impl Event {
    fn done(&self) -> bool {
        self.torque.is_some() && self.motion.is_some() && self.receipt_us.is_some()
    }

    /// Fold in a state frame read after the step was sent.
    fn see(&mut self, s: &proto::State, kp: f32, step: f32) {
        if self.receipt_us.is_none() && s.command_seq == self.seq {
            self.receipt_us = Some(s.bridge_time_us - s.command_age_us as u64);
        }
        let v = &s.servos[self.servo];
        if self.last_reply == Some(v.replies) {
            return;
        }
        self.last_reply = Some(v.replies);
        let at = reply_us(s, self.servo);
        let responded = (v.torque - self.torque_before).abs() > TORQUE_SHARE * kp * step;
        self.seen.push((self.since_sample(at), responded || self.torque.is_some()));
        if self.torque.is_none() {
            if responded {
                self.torque = Some((self.last_quiet_torque_us, at));
            } else {
                self.last_quiet_torque_us = at;
            }
        }
        if self.motion.is_none() {
            if (v.position - self.position_before).abs() > MOTION_SHARE * step {
                self.motion = Some((self.last_quiet_motion_us, at));
            } else {
                self.last_quiet_motion_us = at;
            }
        }
    }

    fn since_sample(&self, t: u64) -> i64 {
        t as i64 - self.sample_us as i64
    }
}

fn percentiles(label: &str, mut v: Vec<i64>) {
    if v.is_empty() {
        println!("  {label:34} none");
        return;
    }
    v.sort();
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize] as f64 / 1e3;
    println!(
        "  {label:34} min {:6.2}  median {:6.2}  p95 {:6.2}  max {:6.2} ms  (n {})",
        p(0.0),
        p(0.5),
        p(0.95),
        p(1.0),
        v.len()
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let port = args.next().unwrap_or_else(|| "/dev/ttyS7".into());
    let kp: f64 = args.next().map_or(1.0, |a| a.parse().expect("kp"));
    let step: f64 = args.next().map_or(0.3, |a| a.parse().expect("step_rad"));
    let per_servo: usize = args.next().map_or(10, |a| a.parse().expect("events_per_servo"));
    let compute = Duration::from_secs_f64(
        args.next().map_or(0.0, |a| a.parse::<f64>().expect("compute_ms")) / 1e3,
    );
    let csv_path = args.next().filter(|p| !p.is_empty() && p != "-");
    let move_rad: f64 = args.next().map_or(0.0, |a| a.parse().expect("move_rad"));
    assert!(step.abs() <= 0.5, "keep the step at 0.5 rad or less");
    assert!(move_rad.abs() <= 0.5, "keep the motion at 0.5 rad or less");

    let config = BridgeConfig {
        kp_per_gain: 0.01,
        kd: KD,
    };
    let mut io = BridgeIo::open(&port, duck_control::bridge::DEFAULT_BAUD, config)
        .unwrap_or_else(|e| {
            eprintln!("cannot open the bridge on {port}: {e}");
            std::process::exit(1);
        });
    let segments = io.info().segments;
    let servos: Vec<usize> = (0..NUM_JOINTS)
        .filter(|&j| segments[j] != proto::ABSENT)
        .collect();
    println!(
        "bridge {}: {} servos, kp {kp} kd {KD}, step {step} rad, {per_servo} steps each, compute {:.1} ms, others moving {move_rad} rad at {MOVE_HZ} Hz",
        io.info().firmware_str(),
        servos.len(),
        compute.as_secs_f64() * 1e3
    );

    io.set_gain((kp / config.kp_per_gain).round() as u16).expect("set gain");
    io.set_torque(true).expect("torque on");
    let home = io.present_positions().expect("present positions");
    let mut targets = home;

    let mut plan: Vec<(usize, bool)> = Vec::new();
    for &j in &servos {
        for k in 0..per_servo {
            plan.push((j, k % 2 == 0));
        }
        if per_servo % 2 == 1 {
            plan.push((j, false)); // end where it started
        }
    }
    let mut rng = XorShift(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            | 1,
    );

    let phases: Vec<f64> = (0..NUM_JOINTS)
        .map(|_| rng.range(0, 6283) as f64 / 1000.0)
        .collect();
    // The servo whose steps are being timed, which holds rather than moving.
    let mut stepping: Option<usize> = None;
    let begin = Instant::now();

    let period = Duration::from_secs_f64(1.0 / HZ);
    let mut next = Instant::now();
    let mut tick: u64 = 0;
    let mut due = SETTLE_TICKS;
    let mut planned = plan.into_iter().peekable();
    let mut active: Option<Event> = None;
    let mut finished: Vec<Event> = Vec::new();
    let mut missed_states = 0u64;
    let mut late_ticks = 0u64;
    let mut out_of_plan = false;

    while !(out_of_plan && active.is_none()) {
        let t0 = Instant::now();
        if t0 > next + period / 4 {
            late_ticks += 1;
        }
        let sample = match io.sample() {
            Ok(s) => s,
            Err(_) => {
                missed_states += 1;
                next += period;
                continue;
            }
        };
        let rtt = t0.elapsed();
        let t_rx = Instant::now();
        std::thread::sleep(compute);

        let mut stepped = None;
        // The next servo to be timed stops moving and settles at home before its first step.
        if let Some(&(j, _)) = planned.peek()
            && active.is_none()
            && stepping != Some(j)
        {
            stepping = Some(j);
            targets[j] = home[j];
            due = due.max(tick + 25);
        }
        if active.is_none() && tick >= due {
            match planned.next() {
                Some((j, out)) => {
                    targets[j] = if out { home[j] + step } else { home[j] };
                    stepped = Some((j, out));
                    due = tick + rng.range(25, 50);
                }
                None => out_of_plan = true,
            }
        }
        if move_rad != 0.0 {
            let t = begin.elapsed().as_secs_f64();
            for &k in &servos {
                if Some(k) != stepping {
                    targets[k] = home[k]
                        + move_rad * (2.0 * std::f64::consts::PI * MOVE_HZ * t + phases[k]).sin();
                }
            }
        }
        io.write(&JointTargets::new(targets)).expect("command");
        let sent_seq = io.command_seq();
        if let Some((j, out)) = stepped {
            let v = &sample.servos[j];
            active = Some(Event {
                servo: j,
                segment: segments[j],
                out,
                seq: sent_seq,
                rtt_us: rtt.as_micros() as u64,
                host_us: t_rx.elapsed().as_micros() as u64,
                state_age_us: v.age_us as u64,
                sample_us: sampled_us(&sample, j),
                torque_before: v.torque,
                position_before: v.position,
                started: Instant::now(),
                last_reply: None,
                seen: Vec::new(),
                last_quiet_torque_us: sampled_us(&sample, j),
                last_quiet_motion_us: sampled_us(&sample, j),
                receipt_us: None,
                torque: None,
                motion: None,
            });
        }

        next += period;
        if stepped.is_some() {
            std::thread::sleep(Duration::from_micros(rng.range(0, POLL_JITTER_US)));
        }
        // Watch the step between ticks.
        while let Some(e) = active.as_mut() {
            if e.done() || e.started.elapsed() > WATCH {
                finished.push(active.take().unwrap());
                break;
            }
            if Instant::now() + POLL_MARGIN > next {
                break;
            }
            match io.state() {
                Ok(s) => e.see(&s, kp as f32, step as f32),
                Err(_) => missed_states += 1,
            }
        }
        if let Some(wait) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        tick += 1;
    }

    io.set_torque(false).expect("torque off");

    if let Some(path) = csv_path {
        let mut f = File::create(&path).expect("csv");
        writeln!(
            f,
            "servo,segment,out,rtt_us,host_us,state_age_us,receipt_us,torque_lo_us,torque_hi_us,motion_lo_us,motion_hi_us"
        )
        .unwrap();
        let opt = |v: Option<i64>| v.map_or(String::new(), |x| x.to_string());
        for e in &finished {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{}",
                e.servo,
                e.segment,
                e.out as u8,
                e.rtt_us,
                e.host_us,
                e.state_age_us,
                opt(e.receipt_us.map(|t| e.since_sample(t))),
                opt(e.torque.map(|t| e.since_sample(t.0))),
                opt(e.torque.map(|t| e.since_sample(t.1))),
                opt(e.motion.map(|t| e.since_sample(t.0))),
                opt(e.motion.map(|t| e.since_sample(t.1))),
            )
            .unwrap();
        }
        println!("wrote {} steps to {path}", finished.len());
    }

    let col = |f: &dyn Fn(&Event) -> Option<i64>| finished.iter().filter_map(f).collect::<Vec<_>>();
    println!(
        "{} steps, {} without a torque response, {} state requests unanswered, {} ticks started late",
        finished.len(),
        finished.iter().filter(|e| e.torque.is_none()).count(),
        missed_states,
        late_ticks
    );
    println!("From the servo's sample in the state a tick read:");
    percentiles("sample built (servo age)", col(&|e| Some(e.state_age_us as i64)));
    percentiles("command reached the bridge", col(&|e| e.receipt_us.map(|t| e.since_sample(t))));
    percentiles("torque response, earliest", col(&|e| e.torque.map(|t| e.since_sample(t.0))));
    percentiles("torque response, latest", col(&|e| e.torque.map(|t| e.since_sample(t.1))));
    percentiles("position response, latest", col(&|e| e.motion.map(|t| e.since_sample(t.1))));
    // The delay's distribution from every reply seen: the share responded by bin.
    let mut yes = [0u32; BINS];
    let mut all = [0u32; BINS];
    for (t, responded) in finished.iter().flat_map(|e| e.seen.iter()) {
        let b = (t / BIN_US).clamp(0, BINS as i64 - 1) as usize;
        all[b] += 1;
        yes[b] += *responded as u32;
    }
    let share: Vec<(f64, f64)> = (0..BINS)
        .filter(|&b| all[b] > 0)
        .map(|b| ((b as f64 + 0.5) * BIN_US as f64 / 1e3, yes[b] as f64 / all[b] as f64))
        .collect();
    let crossing = |q: f64| share.iter().find(|(_, f)| *f >= q).map(|(t, _)| *t);
    let fmt = |v: Option<f64>| v.map_or("-".into(), |t| format!("{t:.2}"));
    println!(
        "  torque response, from every reply   5% {} ms  50% {} ms  95% {} ms  ({} replies, {:.1} ms bins)",
        fmt(crossing(0.05)),
        fmt(crossing(0.5)),
        fmt(crossing(0.95)),
        all.iter().sum::<u32>(),
        BIN_US as f64 / 1e3
    );
    let bins: Vec<String> = share
        .iter()
        .filter(|(_, f)| *f > 0.0 && *f < 1.0)
        .map(|(t, f)| format!("{t:.2}:{:.0}%", f * 100.0))
        .collect();
    println!("  share responded by bin (ms:share)   {}", bins.join(" "));
    println!("On the head:");
    percentiles("sample round trip", col(&|e| Some(e.rtt_us as i64)));
    percentiles("sample received to targets sent", col(&|e| Some(e.host_us as i64)));
    println!("Torque response, latest, by segment:");
    for seg in 0..4u8 {
        let v = col(&|e| (e.segment == seg).then_some(e.torque.map(|t| e.since_sample(t.1))).flatten());
        if !v.is_empty() {
            percentiles(["A", "B", "C", "D"][seg as usize], v);
        }
    }
}
