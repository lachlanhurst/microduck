//! Left against right: swing a joint and its mirror twin through the same sine, one at a time,
//! with the robot held in the air, and record how each follows. For finding a sticky or loose
//! side (friction, play) that calibration offsets cannot explain. Runs without robotd, which must
//! be stopped first since it owns the bridge link.
//!
//!   joint_sweep [--joint hip_yaw] [--amp 15] [--freqs 0.5,2] [--cycles 4] [--kp K] [--kd D]
//!               [--config /etc/robot/robotd.toml] [--out sweep.json]
//!
//! Every joint ramps to the home pose ([`DEFAULT_POSITION`]) over 3 s and holds it; hold the robot
//! by the trunk, upright, legs hanging free, and keep it still. Then for each frequency, the left
//! joint swings `--amp` degrees either way for `--cycles` cycles (faded in and out over a cycle
//! each), then the right one, mirrored, with 1.5 s of hold between. The gains default to robotd's
//! (`[bus]`: the policy's gain of 200 times `bridge_kp_per_gain`, and `bridge_kd`), so the joints
//! behave as they do under the policy.
//!
//! Each tick (20 ms) is one full state request, so the recording carries the servo's torque and
//! the output-shaft encoder as well as the rotor position. JSON:
//!
//! ```text
//! {"source":"joint_sweep","joint":"hip_yaw","amp_deg":15,"kp":1.0,"kd":0.07,
//!  "frames":[{"t":0.02,"phase":"left 0.5 Hz","targets":[...15],"positions":[...15],
//!             "velocities":[...15],"torques":[...15],"encoders":[...15]}, ...]}
//! ```
//!
//! Positions and targets are joint angles (robotd's zeros), rad; torques are the servos' own,
//! N·m; encoders raw 13-bit counts. The Dukki repository's `scripts/policy/sweep_summary.py`
//! compares the two sides.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use duck_bridge_proto as proto;
use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::model::joint_index;
use duck_control::{DEFAULT_POSITION, JointTargets, NUM_JOINTS, RobotIo};
use serde_json::json;

const PERIOD: Duration = Duration::from_millis(20);
const HOME_RAMP_S: f64 = 3.0;
const HOLD_S: f64 = 1.5;
const GAIN: u16 = 100;
/// The policy's gain, which robotd scales by `bridge_kp_per_gain`.
const POLICY_GAIN: f64 = 200.0;

struct Args {
    config: PathBuf,
    joint: String,
    amp: f64,
    freqs: Vec<f64>,
    cycles: f64,
    kp: Option<f64>,
    kd: Option<f64>,
    out: Option<PathBuf>,
}

fn usage(problem: &str) -> ! {
    if !problem.is_empty() {
        eprintln!("{problem}");
    }
    eprintln!(
        "usage: joint_sweep [--joint hip_yaw] [--amp 15] [--freqs 0.5,2] [--cycles 4] [--kp K] [--kd D] \
         [--config /etc/robot/robotd.toml] [--out sweep.json]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        config: PathBuf::from(robotd_params::DEFAULT_PATH),
        joint: "hip_yaw".into(),
        amp: 15.0,
        freqs: vec![0.5, 2.0],
        cycles: 4.0,
        kp: None,
        kd: None,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .unwrap_or_else(|| usage(&format!("{name} needs a value")))
        };
        let number = |name: &str, v: String| -> f64 {
            v.parse().unwrap_or_else(|_| usage(&format!("bad {name}")))
        };
        match flag.as_str() {
            "--config" => a.config = PathBuf::from(value("--config")),
            "--joint" => a.joint = value("--joint"),
            "--amp" => a.amp = number("--amp", value("--amp")),
            "--freqs" => {
                a.freqs = value("--freqs")
                    .split(',')
                    .map(|f| number("--freqs", f.to_string()))
                    .collect()
            }
            "--cycles" => a.cycles = number("--cycles", value("--cycles")),
            "--kp" => a.kp = Some(number("--kp", value("--kp"))),
            "--kd" => a.kd = Some(number("--kd", value("--kd"))),
            "--out" => a.out = Some(PathBuf::from(value("--out"))),
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    if !(1.0..=30.0).contains(&a.amp) {
        usage("--amp must be 1 to 30 degrees");
    }
    if a.freqs.is_empty() || a.freqs.iter().any(|f| !(0.1..=5.0).contains(f)) {
        usage("--freqs must be 0.1 to 5 Hz");
    }
    if !(1.0..=20.0).contains(&a.cycles) {
        usage("--cycles must be 1 to 20");
    }
    a
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

/// One stretch of the run: a label, its length, and which joint swings at what frequency.
struct Segment {
    label: String,
    seconds: f64,
    swing: Option<(usize, f64, f64)>, // joint, sign, frequency
}

fn main() {
    let args = parse_args();
    let params = robotd_params::Params::load(&args.config, true)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", args.config.display())));
    let bus = &params.bus;
    let Some(zero) = bus.bridge_zero.as_ref() else {
        fail("no [bus.bridge_zero] in the config: joint angles would mean nothing");
    };
    let kp = args.kp.unwrap_or(POLICY_GAIN * bus.bridge_kp_per_gain);
    let kd = args.kd.unwrap_or(bus.bridge_kd);
    if !(0.2..=8.0).contains(&kp) || !(0.0..=0.2).contains(&kd) {
        fail(&format!("kp {kp} or kd {kd} out of range"));
    }
    let config = BridgeConfig {
        kp_per_gain: kp / f64::from(GAIN),
        kd,
        zero_counts: Some(zero.counts()),
        imu_mount: bus
            .bridge_imu_mount
            .map_or(duck_control::imu::SflpDecoder::DUKKI_MOUNT, |m| m.wxyz()),
        imu_median: bus.bridge_imu_median,
        mouth_kp: bus.bridge_mouth_kp,
    };
    let left = joint_index(&format!("left_{}", args.joint))
        .unwrap_or_else(|| usage(&format!("no joint left_{}", args.joint)));
    let right = joint_index(&format!("right_{}", args.joint))
        .unwrap_or_else(|| usage(&format!("no joint right_{}", args.joint)));

    let mut io = BridgeIo::open(&bus.port, bus.bridge_baud, config).unwrap_or_else(|e| {
        fail(&format!(
            "cannot open the bridge on {} at {}: {e}\nis robotd running? stop it first: systemctl stop robotd",
            bus.port, bus.bridge_baud
        ))
    });
    println!(
        "bridge: {} protocol {}; {} left then right, ±{}° at {:?} Hz, {} cycles, kp {kp} kd {kd}",
        io.info().firmware_str(),
        io.info().protocol_version,
        args.joint,
        args.amp,
        args.freqs,
        args.cycles
    );

    // Joint angles from a sample, and the servos' raw positions from a state frame at the same
    // moment: their difference is each joint's offset, used to read the state frames as angles.
    let mut first = io.read();
    for _ in 0..5 {
        if first.is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
        first = io.read();
    }
    let first = first.unwrap_or_else(|e| fail(&format!("no sample from the bridge: {e}")));
    let raw = io
        .state()
        .unwrap_or_else(|e| fail(&format!("no state from the bridge: {e}")));
    let offset: [f64; NUM_JOINTS] =
        std::array::from_fn(|j| raw.servos[j].position as f64 - first.positions[j]);
    let start_pos = first.positions;
    let mouth = joint_index("mouth").expect("mouth");
    let mut home = DEFAULT_POSITION;
    home[mouth] = start_pos[mouth];

    io.write(&JointTargets::new(start_pos))
        .unwrap_or_else(|e| fail(&format!("first command not sent: {e}")));
    let mut tries = 0;
    loop {
        std::thread::sleep(Duration::from_millis(20));
        match io.set_gain(GAIN).and_then(|()| {
            std::thread::sleep(Duration::from_millis(20));
            io.set_torque(true)
        }) {
            Ok(()) => break,
            Err(e) if tries < 5 => {
                tries += 1;
                eprintln!("torque on, try {tries}: {e}");
            }
            Err(e) => fail(&format!("cannot turn torque on: {e}")),
        }
    }

    let mut plan = vec![Segment {
        label: "home".into(),
        seconds: HOME_RAMP_S + HOLD_S,
        swing: None,
    }];
    for &f in &args.freqs {
        for (side, j, sign) in [("left", left, 1.0), ("right", right, -1.0)] {
            plan.push(Segment {
                label: format!("{side} {f} Hz"),
                seconds: (args.cycles + 2.0) / f,
                swing: Some((j, sign, f)),
            });
            plan.push(Segment {
                label: "hold".into(),
                seconds: HOLD_S,
                swing: None,
            });
        }
    }
    let total: f64 = plan.iter().map(|s| s.seconds).sum();
    println!(
        "ramping to the home pose: hold the robot up by the trunk, legs hanging; {total:.0} s in all"
    );

    let amp = args.amp.to_radians();
    let mut frames = Vec::new();
    let t0 = Instant::now();
    let mut next = Instant::now();
    let mut seg_start = 0.0;
    let mut seg = 0;
    let mut misses = 0u32;
    let mut announced = usize::MAX;
    while seg < plan.len() {
        let t = t0.elapsed().as_secs_f64();
        if t - seg_start >= plan[seg].seconds {
            seg_start += plan[seg].seconds;
            seg += 1;
            continue;
        }
        if announced != seg {
            announced = seg;
            println!("{t:6.1} s  {}", plan[seg].label);
        }
        let s = t - seg_start;
        let mut targets = home;
        if seg == 0 {
            let k = (t / HOME_RAMP_S).min(1.0);
            for j in 0..NUM_JOINTS {
                targets[j] = start_pos[j] + (home[j] - start_pos[j]) * k;
            }
        }
        if let Some((j, sign, f)) = plan[seg].swing {
            // A cycle of fade in, the full cycles, a cycle of fade out.
            let n = args.cycles + 2.0;
            let c = s * f;
            let envelope = (c.min(1.0)).min((n - c).max(0.0)).min(1.0);
            targets[j] = home[j] + sign * amp * envelope * (2.0 * std::f64::consts::PI * c).sin();
        }
        if let Err(e) = io.write(&JointTargets::new(targets)) {
            eprintln!("command not sent: {e}");
        }
        match io.state() {
            Ok(st) => {
                misses = 0;
                frames.push(frame(t, &plan[seg].label, &targets, &st, &offset));
            }
            Err(e) => {
                misses += 1;
                eprintln!("state miss {misses}: {e}");
                if misses > 20 {
                    fail("lost the bridge");
                }
            }
        }
        next += PERIOD;
        match next.checked_duration_since(Instant::now()) {
            Some(wait) => std::thread::sleep(wait),
            None => next = Instant::now(),
        }
    }
    if let Err(e) = io.set_torque(false) {
        eprintln!("torque off not sent ({e}); the bridge's watchdog will stop the servos");
    }
    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/sweep_{}.json", args.joint)));
    let record = json!({
        "source": "joint_sweep",
        "joint": args.joint,
        "amp_deg": args.amp,
        "freqs": args.freqs,
        "cycles": args.cycles,
        "kp": kp,
        "kd": kd,
        "frames": frames,
    });
    match std::fs::write(&out, record.to_string()) {
        Ok(()) => println!(
            "wrote {} ({} ticks); torque off",
            out.display(),
            frames.len()
        ),
        Err(e) => fail(&format!("cannot write {}: {e}", out.display())),
    }
}

fn frame(
    t: f64,
    phase: &str,
    targets: &[f64; NUM_JOINTS],
    s: &proto::State,
    offset: &[f64; NUM_JOINTS],
) -> serde_json::Value {
    let pos: Vec<f64> = (0..NUM_JOINTS)
        .map(|j| s.servos[j].position as f64 - offset[j])
        .collect();
    json!({
        "t": (t * 1e3).round() / 1e3,
        "phase": phase,
        "targets": targets,
        "positions": pos,
        "velocities": s.servos.iter().take(NUM_JOINTS).map(|v| v.velocity).collect::<Vec<_>>(),
        "torques": s.servos.iter().take(NUM_JOINTS).map(|v| v.torque).collect::<Vec<_>>(),
        "encoders": s.servos.iter().take(NUM_JOINTS).map(|v| v.output_encoder).collect::<Vec<_>>(),
    })
}
