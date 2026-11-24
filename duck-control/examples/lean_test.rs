//! The ankle-lean test: where is the robot's centre of mass over its feet, against the model?
//! Runs without robotd, which must be stopped first since it owns the bridge link.
//!
//!   lean_test --dir back|forward [--config /etc/robot/robotd.toml] [--kp 4.0] [--kd 0.05]
//!             [--rate 1.0] [--settle 2.0] [--max-deg 25] [--out lean.json]
//!
//! No policy runs. Every joint ramps to the home pose ([`DEFAULT_POSITION`]) at `--kp` and holds
//! it; someone stands the robot on a hard, flat table and presses Enter. After `--settle` seconds
//! both ankles turn together at `--rate` degrees a second, so everything above them leans over
//! the flat feet, until the robot tips over the heel (`back`) or toe (`forward`) edge of its soles.
//! The tip is the trunk pitching faster than 1 rad/s for three ticks; the ramp stops there and
//! the targets hold, so **catch it**. After 1.5 s more the recording is written and torque goes
//! off (every J288 to `Stop`, braked, as when any program on the bridge stops).
//!
//! The kp is stiffer than the policy's 1 N·m/rad on purpose: at that gain the legs give under
//! the robot's weight and, in simulation, it buckles before its centre of mass reaches the edge
//! of its soles, by an amount that depends on joint friction. At 4 N·m/rad (the stiffest gain
//! the J288 model was validated at) it tips at the edge. The joints, the zeros and the IMU mount
//! are robotd's (`[bus]` of its config, both layers), so the angles are the policy's.
//!
//! Positive lean is forwards: `left_ankle` goes down by the lean and `right_ankle` up. The
//! recording is JSON, every 10 ms tick:
//!
//! ```text
//! {"source":"lean_test","kp":4.0,"kd":0.05,"rate_deg_s":1.0,"direction":"back","home":[...15],
//!  "frames":[{"t":0.01,"phase":"home","delta_deg":0.0,"targets":[...15],"positions":[...15],
//!             "velocities":[...15],"currents_ma":[...15],"gravity":[x,y,z],"gyro":[x,y,z]}, ...]}
//! ```
//!
//! Phases: `home` (ramping to the home pose), `hold` (waiting for Enter), `settle`, `ramp`,
//! `after`. The Dukki repository's `scripts/policy/sim/lean_sim.py --recording` compares the
//! recordings with the model.

use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::model::joint_index;
use duck_control::{DEFAULT_POSITION, JointTargets, NUM_JOINTS, RobotIo, Sensors};
use serde_json::json;

const PERIOD: Duration = Duration::from_millis(10);
const HOME_RAMP_S: f64 = 3.0;
const AFTER_S: f64 = 1.5;
/// Tipping: the trunk's pitch rate past this for `TIP_TICKS` ticks in a row. A 1°/s lean is
/// 0.017 rad/s. On its own soles the robot rolls forward onto its toes at up to 0.3 rad/s and
/// stops there (23/11/2026), which is not the tip; a fall passes 1 rad/s within a few degrees.
/// `lean_sim.py` detects it the same way.
const TIP_RATE: f64 = 1.0;
const TIP_TICKS: u32 = 3;
/// The gain `BridgeIo::set_gain` is given; `kp_per_gain` is set so it comes to `--kp`.
const GAIN: u16 = 100;
/// Each ankle's turn for a positive (forward) lean, in the joint's own sense.
const ANKLE_SIGNS: [(&str, f64); 2] = [("left_ankle", -1.0), ("right_ankle", 1.0)];

struct Args {
    config: PathBuf,
    forward: bool,
    kp: f64,
    kd: f64,
    rate: f64,
    settle: f64,
    max_deg: f64,
    out: Option<PathBuf>,
}

fn usage(problem: &str) -> ! {
    if !problem.is_empty() {
        eprintln!("{problem}");
    }
    eprintln!(
        "usage: lean_test --dir back|forward [--config /etc/robot/robotd.toml] [--kp 4.0] [--kd 0.05] \
         [--rate 1.0] [--settle 2.0] [--max-deg 25] [--out lean.json]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        config: PathBuf::from(robotd_params::DEFAULT_PATH),
        forward: false,
        kp: 4.0,
        kd: 0.05,
        rate: 1.0,
        settle: 2.0,
        max_deg: 25.0,
        out: None,
    };
    let mut dir = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .unwrap_or_else(|| usage(&format!("{name} needs a value")))
        };
        let mut number = |name: &str| -> f64 {
            value(name)
                .parse()
                .unwrap_or_else(|_| usage(&format!("bad {name}")))
        };
        match flag.as_str() {
            "--dir" => dir = Some(value("--dir")),
            "--config" => a.config = PathBuf::from(value("--config")),
            "--kp" => a.kp = number("--kp"),
            "--kd" => a.kd = number("--kd"),
            "--rate" => a.rate = number("--rate"),
            "--settle" => a.settle = number("--settle"),
            "--max-deg" => a.max_deg = number("--max-deg"),
            "--out" => a.out = Some(PathBuf::from(value("--out"))),
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    a.forward = match dir.as_deref() {
        Some("back") => false,
        Some("forward") => true,
        _ => usage("--dir back or --dir forward"),
    };
    if !(0.5..=8.0).contains(&a.kp) {
        usage("--kp must be 0.5 to 8 N·m/rad");
    }
    if !(0.0..=0.2).contains(&a.kd) {
        usage("--kd must be 0 to 0.2 N·m·s/rad");
    }
    if !(0.1..=5.0).contains(&a.rate) {
        usage("--rate must be 0.1 to 5 deg/s");
    }
    if !(1.0..=30.0).contains(&a.max_deg) {
        usage("--max-deg must be 1 to 30");
    }
    a
}

fn joint(name: &str) -> usize {
    joint_index(name).unwrap_or_else(|| panic!("no joint {name}"))
}

/// Trunk lean from the IMU's gravity, deg, forwards positive.
fn lean_deg(g: &[f64; 3]) -> f64 {
    let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-9);
    (g[0] / n).clamp(-1.0, 1.0).asin().to_degrees()
}

fn main() {
    let args = parse_args();
    let params = robotd_params::Params::load(&args.config, true).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", args.config.display());
        std::process::exit(1);
    });
    let bus = &params.bus;
    let Some(zero) = bus.bridge_zero.as_ref() else {
        eprintln!("no [bus.bridge_zero] in the config: joint angles would mean nothing");
        std::process::exit(1);
    };
    let config = BridgeConfig {
        kp_per_gain: args.kp / f64::from(GAIN),
        kd: args.kd,
        zero_counts: Some(zero.counts()),
        imu_mount: bus
            .bridge_imu_mount
            .map_or(duck_control::imu::SflpDecoder::DUKKI_MOUNT, |m| m.wxyz()),
        imu_median: bus.bridge_imu_median,
        mouth_kp: bus.bridge_mouth_kp,
    };
    let mut io = BridgeIo::open(&bus.port, bus.bridge_baud, config).unwrap_or_else(|e| {
        eprintln!(
            "cannot open the bridge on {} at {}: {e}\nis robotd running? stop it first: systemctl stop robotd",
            bus.port, bus.bridge_baud
        );
        std::process::exit(1);
    });
    let info = *io.info();
    println!(
        "bridge: {} protocol {}; kp {} kd {}, lean {} at {} deg/s",
        info.firmware_str(),
        info.protocol_version,
        args.kp,
        args.kd,
        if args.forward {
            "forwards"
        } else {
            "backwards"
        },
        args.rate
    );

    // The first sample after the port opens can miss its window; a few tries.
    let mut first = io.read();
    for _ in 0..5 {
        if first.is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
        first = io.read();
    }
    let first = first.unwrap_or_else(|e| fail(&format!("no sample from the bridge: {e}")));
    let start_pos = first.positions;
    // The mouth is in no pose here: it holds where it is.
    let mouth = joint("mouth");
    let mut home = DEFAULT_POSITION;
    home[mouth] = start_pos[mouth];
    let ankles = ANKLE_SIGNS.map(|(n, s)| (joint(n), s));

    io.write(&JointTargets::new(start_pos))
        .unwrap_or_else(|e| fail(&format!("first command not sent: {e}")));
    // The port's short timeout covers writes too, and full commands back to back can outrun it.
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

    let enter = Arc::new(AtomicBool::new(false));
    {
        let enter = Arc::clone(&enter);
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().lock().read_line(&mut line);
            enter.store(true, Ordering::SeqCst);
        });
    }

    println!("ramping to the home pose over {HOME_RAMP_S} s: hold the robot, feet down");
    let sign = if args.forward { 1.0 } else { -1.0 };
    let mut frames = Vec::new();
    let t0 = Instant::now();
    let mut next = Instant::now();
    let mut phase = "home";
    let mut phase_t = 0.0;
    let mut delta_deg = 0.0;
    let mut fast_ticks = 0;
    let mut tipped = false;
    let mut last_print = -1.0f64;
    let mut misses = 0u32;
    loop {
        let t = t0.elapsed().as_secs_f64();
        let sensors = match io.read() {
            Ok(s) => {
                misses = 0;
                s
            }
            Err(e) => {
                misses += 1;
                eprintln!("sample miss {misses}: {e}");
                if misses > 20 {
                    fail("lost the bridge");
                }
                next += PERIOD;
                sleep_until(&mut next);
                continue;
            }
        };
        let since = t - phase_t;
        let mut targets = home;
        match phase {
            "home" => {
                let f = (t / HOME_RAMP_S).min(1.0);
                for j in 0..NUM_JOINTS {
                    targets[j] = start_pos[j] + (home[j] - start_pos[j]) * f;
                }
                if f >= 1.0 {
                    phase = "hold";
                    phase_t = t;
                    println!(
                        "holding the home pose. Stand it on its feet on the table, let go, and press Enter"
                    );
                }
            }
            "hold" => {
                if enter.load(Ordering::SeqCst) {
                    if !io.imu_ready() {
                        println!("the IMU is not ready yet; waiting");
                    } else {
                        phase = "settle";
                        phase_t = t;
                        println!("settling for {} s", args.settle);
                    }
                }
            }
            "settle" => {
                if since >= args.settle {
                    phase = "ramp";
                    phase_t = t;
                    println!("leaning; be ready to catch it");
                }
            }
            "ramp" => {
                delta_deg = sign * since * args.rate;
                fast_ticks = if sensors.imu.gyro[1].abs() > TIP_RATE {
                    fast_ticks + 1
                } else {
                    0
                };
                if fast_ticks >= TIP_TICKS {
                    tipped = true;
                    phase = "after";
                    phase_t = t;
                    println!(
                        "tipped at {delta_deg:+.2} deg on the ankles, IMU lean {:+.2} deg",
                        lean_deg(&sensors.imu.gravity)
                    );
                } else if delta_deg.abs() >= args.max_deg {
                    phase = "after";
                    phase_t = t;
                    println!("reached {delta_deg:+.1} deg without tipping");
                }
            }
            _ => {
                if since >= AFTER_S {
                    break;
                }
            }
        }
        if matches!(phase, "ramp" | "after") {
            for &(j, s) in &ankles {
                targets[j] = home[j] + (s * delta_deg).to_radians();
            }
        }
        if let Err(e) = io.write(&JointTargets::new(targets)) {
            eprintln!("command not sent: {e}");
        }
        let every = if matches!(phase, "home" | "hold") {
            5.0
        } else {
            1.0
        };
        if t - last_print >= every {
            last_print = t;
            println!(
                "{t:6.1} s {phase:6} lean {delta_deg:+6.2} deg  IMU {:+6.2} deg  ankles {:+6.2} {:+6.2} deg from home",
                lean_deg(&sensors.imu.gravity),
                ankles[0].1 * (sensors.positions[ankles[0].0] - home[ankles[0].0]).to_degrees(),
                ankles[1].1 * (sensors.positions[ankles[1].0] - home[ankles[1].0]).to_degrees(),
            );
        }
        frames.push(frame(t, phase, delta_deg, &targets, &sensors));
        next += PERIOD;
        sleep_until(&mut next);
    }
    if let Err(e) = io.set_torque(false) {
        eprintln!("torque off not sent ({e}); the bridge's watchdog will stop the servos");
    }

    let out = args.out.unwrap_or_else(|| {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        PathBuf::from(format!(
            "/tmp/lean_{}_{stamp}.json",
            if args.forward { "forward" } else { "back" }
        ))
    });
    let record = json!({
        "source": "lean_test",
        "kp": args.kp,
        "kd": args.kd,
        "rate_deg_s": args.rate,
        "direction": if args.forward { "forward" } else { "back" },
        "tipped": tipped,
        "home": home,
        "frames": frames,
    });
    match std::fs::write(&out, record.to_string()) {
        Ok(()) => println!(
            "wrote {} ({} ticks); torque off",
            out.display(),
            record["frames"].as_array().map_or(0, Vec::len)
        ),
        Err(e) => fail(&format!("cannot write {}: {e}", out.display())),
    }
}

fn frame(
    t: f64,
    phase: &str,
    delta_deg: f64,
    targets: &[f64; NUM_JOINTS],
    s: &Sensors,
) -> serde_json::Value {
    json!({
        "t": (t * 1e3).round() / 1e3,
        "phase": phase,
        "delta_deg": delta_deg,
        "targets": targets,
        "positions": s.positions,
        "velocities": s.velocities,
        "currents_ma": s.currents_ma,
        "gravity": s.imu.gravity,
        "gyro": s.imu.gyro,
    })
}

fn sleep_until(next: &mut Instant) {
    match next.checked_duration_since(Instant::now()) {
        Some(wait) => std::thread::sleep(wait),
        None => *next = Instant::now(),
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
