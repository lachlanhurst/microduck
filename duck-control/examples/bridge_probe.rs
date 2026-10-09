//! Exercise the link to the servo bridge without robotd: info, then state requests and commands
//! at a control rate, with every servo commanded to stop. Nothing moves.
//!
//!   bridge_probe [port] [baud] [hz] [seconds]
//!   bridge_probe /dev/ttyS7 2000000 100 10
//!
//! Prints what the bridge says it is, the round-trip time of each state request, how many ticks
//! got no answer, and the last state: every servo and the IMU.
//!
//! Also stands in for robotd's side of the shutdown button, without shutting anything down. It
//! sends a `HostStatus` once a second with the hottest thermal zone's temperature. When the
//! bridge raises `SHUTDOWN_REQUESTED` it says so, reports `ShuttingDown` at once, keeps
//! commanding for `SIT_STAND_IN` as robotd would while the robot sits, then stops early the way
//! a powered-off compute module does.

use std::time::{Duration, Instant};

use duck_bridge_proto as proto;
use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::{JointTargets, RobotIo};

/// Stands in for robotd's sit-down before power-off.
const SIT_STAND_IN: Duration = Duration::from_secs(3);
const STATUS_EVERY: Duration = Duration::from_secs(1);

/// The hottest thermal zone, °C, as robotd reports it.
fn hottest_zone_c() -> Option<f64> {
    let zones = std::fs::read_dir("/sys/class/thermal").ok()?;
    zones
        .filter_map(|z| std::fs::read_to_string(z.ok()?.path().join("temp")).ok())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|&mc| mc > 0.0)
        .map(|mc| mc / 1000.0)
        .reduce(f64::max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let port = args.next().unwrap_or_else(|| "/dev/ttyS7".into());
    let baud: u32 = args.next().map_or(2_000_000, |a| a.parse().expect("baud"));
    let hz: f64 = args.next().map_or(50.0, |a| a.parse().expect("hz"));
    let seconds: f64 = args.next().map_or(5.0, |a| a.parse().expect("seconds"));

    let mut io = match BridgeIo::open(&port, baud, BridgeConfig::default()) {
        Ok(io) => io,
        Err(e) => {
            eprintln!("cannot open the bridge on {port} at {baud}: {e}");
            std::process::exit(1);
        }
    };
    let info = *io.info();
    println!(
        "bridge: {} protocol {} sysclk {} MHz link {} baud watchdog {} ms",
        info.firmware_str(),
        info.protocol_version,
        info.sysclk_mhz,
        info.link_baud,
        info.watchdog_ms
    );
    println!(
        "servo segments by id: {:?}",
        info.segments
            .map(|s| if s == proto::ABSENT { -1 } else { s as i32 })
    );

    let period = Duration::from_secs_f64(1.0 / hz);
    let ticks = (seconds * hz).round() as u64;
    let mut round_trips = Vec::with_capacity(ticks as usize);
    let mut failures = 0u64;
    let mut failed_ticks = Vec::new();
    let mut last_err = None;
    let mut next = Instant::now();
    let start = Instant::now();
    let mut head = proto::HeadState::Running;
    let mut status_due = Instant::now();
    let mut shutting_since: Option<Instant> = None;
    let mut ran = 0u64;
    for tick in 0..ticks {
        ran += 1;
        let t0 = Instant::now();
        let state = io.state();
        if state.is_ok() && io.shutdown_requested() && head == proto::HeadState::Running {
            println!(
                "shutdown requested by the bridge at {:.2} s: robotd would sit and power off; reporting shutting down",
                start.elapsed().as_secs_f64()
            );
            head = proto::HeadState::ShuttingDown;
            shutting_since = Some(Instant::now());
            status_due = Instant::now();
        }
        if Instant::now() >= status_due {
            if let Err(e) = io.send_host_status(head, hottest_zone_c()) {
                eprintln!("host status not sent: {e}");
            }
            status_due += STATUS_EVERY;
        }
        if shutting_since.is_some_and(|t| t.elapsed() >= SIT_STAND_IN) {
            println!(
                "stopping at {:.2} s, as a powered-off compute module would",
                start.elapsed().as_secs_f64()
            );
            break;
        }
        match state {
            Ok(_) => round_trips.push(t0.elapsed()),
            Err(e) => {
                failures += 1;
                failed_ticks.push((tick, start.elapsed().as_secs_f64()));
                last_err = Some(e.to_string());
            }
        }
        // Torque is off, so this commands every servo to stop; it feeds the bridge's watchdog
        // and exercises the command path at the control rate.
        if let Err(e) = io.write(&JointTargets::new([0.0; duck_control::NUM_JOINTS])) {
            failures += 1;
            last_err = Some(e.to_string());
        }
        next += period;
        if let Some(wait) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
    }

    round_trips.sort();
    let pct = |p: f64| {
        round_trips
            .get(((round_trips.len() as f64 - 1.0) * p).round() as usize)
            .map_or(0.0, |d| d.as_secs_f64() * 1e3)
    };
    println!(
        "{ran} ticks at {hz} Hz: {} answered, {failures} failures; state round trip ms: median {:.2} p95 {:.2} max {:.2}",
        round_trips.len(),
        pct(0.5),
        pct(0.95),
        pct(1.0)
    );
    if let Some(e) = last_err {
        println!("last error: {e}");
    }
    if !failed_ticks.is_empty() {
        let shown: Vec<String> = failed_ticks
            .iter()
            .take(20)
            .map(|(t, s)| format!("{t}@{s:.2}s"))
            .collect();
        println!("failed ticks: {}", shown.join(" "));
    }

    let Some(s) = io.last_state().copied() else {
        return;
    };
    println!(
        "last state: seq {} command_seq {} flags {:#06x} bridge time {:.3} s, command age {} us, link errors {}, rounds {} ({} us each)",
        s.seq,
        s.command_seq,
        s.flags,
        s.bridge_time_us as f64 * 1e-6,
        s.command_age_us,
        s.link_errors,
        s.rounds,
        s.round_us
    );
    for (id, v) in s.servos.iter().enumerate() {
        if v.segment == proto::ABSENT {
            println!("  id {id:2}: absent");
            continue;
        }
        println!(
            "  id {id:2} seg {} pos {:+8.3} vel {:+6.2} spd {:+6.2} tq {:+6.3} {:4.1} V {}/{} C mode {} flags {:#04x} err {:#x} age {} us ok {} fail {}",
            ["A", "B", "C", "D"][v.segment as usize % 4],
            v.position,
            v.velocity,
            v.speed,
            v.torque,
            v.volts_half as f64 / 2.0,
            v.temp_c,
            v.winding_c,
            v.mode,
            v.flags,
            v.error,
            v.age_us,
            v.replies,
            v.failures
        );
    }
    let i = &s.imu;
    println!(
        "  imu status {} age {} us gravity mg {:?} gbias mdps {:?} samples {}/{} fifo overruns {} int1 timeouts {}",
        i.status,
        i.age_us,
        i.gravity_mg,
        i.gbias_mdps,
        i.gyro_samples,
        i.rotation_samples,
        i.fifo_overruns,
        i.int1_timeouts
    );
}
