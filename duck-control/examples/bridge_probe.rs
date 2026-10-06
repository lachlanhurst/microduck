//! Exercise the link to the servo bridge without robotd: info, then state requests and commands
//! at a control rate, with every servo commanded to stop. Nothing moves.
//!
//!   bridge_probe [port] [baud] [hz] [seconds]
//!   bridge_probe /dev/ttyS7 2000000 100 10
//!
//! Prints what the bridge says it is, the round-trip time of each state request, how many ticks
//! got no answer, and the last state: every servo and the IMU.

use std::time::{Duration, Instant};

use duck_bridge_proto as proto;
use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::{JointTargets, RobotIo};

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
    for tick in 0..ticks {
        let t0 = Instant::now();
        match io.state() {
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
        "{ticks} ticks at {hz} Hz: {} answered, {failures} failures; state round trip ms: median {:.2} p95 {:.2} max {:.2}",
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
