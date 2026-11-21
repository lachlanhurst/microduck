//! Every servo limp, and what each one reports streamed to a laptop: for checking servo IDs
//! and directions against the model, and for finding each joint's range by hand. Runs without
//! robotd, which must be stopped first since it owns the bridge link.
//!
//!   joint_scope [--port /dev/ttyS7] [--baud 2000000] [--listen 0.0.0.0:7788] [--hz 50]
//!               [--kd 0.0] [--brake]
//!
//! Limp is J288 FOC with kp, kd and torque all zero, so a joint moves by hand against nothing
//! but its gearbox. `--kd` adds a little damping; `--brake` leaves every servo in `Stop`, the
//! shorted-winding brake, for streaming without letting anything move freely.
//!
//! Each tick is one full state request, so the stream carries the output-shaft encoder as well
//! as the multi-turn position, and one command repeating where every servo is, which feeds the
//! bridge's watchdog. When this program stops, for whatever reason, the watchdog puts every
//! servo back to `Stop` within its period: the robot goes stiff, it does not drop.
//!
//! Any number of TCP clients may connect to `--listen`. Each gets one JSON object per tick,
//! newline-terminated:
//!
//! ```text
//! {"t":12.34,"limp":true,"flags":2,"pack_mv":24100,"servos":[
//!   {"id":0,"seg":0,"mode":1,"pos":0.123,"vel":0.0,"tq":0.001,"enc":4096,"temp":31,"err":0,"age":420},
//!   {"id":1,"seg":null}, ...]}
//! ```
//!
//! `pos` is the servo's multi-turn position relative to where it powered up, rad, output side;
//! `enc` is the raw 13-bit output encoder (8192 counts a turn). `scripts/joint_scope` in the
//! Dukki repository is the client.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use duck_bridge_proto as proto;
use duck_control::bridge::{BridgeConfig, BridgeIo};
use duck_control::{JointTargets, NUM_JOINTS, RobotIo};
use serde_json::json;

const STATUS_EVERY: Duration = Duration::from_secs(1);
/// A client that cannot take a line this quickly is dropped rather than allowed to stall the
/// loop past the bridge's watchdog.
const CLIENT_WRITE_TIMEOUT: Duration = Duration::from_millis(5);

struct Args {
    port: String,
    baud: u32,
    listen: String,
    hz: f64,
    kd: f64,
    brake: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        port: "/dev/ttyS7".into(),
        baud: 2_000_000,
        listen: "0.0.0.0:7788".into(),
        hz: 50.0,
        kd: 0.0,
        brake: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .unwrap_or_else(|| usage(&format!("{name} needs a value")))
        };
        match flag.as_str() {
            "--port" => a.port = value("--port"),
            "--baud" => {
                a.baud = value("--baud")
                    .parse()
                    .unwrap_or_else(|_| usage("bad --baud"))
            }
            "--listen" => a.listen = value("--listen"),
            "--hz" => a.hz = value("--hz").parse().unwrap_or_else(|_| usage("bad --hz")),
            "--kd" => a.kd = value("--kd").parse().unwrap_or_else(|_| usage("bad --kd")),
            "--brake" => a.brake = true,
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    if !(1.0..=100.0).contains(&a.hz) {
        usage("--hz must be 1 to 100");
    }
    if !(0.0..=0.2).contains(&a.kd) {
        usage("--kd must be 0 to 0.2 N·m·s/rad");
    }
    a
}

fn usage(problem: &str) -> ! {
    if !problem.is_empty() {
        eprintln!("{problem}");
    }
    eprintln!(
        "usage: joint_scope [--port /dev/ttyS7] [--baud 2000000] [--listen 0.0.0.0:7788] [--hz 50] [--kd 0.0] [--brake]"
    );
    std::process::exit(2);
}

fn hottest_zone_c() -> Option<f64> {
    let zones = std::fs::read_dir("/sys/class/thermal").ok()?;
    zones
        .filter_map(|z| std::fs::read_to_string(z.ok()?.path().join("temp")).ok())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|&mc| mc > 0.0)
        .map(|mc| mc / 1000.0)
        .reduce(f64::max)
}

fn line(t: f64, limp: bool, s: &proto::State) -> String {
    let servos: Vec<_> = s
        .servos
        .iter()
        .enumerate()
        .map(|(id, v)| {
            if v.segment == proto::ABSENT {
                json!({ "id": id, "seg": null })
            } else {
                json!({
                    "id": id,
                    "seg": v.segment,
                    "mode": v.mode,
                    "pos": v.position,
                    "vel": v.velocity,
                    "tq": v.torque,
                    "enc": v.output_encoder,
                    "temp": v.temp_c,
                    "err": v.error,
                    "age": v.age_us,
                })
            }
        })
        .collect();
    let mut out = json!({
        "t": (t * 1e3).round() / 1e3,
        "limp": limp,
        "flags": s.flags,
        "pack_mv": s.pack_mv,
        "servos": servos,
    })
    .to_string();
    out.push('\n');
    out
}

fn main() {
    let args = parse_args();
    let config = BridgeConfig {
        kp_per_gain: 0.0,
        kd: args.kd,
        // The servos' own frame: the stream carries raw positions and encoder counts, and the
        // calibration it feeds is what produces the zeros.
        zero_counts: None,
    };
    let mut io = match BridgeIo::open(&args.port, args.baud, config) {
        Ok(io) => io,
        Err(e) => {
            eprintln!(
                "cannot open the bridge on {} at {}: {e}\nis robotd running? stop it first: sudo systemctl stop robotd",
                args.port, args.baud
            );
            std::process::exit(1);
        }
    };
    let info = *io.info();
    println!(
        "bridge: {} protocol {} watchdog {} ms; servo segments by id: {:?}",
        info.firmware_str(),
        info.protocol_version,
        info.watchdog_ms,
        info.segments
            .map(|s| if s == proto::ABSENT { -1 } else { s as i32 })
    );

    let listener = TcpListener::bind(&args.listen).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}: {e}", args.listen);
        std::process::exit(1);
    });
    listener
        .set_nonblocking(true)
        .expect("non-blocking listener");
    println!("streaming on {}", args.listen);

    // Targets first, at where every servo is, so torque goes on without a fresh sample (which
    // needs all fifteen servos present) and the command never carries position zero. With kp
    // and torque zero the targets do nothing; they are there so that nothing changes if they
    // ever do.
    let first = io.state().unwrap_or_else(|e| {
        eprintln!("no state from the bridge: {e}");
        std::process::exit(1);
    });
    let here = |s: &proto::State| -> [f64; NUM_JOINTS] {
        std::array::from_fn(|j| s.servos[j].position as f64)
    };
    let mut targets = here(&first);
    if let Err(e) = io.write(&JointTargets::new(targets)) {
        eprintln!("first command not sent: {e}");
        std::process::exit(1);
    }
    let limp = !args.brake;
    if limp {
        // kp starts at zero in `BridgeIo`, so torque on is all limp needs. The port's short
        // timeout covers writes too, and full commands sent back to back can outrun it: give
        // the first one time to leave, and retry.
        let mut tries = 0;
        loop {
            std::thread::sleep(Duration::from_millis(20));
            match io.set_torque(true) {
                Ok(()) => break,
                Err(e) if tries < 5 => {
                    tries += 1;
                    eprintln!("going limp, try {tries}: {e}");
                }
                Err(e) => {
                    eprintln!("cannot go limp: {e}");
                    std::process::exit(1);
                }
            }
        }
        println!(
            "every servo limp (FOC, kp 0, kd {}); move the joints by hand",
            args.kd
        );
    } else {
        println!("every servo braked (Stop)");
    }

    let mut clients: Vec<(TcpStream, String)> = Vec::new();
    let period = Duration::from_secs_f64(1.0 / args.hz);
    let start = Instant::now();
    let mut next = Instant::now();
    let mut status_due = Instant::now();
    let mut misses = 0u64;
    let mut warned_shutdown = false;
    loop {
        while let Ok((stream, peer)) = listener.accept() {
            let _ = stream.set_nodelay(true);
            let _ = stream.set_nonblocking(false);
            let _ = stream.set_write_timeout(Some(CLIENT_WRITE_TIMEOUT));
            println!("client {peer} connected");
            clients.push((stream, peer.to_string()));
        }

        match io.state() {
            Ok(state) => {
                for (j, v) in state.servos.iter().enumerate() {
                    if v.segment != proto::ABSENT {
                        targets[j] = v.position as f64;
                    }
                }
                let text = line(start.elapsed().as_secs_f64(), limp, &state);
                clients.retain_mut(|(stream, peer)| match stream.write_all(text.as_bytes()) {
                    Ok(()) => true,
                    Err(e) => {
                        println!("client {peer} dropped: {e}");
                        false
                    }
                });
                if io.shutdown_requested() && !warned_shutdown {
                    warned_shutdown = true;
                    println!(
                        "shutdown button held: joint_scope does not shut the head down; stop it and use robotd or poweroff"
                    );
                }
            }
            Err(e) => {
                misses += 1;
                if misses < 10 || misses.is_multiple_of(100) {
                    eprintln!("state miss {misses}: {e}");
                }
            }
        }
        if let Err(e) = io.write(&JointTargets::new(targets)) {
            eprintln!("command not sent: {e}");
        }
        if Instant::now() >= status_due {
            let _ = io.send_host_status(proto::HeadState::Running, hottest_zone_c());
            status_due += STATUS_EVERY;
        }

        next += period;
        match next.checked_duration_since(Instant::now()) {
            Some(wait) => std::thread::sleep(wait),
            None => next = Instant::now(),
        }
    }
}
