//! `expressd`: owns the robot's expressive outputs (the eye LED and the fan) and decides what
//! they show.
//!
//! Four tasks share the eye's and the fan's [`Stack`]s (see `lib.rs` for the layers):
//!
//! - **The follower** subscribes to robotd's `robot.state` at 10 Hz and writes the eye's
//!   `fault` and `ambient` layers. No robotd, or no frame for two seconds, is the `asleep` look:
//!   an eye that froze on "walking" when the brain died would be worse than no eye.
//! - **The thermometer** reads the hottest thermal zone every two seconds, for the fan's floor.
//! - **The server** answers `express.set`, `express.clear`, `express.identify` and
//!   `express.status` on `/run/expressd/express.sock`, one JSON-RPC request per line, as many as
//!   a client likes.
//! - **The renderer** drives both outputs at ~30 Hz while either animates, and otherwise only on
//!   a change (plus a refresh every second, so a glitched frame does not stick).
//!
//! The methods are this daemon's own and live here rather than in `duck-ipc-proto`: expression
//! is Dukki's addition, and keeping it out of the shared protocol keeps the upstream diff at
//! zero.
//!
//! **The fan fails on.** When the daemon stops it leaves the fan at full speed, so an expressd
//! that has been stopped or has crashed cannot leave the compute module without cooling, and
//! systemd restarts it. Between the PWM driver claiming the pin at boot and expressd first
//! running, the fan is not driven; that is seconds, and the SoC's own throttling covers it.
//!
//! `expressd set mood --eye 0,255,0 --pattern breathe --fan 0.6` and friends are a client for the
//! same socket, so a shell on the robot can drive both without writing JSON.

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use duck_ipc_proto as proto;
use express::{FanCurve, FanDrive, FanLook, Layer, Look, Order, Pattern, Rgb, Stack};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;

const SOCKET: &str = "/run/expressd/express.sock";
const SOCKET_MODE: u32 = 0o660;
/// Whoever may watch the robot may set its mood, as with tofd's and robotd's sockets.
const GROUP: &str = "robot";

const SET: &str = "express.set";
const CLEAR: &str = "express.clear";
const IDENTIFY: &str = "express.identify";
const STATUS: &str = "express.status";

/// robot.state rate asked for. The eye changes at human speed; 10 Hz is already generous.
const STATE_HZ: u32 = 10;
/// No frame for this long means robotd has stopped talking, whatever the socket says.
const STALE: Duration = Duration::from_secs(2);
const RETRY: Duration = Duration::from_secs(2);
/// Frame interval while animating. Smooth to the eye, and ~30 small writes a second.
const FRAME: Duration = Duration::from_millis(33);
/// Rewrite an unchanged output this often, so one corrupted frame cannot stick.
const REFRESH: Duration = Duration::from_secs(1);
/// How often the thermometer reads. The SoC's temperature moves over seconds, not frames.
const THERMAL_EVERY: Duration = Duration::from_secs(2);
/// A start from rest runs the fan at full for this long, so it spins up before it is slowed.
const FAN_KICK_S: f32 = 0.5;

#[derive(Parser)]
#[command(about = "The expression daemon (eye LED and fan), and a client for it")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// spidev node the LED's data line hangs off (setup-express.sh names it).
    #[arg(long, default_value = "/dev/spidev-eye")]
    device: PathBuf,
    /// Brightness cap, 0 to 1. A bare 5050 LED at full scale is painful to look at.
    #[arg(long, default_value_t = 0.15)]
    brightness: f32,
    /// Colour order on the wire.
    #[arg(long, value_enum, default_value_t = Order::Grb)]
    order: Order,
    /// Log instead of driving the LED and the fan: for a laptop, or a board without them.
    #[arg(long)]
    fake: bool,
    /// The fan's PWM controller in sysfs: the directory holding its pwmchipN.
    #[arg(long, default_value = "/sys/devices/platform/2add2000.pwm/pwm")]
    fan_pwm: PathBuf,
    /// Channel on that controller.
    #[arg(long, default_value_t = 0)]
    fan_channel: u32,
    /// PWM frequency. 25 kHz is above hearing; some 2-wire fans run better slow, around 100 Hz.
    #[arg(long, default_value_t = 25_000)]
    fan_hz: u32,
    /// Lowest running duty, 0 to 1. Below it a 2-wire fan stalls.
    #[arg(long, default_value_t = 0.3)]
    fan_min: f32,
    /// SoC temperature (°C) the fan starts at, at its minimum speed.
    #[arg(long, default_value_t = 60.0)]
    fan_on_c: f32,
    /// SoC temperature (°C) it stops below, once running.
    #[arg(long, default_value_t = 50.0)]
    fan_off_c: f32,
    /// SoC temperature (°C) for full speed.
    #[arg(long, default_value_t = 75.0)]
    fan_full_c: f32,
    /// No fan fitted: do not touch the PWM.
    #[arg(long)]
    no_fan: bool,
    /// This daemon's socket (the client subcommands use it too).
    #[arg(long, default_value = SOCKET)]
    socket: PathBuf,
    /// robotd's socket.
    #[arg(long, default_value = proto::socket::ROBOT)]
    robot: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Set a layer (privacy, identify or mood): the eye, the fan, or both.
    Set {
        #[arg(value_parser = parse_layer)]
        layer: Layer,
        /// Eye colour, R,G,B with each 0 to 255.
        #[arg(long, value_parser = parse_rgb)]
        eye: Option<Rgb>,
        #[arg(long, value_parser = parse_pattern, default_value = "solid")]
        pattern: Pattern,
        /// Seconds per cycle of the eye's pattern.
        #[arg(long, default_value_t = 2.0)]
        period: f32,
        /// Fan speed, 0 to 1. Never below what cooling needs.
        #[arg(long)]
        fan: Option<f32>,
        #[arg(long, value_parser = parse_pattern, default_value = "solid")]
        fan_pattern: Pattern,
        /// Seconds per cycle of the fan's pattern.
        #[arg(long, default_value_t = 4.0)]
        fan_period: f32,
        /// Clear the layer after this many seconds.
        #[arg(long)]
        ttl: Option<f32>,
    },
    /// Clear a layer for both outputs, uncovering the one beneath.
    Clear {
        #[arg(value_parser = parse_layer)]
        layer: Layer,
    },
    /// Blink the eye white for a few seconds, to find this robot among others.
    Identify {
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
    },
    /// What the eye and the fan are doing, and why.
    Status,
}

fn parse_layer(s: &str) -> Result<Layer, String> {
    serde_json::from_value(serde_json::Value::String(s.to_owned()))
        .map_err(|_| "one of privacy, fault, identify, mood, ambient".to_owned())
}

fn parse_pattern(s: &str) -> Result<Pattern, String> {
    serde_json::from_value(serde_json::Value::String(s.to_owned()))
        .map_err(|_| "one of solid, breathe, blink, pulse".to_owned())
}

fn parse_rgb(s: &str) -> Result<Rgb, String> {
    let v: Vec<u8> = s
        .split(',')
        .map(|c| c.trim().parse())
        .collect::<Result<_, _>>()
        .map_err(|_| "R,G,B with each 0 to 255".to_owned())?;
    match v[..] {
        [r, g, b] => Ok(Rgb(r, g, b)),
        _ => Err("R,G,B with each 0 to 255".to_owned()),
    }
}

// ── Wire shapes for this daemon's methods ────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct SetParams {
    layer: Layer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eye: Option<Look>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fan: Option<FanLook>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ttl_s: Option<f32>,
}

#[derive(Serialize, Deserialize)]
struct ClearParams {
    layer: Layer,
}

#[derive(Serialize, Deserialize)]
struct IdentifyParams {
    #[serde(default = "identify_seconds")]
    seconds: f32,
}

fn identify_seconds() -> f32 {
    10.0
}

#[derive(Serialize)]
struct LayerLook<T> {
    layer: Layer,
    #[serde(flatten)]
    look: T,
}

#[derive(Serialize)]
struct EyeStatus {
    /// The layer being shown, if any.
    showing: Option<Layer>,
    /// The colour on the LED this instant, before brightness.
    colour: Rgb,
    layers: Vec<LayerLook<Look>>,
    brightness: f32,
}

#[derive(Serialize)]
struct FanStatus {
    /// Whether a fan output is open. False with `--no-fan`, or when the PWM could not be opened.
    available: bool,
    /// The duty being driven this instant, 0 to 1.
    duty: f32,
    /// The cooling floor, 0 to 1.
    floor: f32,
    /// The hottest thermal zone, °C. Absent when none could be read, which runs the fan full.
    temp_c: Option<f32>,
    /// The layer asking for more than the floor, if any.
    showing: Option<Layer>,
    layers: Vec<LayerLook<FanLook>>,
}

#[derive(Serialize)]
struct StatusResult {
    /// Whether robot.state is arriving.
    robotd: bool,
    eye: EyeStatus,
    fan: FanStatus,
}

// ── Shared state ─────────────────────────────────────────────────────────────

struct Shared {
    start: Instant,
    eye: Mutex<Stack<Look>>,
    fan: Mutex<Stack<FanLook>>,
    robotd: Mutex<bool>,
    temp_c: Mutex<Option<f32>>,
    /// What the renderer last drove, for `express.status`: (duty, floor).
    fan_out: Mutex<(f32, f32)>,
    fan_available: bool,
    changed: Notify,
}

impl Shared {
    fn now(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    fn changed(&self) {
        self.changed.notify_one();
    }
}

// ── Outputs ──────────────────────────────────────────────────────────────────

enum Eye {
    Spi(File),
    Fake,
}

impl Eye {
    fn open(device: &Path, fake: bool) -> Result<Self> {
        if fake {
            return Ok(Eye::Fake);
        }
        let file = File::options().write(true).open(device).with_context(|| {
            format!(
                "opening {} (run setup-express.sh, or --fake)",
                device.display()
            )
        })?;
        // spidev ioctls: _IOW('k', 1, u8) and _IOW('k', 4, u32).
        const SPI_IOC_WR_MODE: u64 = 0x4001_6b01;
        const SPI_IOC_WR_MAX_SPEED_HZ: u64 = 0x4004_6b04;
        let mode: u8 = 0;
        let hz: u32 = express::SPI_HZ;
        // SAFETY: valid fd for the life of `file`; each ioctl reads one value of the size its
        // request number encodes, from a live local.
        unsafe {
            if libc::ioctl(file.as_raw_fd(), SPI_IOC_WR_MODE as _, &mode) != 0
                || libc::ioctl(file.as_raw_fd(), SPI_IOC_WR_MAX_SPEED_HZ as _, &hz) != 0
            {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("configuring {}", device.display()));
            }
        }
        Ok(Eye::Spi(file))
    }

    /// One frame. A single `write` is a single SPI transfer, which is what keeps the timing.
    fn show(&mut self, rgb: Rgb, brightness: f32, order: Order) -> std::io::Result<()> {
        match self {
            Eye::Spi(file) => file.write_all(&express::encode(rgb, brightness, order)),
            Eye::Fake => {
                tracing::debug!(r = rgb.0, g = rgb.1, b = rgb.2, "eye");
                Ok(())
            }
        }
    }
}

/// The fan's PWM channel, through sysfs. Writes `duty_cycle` in nanoseconds of `period`.
enum Fan {
    Pwm { channel: PathBuf, period_ns: u64 },
    Fake,
    None,
}

impl Fan {
    fn open(args: &Args) -> Result<Self> {
        if args.no_fan {
            return Ok(Fan::None);
        }
        if args.fake {
            return Ok(Fan::Fake);
        }
        let chip = std::fs::read_dir(&args.fan_pwm)
            .with_context(|| format!("reading {} (run setup-express.sh)", args.fan_pwm.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("pwmchip"))
            })
            .with_context(|| format!("no pwmchip under {}", args.fan_pwm.display()))?;
        let channel = chip.join(format!("pwm{}", args.fan_channel));
        if !channel.exists() {
            std::fs::write(chip.join("export"), args.fan_channel.to_string()).with_context(
                || {
                    format!(
                        "exporting channel {} of {}",
                        args.fan_channel,
                        chip.display()
                    )
                },
            )?;
        }
        let period_ns = 1_000_000_000 / u64::from(args.fan_hz.max(1));
        // The channel's files appear root-only and the udev rule regroups them a moment later:
        // retry briefly rather than fail on the race.
        let mut tries = 0;
        loop {
            match Self::configure(&channel, period_ns) {
                Ok(()) => break,
                Err(e) if tries < 30 => {
                    tries += 1;
                    tracing::debug!(error = %e, "fan PWM not writable yet");
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(e).context(format!("configuring {}", channel.display())),
            }
        }
        Ok(Fan::Pwm { channel, period_ns })
    }

    fn configure(channel: &Path, period_ns: u64) -> std::io::Result<()> {
        let write = |name: &str, value: &str| std::fs::write(channel.join(name), value);
        // A fresh channel has a period of 0, and the kernel refuses every write while it is,
        // even `enable`: give it a period before anything else.
        let current: u64 = std::fs::read_to_string(channel.join("period"))?
            .trim()
            .parse()
            .unwrap_or(0);
        if current == 0 {
            write("period", &period_ns.to_string())?;
        }
        // Polarity can only change while the channel is disabled, and the RK3576 vendor driver
        // starts it inversed, which would turn every duty into its complement.
        write("enable", "0")?;
        write("polarity", "normal")?;
        // The duty may not exceed the period, so it goes to zero before the period changes.
        write("duty_cycle", "0")?;
        write("period", &period_ns.to_string())?;
        write("enable", "1")
    }

    fn set(&mut self, duty: f32) -> std::io::Result<()> {
        match self {
            Fan::Pwm { channel, period_ns } => {
                let ns = (f64::from(duty.clamp(0.0, 1.0)) * *period_ns as f64).round() as u64;
                std::fs::write(channel.join("duty_cycle"), ns.to_string())
            }
            Fan::Fake => {
                tracing::debug!(duty, "fan");
                Ok(())
            }
            Fan::None => Ok(()),
        }
    }

    fn available(&self) -> bool {
        !matches!(self, Fan::None)
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    match args.command {
        None => rt.block_on(serve(args)),
        Some(ref c) => rt.block_on(client(&args.socket, c)),
    }
}

async fn serve(args: Args) -> Result<()> {
    let mut eye = Eye::open(&args.device, args.fake)?;
    // A missing fan is not a reason to lose the eye: run without it and say so in the status.
    let mut fan = Fan::open(&args).unwrap_or_else(|e| {
        tracing::warn!(
            error = format!("{e:#}"),
            "no fan output; running without one"
        );
        Fan::None
    });
    let shared = Arc::new(Shared {
        start: Instant::now(),
        eye: Mutex::new(Stack::default()),
        fan: Mutex::new(Stack::default()),
        robotd: Mutex::new(false),
        temp_c: Mutex::new(read_temperature()),
        fan_out: Mutex::new((0.0, 0.0)),
        fan_available: fan.available(),
        changed: Notify::new(),
    });
    shared
        .eye
        .lock()
        .unwrap()
        .set(Layer::Ambient, express::ASLEEP, 0.0, None);
    tracing::info!(
        device = %args.device.display(), fake = args.fake, brightness = args.brightness,
        order = ?args.order, fan = shared.fan_available, fan_hz = args.fan_hz,
        "starting"
    );

    let listener = bind(&args.socket)?;
    let follower = tokio::spawn(follow(args.robot.clone(), shared.clone()));
    let thermometer = tokio::spawn(thermometer(shared.clone()));
    let server = tokio::spawn(accept(listener, shared.clone(), args.brightness));

    let curve = FanCurve::new(args.fan_off_c, args.fan_on_c, args.fan_full_c, args.fan_min);
    let drive = FanDrive::new(args.fan_min, FAN_KICK_S);
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    tokio::select! {
        r = render(&mut eye, &mut fan, &shared, args.brightness, args.order, curve, drive) => r?,
        _ = term.recv() => tracing::info!("SIGTERM; stopping"),
        _ = int.recv() => tracing::info!("SIGINT; stopping"),
    }
    follower.abort();
    thermometer.abort();
    server.abort();
    // Off on the way out: a lit eye on a robot whose daemon has stopped is a lie about state.
    let _ = eye.show(Rgb::OFF, args.brightness, args.order);
    // And the fan full: nothing will be watching the temperature once this process is gone.
    let _ = fan.set(1.0);
    let _ = std::fs::remove_file(&args.socket);
    Ok(())
}

/// Drive both outputs. Returns only on an eye output error; a fan write error is logged and the
/// next frame tries again.
async fn render(
    eye: &mut Eye,
    fan: &mut Fan,
    shared: &Shared,
    brightness: f32,
    order: Order,
    mut curve: FanCurve,
    mut drive: FanDrive,
) -> Result<()> {
    let mut last_rgb: Option<(Rgb, Instant)> = None;
    let mut last_duty: Option<(f32, Instant)> = None;
    let mut shown: Option<Layer> = None;
    let mut fan_failing = false;
    loop {
        let now = shared.now();

        let top = shared.eye.lock().unwrap().top(now);
        let (layer, rgb, eye_animated) = match top {
            Some((layer, look)) => (Some(layer), look.at(now), look.animated()),
            None => (None, Rgb::OFF, false),
        };
        if layer != shown {
            tracing::info!(layer = ?layer, "eye now showing");
            shown = layer;
        }
        if last_rgb.is_none_or(|(prev, at)| prev != rgb || at.elapsed() >= REFRESH) {
            eye.show(rgb, brightness, order)
                .context("writing to the eye")?;
            last_rgb = Some((rgb, Instant::now()));
        }

        let asked = shared.fan.lock().unwrap().top(now).map(|(_, look)| look);
        let floor = curve.floor(*shared.temp_c.lock().unwrap());
        let want = asked.map_or(0.0, |look| look.at(now)).max(floor);
        let duty = drive.duty(want, now);
        *shared.fan_out.lock().unwrap() = (duty, floor);
        let moved = |prev: f32| (prev - duty).abs() >= 0.005 || (prev == 0.0) != (duty == 0.0);
        if last_duty.is_none_or(|(prev, at)| moved(prev) || at.elapsed() >= REFRESH) {
            match fan.set(duty) {
                Ok(()) => {
                    if fan_failing {
                        tracing::info!("fan writes work again");
                        fan_failing = false;
                    }
                    last_duty = Some((duty, Instant::now()));
                }
                Err(e) if !fan_failing => {
                    tracing::warn!(error = %e, "writing the fan duty failed; retrying");
                    fan_failing = true;
                }
                Err(_) => {}
            }
        }

        // A kick in progress is animation too: it has to end on time.
        let kicking = duty == 1.0 && want < 1.0;
        let animated = eye_animated || asked.is_some_and(|l| l.animated()) || kicking;
        let wait = if animated { FRAME } else { REFRESH };
        let _ = tokio::time::timeout(wait, shared.changed.notified()).await;
    }
}

/// The hottest thermal zone, °C, or `None` when none could be read.
fn read_temperature() -> Option<f32> {
    let zones = std::fs::read_dir("/sys/class/thermal").ok()?;
    zones
        .filter_map(|z| z.ok())
        .filter(|z| z.file_name().to_string_lossy().starts_with("thermal_zone"))
        .filter_map(|z| std::fs::read_to_string(z.path().join("temp")).ok())
        .filter_map(|t| t.trim().parse::<f32>().ok())
        .map(|milli| milli / 1000.0)
        .reduce(f32::max)
}

async fn thermometer(shared: Arc<Shared>) {
    let mut said_missing = false;
    loop {
        tokio::time::sleep(THERMAL_EVERY).await;
        let t = read_temperature();
        if t.is_none() && !said_missing {
            tracing::warn!("no thermal zone readable; the fan runs full until one is");
            said_missing = true;
        }
        *shared.temp_c.lock().unwrap() = t;
        shared.changed();
    }
}

// ── Follower: robotd's state into the eye's fault and ambient layers ──────────

async fn follow(socket: PathBuf, shared: Arc<Shared>) {
    let mut said_missing = false;
    loop {
        match follow_once(&socket, &shared).await {
            Ok(()) => tracing::warn!("robotd closed the state stream"),
            Err(e) if !said_missing => {
                tracing::warn!(error = %e, "no robot.state; the eye shows asleep until it returns");
                said_missing = true;
            }
            Err(_) => {}
        }
        *shared.robotd.lock().unwrap() = false;
        {
            let now = shared.now();
            let mut eye = shared.eye.lock().unwrap();
            eye.clear(Layer::Fault);
            eye.set(Layer::Ambient, express::ASLEEP, now, None);
        }
        shared.changed();
        tokio::time::sleep(RETRY).await;
    }
}

async fn follow_once(socket: &Path, shared: &Shared) -> Result<()> {
    let stream = UnixStream::connect(socket)
        .await
        .with_context(|| format!("connecting to {}", socket.display()))?;
    let (read, mut write) = stream.into_split();
    let request = proto::Request::call(
        proto::Id::Number(1),
        &proto::Call::RobotSubscribe(proto::SubscribeParams { hz: Some(STATE_HZ) }),
    );
    let mut line = serde_json::to_vec(&request)?;
    line.push(b'\n');
    write.write_all(&line).await?;

    let mut reader = BufReader::new(read);
    let mut line = String::new();
    loop {
        line.clear();
        match tokio::time::timeout(STALE, reader.read_line(&mut line)).await {
            Err(_) => bail!("no robot.state for {STALE:?}"),
            Ok(Ok(0)) => return Ok(()),
            Ok(r) => {
                r?;
            }
        }
        // The subscribe answer, then notifications. Only the latter carry state.
        let Ok(message) = serde_json::from_str::<proto::Request>(line.trim()) else {
            continue;
        };
        if message.method != proto::method::ROBOT_STATE {
            continue;
        }
        let state: proto::RobotState =
            match serde_json::from_value(message.params.unwrap_or_default()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(error = %e, "unreadable robot.state");
                    continue;
                }
            };
        let (fault, ambient) = express::from_state(&state);
        {
            let mut up = shared.robotd.lock().unwrap();
            if !*up {
                tracing::info!("following robot.state");
                *up = true;
            }
        }
        {
            let now = shared.now();
            let mut eye = shared.eye.lock().unwrap();
            match fault {
                Some(look) => eye.set(Layer::Fault, look, now, None),
                None => eye.clear(Layer::Fault),
            }
            eye.set(Layer::Ambient, ambient, now, None);
        }
        shared.changed();
    }
}

// ── Server ───────────────────────────────────────────────────────────────────

fn bind(socket: &Path) -> Result<UnixListener> {
    if let Some(parent) = socket.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    let listener =
        UnixListener::bind(socket).with_context(|| format!("binding {}", socket.display()))?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(SOCKET_MODE))?;
    if let Err(e) = give_to_group(socket, GROUP) {
        tracing::warn!(error = %e, group = GROUP, "the express socket stays private to expressd");
    }
    tracing::info!(path = %socket.display(), "serving express.*");
    Ok(listener)
}

async fn accept(listener: UnixListener, shared: Arc<Shared>, brightness: f32) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let shared = shared.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, &shared, brightness).await {
                        tracing::debug!(error = %e, "express client ended");
                    }
                });
            }
            Err(e) => tracing::warn!(error = %e, "accept failed"),
        }
    }
}

async fn connection(stream: UnixStream, shared: &Shared, brightness: f32) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        let response = match serde_json::from_str::<proto::Request>(line.trim()) {
            Ok(request) => {
                let id = request.id.clone();
                match handle(&request, shared, brightness) {
                    Ok(result) => proto::Response::ok(id, &result),
                    Err(e) => proto::Response::err(id, e),
                }
            }
            Err(e) => proto::Response::err(
                None,
                proto::Error::new(proto::code::PARSE_ERROR, e.to_string()),
            ),
        };
        let mut out = serde_json::to_vec(&response)?;
        out.push(b'\n');
        write.write_all(&out).await?;
    }
}

fn params<T: for<'de> Deserialize<'de>>(request: &proto::Request) -> Result<T, proto::Error> {
    serde_json::from_value(request.params.clone().unwrap_or(serde_json::json!({})))
        .map_err(|e| proto::Error::new(proto::code::INVALID_PARAMS, e.to_string()))
}

fn invalid(message: String) -> proto::Error {
    proto::Error::new(proto::code::INVALID_PARAMS, message)
}

fn handle(
    request: &proto::Request,
    shared: &Shared,
    brightness: f32,
) -> Result<serde_json::Value, proto::Error> {
    let ok = || serde_json::json!({ "accepted": true });
    match request.method.as_str() {
        SET => {
            let p: SetParams = params(request)?;
            if !p.layer.client_settable() {
                return Err(invalid(format!(
                    "{:?} comes from robotd's state; set privacy, identify or mood",
                    p.layer
                )));
            }
            if p.eye.is_none() && p.fan.is_none() {
                return Err(invalid("nothing to set: give eye, fan or both".to_owned()));
            }
            let ttl = p.ttl_s.filter(|t| *t > 0.0).map(Duration::from_secs_f32);
            let now = shared.now();
            if let Some(look) = p.eye {
                shared.eye.lock().unwrap().set(p.layer, look, now, ttl);
            }
            if let Some(look) = p.fan {
                shared.fan.lock().unwrap().set(p.layer, look, now, ttl);
            }
            shared.changed();
            Ok(ok())
        }
        CLEAR => {
            let p: ClearParams = params(request)?;
            if !p.layer.client_settable() {
                return Err(invalid(format!(
                    "{:?} comes from robotd's state and cannot be cleared",
                    p.layer
                )));
            }
            shared.eye.lock().unwrap().clear(p.layer);
            shared.fan.lock().unwrap().clear(p.layer);
            shared.changed();
            Ok(ok())
        }
        IDENTIFY => {
            let p: IdentifyParams = params(request)?;
            let ttl = Duration::from_secs_f32(p.seconds.clamp(0.5, 120.0));
            let now = shared.now();
            shared
                .eye
                .lock()
                .unwrap()
                .set(Layer::Identify, express::IDENTIFY, now, Some(ttl));
            shared.changed();
            Ok(ok())
        }
        STATUS => {
            let now = shared.now();
            let eye = {
                let mut stack = shared.eye.lock().unwrap();
                let top = stack.top(now);
                EyeStatus {
                    showing: top.map(|(l, _)| l),
                    colour: top.map_or(Rgb::OFF, |(_, look)| look.at(now)),
                    layers: stack
                        .active()
                        .into_iter()
                        .map(|(layer, look)| LayerLook { layer, look })
                        .collect(),
                    brightness,
                }
            };
            let fan = {
                let mut stack = shared.fan.lock().unwrap();
                let (duty, floor) = *shared.fan_out.lock().unwrap();
                FanStatus {
                    available: shared.fan_available,
                    duty,
                    floor,
                    temp_c: *shared.temp_c.lock().unwrap(),
                    showing: stack.top(now).map(|(l, _)| l),
                    layers: stack
                        .active()
                        .into_iter()
                        .map(|(layer, look)| LayerLook { layer, look })
                        .collect(),
                }
            };
            let result = StatusResult {
                robotd: *shared.robotd.lock().unwrap(),
                eye,
                fan,
            };
            serde_json::to_value(result)
                .map_err(|e| proto::Error::new(proto::code::INTERNAL_ERROR, e.to_string()))
        }
        other => Err(proto::Error::new(
            proto::code::METHOD_NOT_FOUND,
            format!("{other}: expressd serves {SET}, {CLEAR}, {IDENTIFY} and {STATUS}"),
        )),
    }
}

fn give_to_group(socket: &Path, group: &str) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let name = CString::new(group).map_err(std::io::Error::other)?;
    // SAFETY: as in tofd: `getgrnam` returns a pointer into storage it owns, read at once, and
    // nothing else in this process calls into the group database.
    let entry = unsafe { libc::getgrnam(name.as_ptr()) };
    if entry.is_null() {
        return Err(std::io::Error::other(format!(
            "no {group} group on this system"
        )));
    }
    // SAFETY: checked non-null above.
    let gid = unsafe { (*entry).gr_gid };
    let path = CString::new(socket.as_os_str().as_bytes()).map_err(std::io::Error::other)?;
    // SAFETY: a valid C string path; `-1` leaves the owner alone.
    if unsafe { libc::chown(path.as_ptr(), u32::MAX, gid) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

// ── Client ───────────────────────────────────────────────────────────────────

async fn client(socket: &Path, command: &Command) -> Result<()> {
    let (method, params) = match *command {
        Command::Set {
            layer,
            eye,
            pattern,
            period,
            fan,
            fan_pattern,
            fan_period,
            ttl,
        } => {
            if eye.is_none() && fan.is_none() {
                bail!("nothing to set: give --eye R,G,B, --fan LEVEL, or both");
            }
            let params = SetParams {
                layer,
                eye: eye.map(|colour| Look {
                    colour,
                    pattern,
                    period_s: period,
                }),
                fan: fan.map(|level| FanLook {
                    level,
                    pattern: fan_pattern,
                    period_s: fan_period,
                }),
                ttl_s: ttl,
            };
            (SET, serde_json::to_value(params)?)
        }
        Command::Clear { layer } => (CLEAR, serde_json::to_value(ClearParams { layer })?),
        Command::Identify { seconds } => {
            (IDENTIFY, serde_json::to_value(IdentifyParams { seconds })?)
        }
        Command::Status => (STATUS, serde_json::json!({})),
    };
    let stream = UnixStream::connect(socket)
        .await
        .with_context(|| format!("connecting to {} (is expressd running?)", socket.display()))?;
    let (read, mut write) = stream.into_split();
    let request =
        serde_json::json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
    let mut line = serde_json::to_vec(&request)?;
    line.push(b'\n');
    write.write_all(&line).await?;
    let mut reply = String::new();
    BufReader::new(read).read_line(&mut reply).await?;
    let response: proto::Response = serde_json::from_str(reply.trim())?;
    if let Some(e) = response.error {
        bail!("{}", e.message);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&response.result.unwrap_or_default())?
    );
    Ok(())
}
