//! `eyed`: owns the eye LED and decides what it shows.
//!
//! Three tasks share one [`Stack`] (see `lib.rs` for the layers):
//!
//! - **The follower** subscribes to robotd's `robot.state` at 10 Hz and writes the `fault` and
//!   `ambient` layers. No robotd, or no frame for two seconds, is the `asleep` look: an eye that
//!   froze on "walking" when the brain died would be worse than no eye.
//! - **The server** answers `eye.set`, `eye.clear`, `eye.identify` and `eye.status` on
//!   `/run/eyed/eye.sock`, one JSON-RPC request per line, as many as a client likes.
//! - **The renderer** draws the winning look at ~30 Hz while it animates, and otherwise only on a
//!   change (plus a refresh every second, so a glitched frame does not stick).
//!
//! The methods are this daemon's own and live here rather than in `duck-ipc-proto`: the eye is
//! Dukki's addition, and keeping it out of the shared protocol keeps the upstream diff at zero.
//!
//! `eyed set mood 0 255 0 --pattern breathe` and friends are a client for the same socket, so a
//! shell on the robot can drive the eye without writing JSON.

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
use eye::{Layer, Look, Order, Pattern, Rgb, Stack};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;

const SOCKET: &str = "/run/eyed/eye.sock";
const SOCKET_MODE: u32 = 0o660;
/// Whoever may watch the robot may set its mood, as with tofd's and robotd's sockets.
const GROUP: &str = "robot";

const SET: &str = "eye.set";
const CLEAR: &str = "eye.clear";
const IDENTIFY: &str = "eye.identify";
const STATUS: &str = "eye.status";

/// robot.state rate asked for. The eye changes at human speed; 10 Hz is already generous.
const STATE_HZ: u32 = 10;
/// No frame for this long means robotd has stopped talking, whatever the socket says.
const STALE: Duration = Duration::from_secs(2);
const RETRY: Duration = Duration::from_secs(2);
/// Frame interval while animating. Smooth to the eye, and ~30 small SPI writes a second.
const FRAME: Duration = Duration::from_millis(33);
/// Rewrite an unchanged colour this often, so one corrupted frame cannot stick.
const REFRESH: Duration = Duration::from_secs(1);

#[derive(Parser)]
#[command(about = "The eye LED daemon, and a client for it")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// spidev node the LED's data line hangs off (setup-eye.sh names it).
    #[arg(long, default_value = "/dev/spidev-eye")]
    device: PathBuf,
    /// Brightness cap, 0 to 1. A bare 5050 LED at full scale is painful to look at.
    #[arg(long, default_value_t = 0.15)]
    brightness: f32,
    /// Colour order on the wire.
    #[arg(long, value_enum, default_value_t = Order::Grb)]
    order: Order,
    /// Log colours instead of driving the LED: for a laptop, or a board without one.
    #[arg(long)]
    fake: bool,
    /// This daemon's socket (the client subcommands use it too).
    #[arg(long, default_value = SOCKET)]
    socket: PathBuf,
    /// robotd's socket.
    #[arg(long, default_value = proto::socket::ROBOT)]
    robot: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Set a layer: privacy, identify or mood.
    Set {
        #[arg(value_parser = parse_layer)]
        layer: Layer,
        r: u8,
        g: u8,
        b: u8,
        #[arg(long, value_parser = parse_pattern, default_value = "solid")]
        pattern: Pattern,
        /// Seconds per cycle of the pattern.
        #[arg(long, default_value_t = 2.0)]
        period: f32,
        /// Clear the layer after this many seconds.
        #[arg(long)]
        ttl: Option<f32>,
    },
    /// Clear a layer, uncovering the one beneath.
    Clear {
        #[arg(value_parser = parse_layer)]
        layer: Layer,
    },
    /// Blink white for a few seconds, to find this robot among others.
    Identify {
        #[arg(long, default_value_t = 10.0)]
        seconds: f32,
    },
    /// What the eye is showing, and why.
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

// ── Wire shapes for this daemon's methods ────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct SetParams {
    layer: Layer,
    #[serde(flatten)]
    look: Look,
    #[serde(default)]
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
struct LayerLook {
    layer: Layer,
    #[serde(flatten)]
    look: Look,
}

#[derive(Serialize)]
struct StatusResult {
    /// The layer being shown, if any.
    showing: Option<Layer>,
    /// The colour on the LED this instant, before brightness.
    colour: Rgb,
    /// Whether robot.state is arriving.
    robotd: bool,
    layers: Vec<LayerLook>,
    brightness: f32,
}

// ── Shared state ─────────────────────────────────────────────────────────────

struct Shared {
    start: Instant,
    stack: Mutex<Stack>,
    robotd: Mutex<bool>,
    changed: Notify,
}

impl Shared {
    fn now(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    fn update(&self, f: impl FnOnce(&mut Stack, f32)) {
        f(&mut self.stack.lock().unwrap(), self.now());
        self.changed.notify_one();
    }
}

// ── Output ───────────────────────────────────────────────────────────────────

enum Output {
    Spi(File),
    Fake,
}

impl Output {
    fn open(device: &Path, fake: bool) -> Result<Self> {
        if fake {
            return Ok(Output::Fake);
        }
        let file = File::options().write(true).open(device).with_context(|| {
            format!("opening {} (run setup-eye.sh, or --fake)", device.display())
        })?;
        // spidev ioctls: _IOW('k', 1, u8) and _IOW('k', 4, u32).
        const SPI_IOC_WR_MODE: u64 = 0x4001_6b01;
        const SPI_IOC_WR_MAX_SPEED_HZ: u64 = 0x4004_6b04;
        let mode: u8 = 0;
        let hz: u32 = eye::SPI_HZ;
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
        Ok(Output::Spi(file))
    }

    /// One frame. A single `write` is a single SPI transfer, which is what keeps the timing.
    fn show(&mut self, rgb: Rgb, brightness: f32, order: Order) -> std::io::Result<()> {
        match self {
            Output::Spi(file) => file.write_all(&eye::encode(rgb, brightness, order)),
            Output::Fake => {
                tracing::debug!(r = rgb.0, g = rgb.1, b = rgb.2, "eye");
                Ok(())
            }
        }
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
    let mut output = Output::open(&args.device, args.fake)?;
    let shared = Arc::new(Shared {
        start: Instant::now(),
        stack: Mutex::new(Stack::default()),
        robotd: Mutex::new(false),
        changed: Notify::new(),
    });
    shared.update(|s, now| s.set(Layer::Ambient, eye::ASLEEP, now, None));
    tracing::info!(
        device = %args.device.display(), fake = args.fake, brightness = args.brightness,
        order = ?args.order, "starting"
    );

    let listener = bind(&args.socket)?;
    let follower = tokio::spawn(follow(args.robot.clone(), shared.clone()));
    let server = tokio::spawn(accept(listener, shared.clone(), args.brightness));

    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    tokio::select! {
        r = render(&mut output, &shared, args.brightness, args.order) => r?,
        _ = term.recv() => tracing::info!("SIGTERM; stopping"),
        _ = int.recv() => tracing::info!("SIGINT; stopping"),
    }
    follower.abort();
    server.abort();
    // Off on the way out: a lit eye on a robot whose daemon has stopped is a lie about state.
    let _ = output.show(Rgb::OFF, args.brightness, args.order);
    let _ = std::fs::remove_file(&args.socket);
    Ok(())
}

/// Draw the winning look. Returns only on an output error.
async fn render(output: &mut Output, shared: &Shared, brightness: f32, order: Order) -> Result<()> {
    let mut last: Option<(Rgb, Instant)> = None;
    let mut shown: Option<Layer> = None;
    loop {
        let now = shared.now();
        let top = shared.stack.lock().unwrap().top(now);
        let (layer, rgb, animated) = match top {
            Some((layer, look)) => (Some(layer), look.at(now), look.animated()),
            None => (None, Rgb::OFF, false),
        };
        if layer != shown {
            tracing::info!(layer = ?layer, "eye now showing");
            shown = layer;
        }
        let due = last.is_none_or(|(prev, at)| prev != rgb || at.elapsed() >= REFRESH);
        if due {
            output
                .show(rgb, brightness, order)
                .context("writing to the eye")?;
            last = Some((rgb, Instant::now()));
        }
        let wait = if animated { FRAME } else { REFRESH };
        let _ = tokio::time::timeout(wait, shared.changed.notified()).await;
    }
}

// ── Follower: robotd's state into the fault and ambient layers ───────────────

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
        shared.update(|s, now| {
            s.clear(Layer::Fault);
            s.set(Layer::Ambient, eye::ASLEEP, now, None);
        });
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
        let (fault, ambient) = eye::from_state(&state);
        {
            let mut up = shared.robotd.lock().unwrap();
            if !*up {
                tracing::info!("following robot.state");
                *up = true;
            }
        }
        shared.update(|s, now| {
            match fault {
                Some(look) => s.set(Layer::Fault, look, now, None),
                None => s.clear(Layer::Fault),
            }
            s.set(Layer::Ambient, ambient, now, None);
        });
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
        tracing::warn!(error = %e, group = GROUP, "the eye socket stays private to eyed");
    }
    tracing::info!(path = %socket.display(), "serving eye.*");
    Ok(listener)
}

async fn accept(listener: UnixListener, shared: Arc<Shared>, brightness: f32) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let shared = shared.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, &shared, brightness).await {
                        tracing::debug!(error = %e, "eye client ended");
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
                return Err(proto::Error::new(
                    proto::code::INVALID_PARAMS,
                    format!(
                        "{:?} comes from robotd's state; set privacy, identify or mood",
                        p.layer
                    ),
                ));
            }
            let ttl = p.ttl_s.filter(|t| *t > 0.0).map(Duration::from_secs_f32);
            shared.update(|s, now| s.set(p.layer, p.look, now, ttl));
            Ok(ok())
        }
        CLEAR => {
            let p: ClearParams = params(request)?;
            if !p.layer.client_settable() {
                return Err(proto::Error::new(
                    proto::code::INVALID_PARAMS,
                    format!(
                        "{:?} comes from robotd's state and cannot be cleared",
                        p.layer
                    ),
                ));
            }
            shared.update(|s, _| s.clear(p.layer));
            Ok(ok())
        }
        IDENTIFY => {
            let p: IdentifyParams = params(request)?;
            let ttl = Duration::from_secs_f32(p.seconds.clamp(0.5, 120.0));
            shared.update(|s, now| s.set(Layer::Identify, eye::IDENTIFY, now, Some(ttl)));
            Ok(ok())
        }
        STATUS => {
            let now = shared.now();
            let mut stack = shared.stack.lock().unwrap();
            let top = stack.top(now);
            let result = StatusResult {
                showing: top.map(|(l, _)| l),
                colour: top.map_or(Rgb::OFF, |(_, look)| look.at(now)),
                robotd: *shared.robotd.lock().unwrap(),
                layers: stack
                    .active()
                    .into_iter()
                    .map(|(layer, look)| LayerLook { layer, look })
                    .collect(),
                brightness,
            };
            serde_json::to_value(result)
                .map_err(|e| proto::Error::new(proto::code::INTERNAL_ERROR, e.to_string()))
        }
        other => Err(proto::Error::new(
            proto::code::METHOD_NOT_FOUND,
            format!("{other}: eyed serves {SET}, {CLEAR}, {IDENTIFY} and {STATUS}"),
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
            r,
            g,
            b,
            pattern,
            period,
            ttl,
        } => (
            SET,
            serde_json::to_value(SetParams {
                layer,
                look: Look {
                    colour: Rgb(r, g, b),
                    pattern,
                    period_s: period,
                },
                ttl_s: ttl,
            })?,
        ),
        Command::Clear { layer } => (CLEAR, serde_json::to_value(ClearParams { layer })?),
        Command::Identify { seconds } => {
            (IDENTIFY, serde_json::to_value(IdentifyParams { seconds })?)
        }
        Command::Status => (STATUS, serde_json::json!({})),
    };
    let stream = UnixStream::connect(socket)
        .await
        .with_context(|| format!("connecting to {} (is eyed running?)", socket.display()))?;
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
