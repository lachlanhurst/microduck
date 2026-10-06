//! The wire protocol between the compute module and the servo bridge.
//!
//! The bridge is an STM32G474 that owns the three J288 servo bus segments and the trunk IMU. It
//! polls the servos on its own schedule and always holds the latest state; the compute module
//! talks to it over a full-duplex UART. `docs/design/bridge-protocol.md` owns the protocol: why
//! it is shaped this way, the timing, the watchdog. This crate is its single implementation, built
//! by both ends — `duck-control` on the compute module and the bridge firmware — so the two cannot
//! drift apart.
//!
//! Three exchanges:
//!
//! | compute sends | bridge answers | when |
//! |---|---|---|
//! | [`Command`] | nothing | once per control tick, after the policy runs |
//! | [`StateRequest`] | [`State`] | once per control tick, at the start |
//! | [`InfoRequest`] | [`Info`] | at startup |
//!
//! Every frame is `SYNC (2) · version (1) · kind (1) · payload length (2) · payload · CRC-32 (4)`,
//! little-endian throughout, with the CRC over everything after the sync bytes. A receiver that
//! loses its place drops bytes until the next sync pair whose frame checks out.
//!
//! All servo quantities are output side and in SI units, in the servo's own frame: position is
//! the J288's multi-turn position relative to where it powered up. Joint zeros and directions are
//! the compute module's business, not the bridge's.

#![no_std]

/// Bumped whenever a payload layout changes. A frame carrying another version is rejected by
/// [`FrameDecoder`] rather than misread, and [`Info`] lets the compute module say which side is
/// out of date before anything moves.
pub const PROTOCOL_VERSION: u8 = 1;

/// Servo slots in [`Command`] and [`State`], indexed by servo ID. The J288 bus addresses 0 to 14
/// (15 is broadcast), and the robot has fifteen joints with each servo's ID set to its joint's
/// index, so one array serves both.
pub const NUM_SERVOS: usize = 15;

/// Marks the start of a frame. An arbitrary pair, chosen to be rare in float payloads.
pub const SYNC: [u8; 2] = [0xD5, 0x6B];

/// Sync, version, kind and length.
pub const HEADER_LEN: usize = 6;
pub const CRC_LEN: usize = 4;
/// Far above any message here; a length beyond it is a corrupt header, not a big frame.
pub const MAX_PAYLOAD: usize = 1024;
pub const MAX_FRAME: usize = HEADER_LEN + MAX_PAYLOAD + CRC_LEN;

/// Message kinds. Compute-to-bridge kinds have the top bit clear, replies have it set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Command = 0x01,
    StateRequest = 0x02,
    InfoRequest = 0x03,
    State = 0x81,
    Info = 0x83,
}

impl Kind {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0x01 => Kind::Command,
            0x02 => Kind::StateRequest,
            0x03 => Kind::InfoRequest,
            0x81 => Kind::State,
            0x83 => Kind::Info,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The output buffer is too small for the frame.
    BufferTooSmall,
    /// A payload shorter or longer than its kind's layout.
    BadLength,
    /// A field holding a value its type does not define.
    BadValue,
    /// A frame of a different kind than the one asked for.
    WrongKind,
}

/// A message with a fixed payload layout.
pub trait Message: Sized {
    const KIND: Kind;
    const PAYLOAD_LEN: usize;
    fn write_payload(&self, w: &mut Writer<'_>);
    fn read_payload(r: &mut Reader<'_>) -> Result<Self, Error>;

    /// Encodes the whole frame into `buf`, returning its length.
    fn encode(&self, buf: &mut [u8]) -> Result<usize, Error> {
        let total = HEADER_LEN + Self::PAYLOAD_LEN + CRC_LEN;
        if buf.len() < total {
            return Err(Error::BufferTooSmall);
        }
        buf[0..2].copy_from_slice(&SYNC);
        buf[2] = PROTOCOL_VERSION;
        buf[3] = Self::KIND as u8;
        buf[4..6].copy_from_slice(&(Self::PAYLOAD_LEN as u16).to_le_bytes());
        let mut w = Writer::new(&mut buf[HEADER_LEN..HEADER_LEN + Self::PAYLOAD_LEN]);
        self.write_payload(&mut w);
        debug_assert_eq!(
            w.pos,
            Self::PAYLOAD_LEN,
            "payload layout and PAYLOAD_LEN disagree"
        );
        let crc = crc32fast::hash(&buf[2..HEADER_LEN + Self::PAYLOAD_LEN]);
        buf[HEADER_LEN + Self::PAYLOAD_LEN..total].copy_from_slice(&crc.to_le_bytes());
        Ok(total)
    }

    /// Decodes the payload of a frame already checked by [`FrameDecoder`].
    fn decode(frame: &Frame<'_>) -> Result<Self, Error> {
        if frame.kind != Self::KIND {
            return Err(Error::WrongKind);
        }
        if frame.payload.len() != Self::PAYLOAD_LEN {
            return Err(Error::BadLength);
        }
        let mut r = Reader::new(frame.payload);
        Self::read_payload(&mut r)
    }
}

// ---------------------------------------------------------------- compute to bridge

/// How a servo is driven, held until the next [`Command`] changes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ServoMode {
    /// J288 mode 0: the windings are shorted, a viscous brake. What every servo does until the
    /// compute module asks otherwise, and what the bridge falls back to when commands stop.
    #[default]
    Stop = 0,
    /// J288 mode 1, FOC: `tau = torque + kp (position - p) + kd (velocity - w)`.
    Foc = 1,
}

/// Something done once to a servo, not held. Performed when [`ServoCommand::action_id`] changes,
/// so a lost or repeated frame neither skips nor repeats it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Action {
    #[default]
    None = 0,
    /// J288 mode 6: clear latched faults.
    ClearFaults = 1,
    /// J288 mode 7: reset; the servo ignores commands for about 0.3 s.
    Reset = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ServoCommand {
    pub mode: ServoMode,
    pub action: Action,
    pub action_id: u8,
    /// Target position, rad.
    pub position: f32,
    /// Target speed, rad/s.
    pub velocity: f32,
    /// N·m/rad.
    pub kp: f32,
    /// N·m·s/rad.
    pub kd: f32,
    /// Feed-forward torque, N·m.
    pub torque: f32,
}

/// What every servo should do, sent once per control tick.
///
/// The bridge applies it on its next pass over the bus and holds it until the next one. If none
/// arrives for the watchdog period it stops every servo by itself (see [`state_flags::WATCHDOG`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Command {
    /// Increments per command. Echoed in [`State::command_seq`] once applied.
    pub seq: u32,
    /// The compute module's clock when it sent this, µs. Opaque to the bridge.
    pub host_time_us: u64,
    pub servos: [ServoCommand; NUM_SERVOS],
}

/// Asks for a [`State`]. The bridge answers from what it already holds, without touching the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StateRequest {
    /// The compute module's clock, µs, echoed in [`State::request_host_time_us`] so it can
    /// measure the round trip without keeping its own record.
    pub host_time_us: u64,
}

/// Asks for an [`Info`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InfoRequest;

// ---------------------------------------------------------------- bridge to compute

/// [`State::flags`] bits.
pub mod state_flags {
    /// No [`crate::Command`] arrived within the watchdog period, so the bridge has stopped every
    /// servo. Cleared by the next command.
    pub const WATCHDOG: u16 = 1 << 0;
    /// At least one [`crate::Command`] has arrived since the bridge started.
    pub const COMMANDED: u16 = 1 << 1;
}

/// [`ServoState::flags`] bits.
pub mod servo_flags {
    /// The last reply had its frame-timeout bit set.
    pub const TIMEOUT_LATCHED: u8 = 1 << 0;
    /// The bridge is sending timeout-clearing stop frames after a gap in replies.
    pub const RECOVERING: u8 = 1 << 1;
}

/// [`ServoState::segment`] for an ID no segment found.
pub const ABSENT: u8 = 0xFF;

/// One servo, as the bridge last heard from it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ServoState {
    /// Bus segment the servo answered on (0 = A, 1 = B, 2 = C, 3 = D), or [`ABSENT`]. Every other field
    /// is meaningless for an absent servo.
    pub segment: u8,
    /// J288 mode from the last reply.
    pub mode: u8,
    /// [`servo_flags`] bits.
    pub flags: u8,
    /// Housing temperature, °C.
    pub temp_c: i8,
    /// Winding temperature, °C. Rises far faster than the housing.
    pub winding_c: u8,
    /// Supply voltage in 0.5 V steps, as the servo reports it.
    pub volts_half: u8,
    pub warning: u8,
    pub error: u32,
    /// Multi-turn position from the rotor encoder, rad.
    pub position: f32,
    /// The servo's own speed estimate, filtered inside it (about 15 ms), rad/s.
    pub speed: f32,
    /// Speed from the bridge's position differences over about 15 ms of polls, rad/s. Not
    /// filtered by the servo, and the one hardware.md recommends for the policy.
    pub velocity: f32,
    /// Motor torque from the current, before gearbox losses, N·m.
    pub torque: f32,
    /// 13-bit output-shaft encoder, raw. Only trustworthy at rest.
    pub output_encoder: u16,
    /// How long before this frame the bridge last had a good reply, µs, saturating.
    pub age_us: u16,
    /// Good replies and failed exchanges since the bridge started, wrapping.
    pub replies: u16,
    pub failures: u16,
}

/// [`ImuState::status`] values.
pub mod imu_status {
    pub const STARTING: u8 = 0;
    pub const NOT_FOUND: u8 = 1;
    pub const CONFIG_FAILED: u8 = 2;
    pub const BUS_ERROR: u8 = 3;
    pub const RUNNING: u8 = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ImuState {
    /// One of [`imu_status`].
    pub status: u8,
    /// The 12-byte block `duck-control`'s `SflpDecoder` reads: gyro x, y, z as `i16` raw counts
    /// at ±500 dps, then the SFLP game-rotation quaternion x, y, z as half floats. All-zero
    /// quaternion bytes until the first fused sample.
    pub block: [u8; 12],
    /// How long before this frame the bridge last drained the IMU's FIFO, µs.
    pub age_us: u32,
    /// The chip's SFLP gravity estimate, mg.
    pub gravity_mg: [f32; 3],
    /// The chip's SFLP gyro bias estimate, mdps.
    pub gbias_mdps: [f32; 3],
    pub gyro_samples: u32,
    pub rotation_samples: u32,
    pub fifo_overruns: u16,
    pub int1_timeouts: u16,
}

/// Everything the bridge knows, answered to each [`StateRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct State {
    /// Increments per state frame.
    pub seq: u32,
    /// The last [`Command::seq`] applied, 0 if none.
    pub command_seq: u32,
    /// [`StateRequest::host_time_us`] of the request this answers.
    pub request_host_time_us: u64,
    /// The bridge's clock when it built this frame, µs since it started.
    pub bridge_time_us: u64,
    /// How long before this frame the last command arrived, µs, saturating.
    pub command_age_us: u32,
    /// [`state_flags`] bits.
    pub flags: u16,
    /// Pack voltage from the bridge's ADC, mV, or 0 while it is not measured.
    pub pack_mv: u16,
    /// Indexed by servo ID.
    pub servos: [ServoState; NUM_SERVOS],
    pub imu: ImuState,
    /// Passes over every segment since the bridge started.
    pub rounds: u32,
    /// Duration of the last pass, µs.
    pub round_us: u16,
    /// Frames the bridge dropped on its receive side: bad CRC, wrong version or length.
    pub link_errors: u16,
}

/// What the bridge is, answered to an [`InfoRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Info {
    pub protocol_version: u8,
    /// Firmware version, NUL-padded UTF-8.
    pub firmware: [u8; 32],
    pub sysclk_mhz: u16,
    /// Link baud rate the bridge is running, bits/s.
    pub link_baud: u32,
    /// Watchdog period, ms.
    pub watchdog_ms: u16,
    /// Segment each servo ID was found on, or [`ABSENT`], from the bridge's last scan.
    pub segments: [u8; NUM_SERVOS],
}

impl Info {
    /// The firmware version as text, up to the first NUL.
    pub fn firmware_str(&self) -> &str {
        let end = self
            .firmware
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.firmware.len());
        core::str::from_utf8(&self.firmware[..end]).unwrap_or("")
    }
}

// ---------------------------------------------------------------- layouts

const SERVO_COMMAND_LEN: usize = 3 + 5 * 4;
const SERVO_STATE_LEN: usize = 7 + 4 + 4 * 4 + 4 * 2;
const IMU_STATE_LEN: usize = 1 + 12 + 4 + 6 * 4 + 4 + 4 + 2 + 2;

impl Message for Command {
    const KIND: Kind = Kind::Command;
    const PAYLOAD_LEN: usize = 4 + 8 + NUM_SERVOS * SERVO_COMMAND_LEN;

    fn write_payload(&self, w: &mut Writer<'_>) {
        w.u32(self.seq);
        w.u64(self.host_time_us);
        for s in &self.servos {
            w.u8(s.mode as u8);
            w.u8(s.action as u8);
            w.u8(s.action_id);
            w.f32(s.position);
            w.f32(s.velocity);
            w.f32(s.kp);
            w.f32(s.kd);
            w.f32(s.torque);
        }
    }

    fn read_payload(r: &mut Reader<'_>) -> Result<Self, Error> {
        let mut c = Command {
            seq: r.u32(),
            host_time_us: r.u64(),
            ..Default::default()
        };
        for s in &mut c.servos {
            s.mode = match r.u8() {
                0 => ServoMode::Stop,
                1 => ServoMode::Foc,
                _ => return Err(Error::BadValue),
            };
            s.action = match r.u8() {
                0 => Action::None,
                1 => Action::ClearFaults,
                2 => Action::Reset,
                _ => return Err(Error::BadValue),
            };
            s.action_id = r.u8();
            s.position = r.f32();
            s.velocity = r.f32();
            s.kp = r.f32();
            s.kd = r.f32();
            s.torque = r.f32();
        }
        Ok(c)
    }
}

impl Message for StateRequest {
    const KIND: Kind = Kind::StateRequest;
    const PAYLOAD_LEN: usize = 8;

    fn write_payload(&self, w: &mut Writer<'_>) {
        w.u64(self.host_time_us);
    }

    fn read_payload(r: &mut Reader<'_>) -> Result<Self, Error> {
        Ok(StateRequest {
            host_time_us: r.u64(),
        })
    }
}

impl Message for InfoRequest {
    const KIND: Kind = Kind::InfoRequest;
    const PAYLOAD_LEN: usize = 0;

    fn write_payload(&self, _w: &mut Writer<'_>) {}

    fn read_payload(_r: &mut Reader<'_>) -> Result<Self, Error> {
        Ok(InfoRequest)
    }
}

impl Message for State {
    const KIND: Kind = Kind::State;
    const PAYLOAD_LEN: usize =
        4 + 4 + 8 + 8 + 4 + 2 + 2 + NUM_SERVOS * SERVO_STATE_LEN + IMU_STATE_LEN + 4 + 2 + 2;

    fn write_payload(&self, w: &mut Writer<'_>) {
        w.u32(self.seq);
        w.u32(self.command_seq);
        w.u64(self.request_host_time_us);
        w.u64(self.bridge_time_us);
        w.u32(self.command_age_us);
        w.u16(self.flags);
        w.u16(self.pack_mv);
        for s in &self.servos {
            w.u8(s.segment);
            w.u8(s.mode);
            w.u8(s.flags);
            w.u8(s.temp_c as u8);
            w.u8(s.winding_c);
            w.u8(s.volts_half);
            w.u8(s.warning);
            w.u32(s.error);
            w.f32(s.position);
            w.f32(s.speed);
            w.f32(s.velocity);
            w.f32(s.torque);
            w.u16(s.output_encoder);
            w.u16(s.age_us);
            w.u16(s.replies);
            w.u16(s.failures);
        }
        let i = &self.imu;
        w.u8(i.status);
        w.bytes(&i.block);
        w.u32(i.age_us);
        for v in i.gravity_mg.iter().chain(i.gbias_mdps.iter()) {
            w.f32(*v);
        }
        w.u32(i.gyro_samples);
        w.u32(i.rotation_samples);
        w.u16(i.fifo_overruns);
        w.u16(i.int1_timeouts);
        w.u32(self.rounds);
        w.u16(self.round_us);
        w.u16(self.link_errors);
    }

    fn read_payload(r: &mut Reader<'_>) -> Result<Self, Error> {
        let mut s = State {
            seq: r.u32(),
            command_seq: r.u32(),
            request_host_time_us: r.u64(),
            bridge_time_us: r.u64(),
            command_age_us: r.u32(),
            flags: r.u16(),
            pack_mv: r.u16(),
            ..Default::default()
        };
        for v in &mut s.servos {
            v.segment = r.u8();
            v.mode = r.u8();
            v.flags = r.u8();
            v.temp_c = r.u8() as i8;
            v.winding_c = r.u8();
            v.volts_half = r.u8();
            v.warning = r.u8();
            v.error = r.u32();
            v.position = r.f32();
            v.speed = r.f32();
            v.velocity = r.f32();
            v.torque = r.f32();
            v.output_encoder = r.u16();
            v.age_us = r.u16();
            v.replies = r.u16();
            v.failures = r.u16();
        }
        let i = &mut s.imu;
        i.status = r.u8();
        r.bytes(&mut i.block);
        i.age_us = r.u32();
        for v in i.gravity_mg.iter_mut().chain(i.gbias_mdps.iter_mut()) {
            *v = r.f32();
        }
        i.gyro_samples = r.u32();
        i.rotation_samples = r.u32();
        i.fifo_overruns = r.u16();
        i.int1_timeouts = r.u16();
        s.rounds = r.u32();
        s.round_us = r.u16();
        s.link_errors = r.u16();
        Ok(s)
    }
}

impl Message for Info {
    const KIND: Kind = Kind::Info;
    const PAYLOAD_LEN: usize = 1 + 32 + 2 + 4 + 2 + NUM_SERVOS;

    fn write_payload(&self, w: &mut Writer<'_>) {
        w.u8(self.protocol_version);
        w.bytes(&self.firmware);
        w.u16(self.sysclk_mhz);
        w.u32(self.link_baud);
        w.u16(self.watchdog_ms);
        w.bytes(&self.segments);
    }

    fn read_payload(r: &mut Reader<'_>) -> Result<Self, Error> {
        let mut i = Info {
            protocol_version: r.u8(),
            ..Default::default()
        };
        r.bytes(&mut i.firmware);
        i.sysclk_mhz = r.u16();
        i.link_baud = r.u32();
        i.watchdog_ms = r.u16();
        r.bytes(&mut i.segments);
        Ok(i)
    }
}

// ---------------------------------------------------------------- byte cursors

/// Little-endian writer over a buffer sized exactly to the payload; layouts are fixed, so an
/// overrun is a bug in a `PAYLOAD_LEN` and panics in the encoder's slice indexing.
pub struct Writer<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Writer<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn bytes(&mut self, b: &[u8]) {
        self.buf[self.pos..self.pos + b.len()].copy_from_slice(b);
        self.pos += b.len();
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.bytes(&v.to_le_bytes());
    }
}

/// Little-endian reader. [`Message::decode`] checks the payload length first, so reads never run
/// past the end.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn bytes(&mut self, out: &mut [u8]) {
        out.copy_from_slice(&self.buf[self.pos..self.pos + out.len()]);
        self.pos += out.len();
    }
    fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0u8; N];
        self.bytes(&mut a);
        a
    }
    fn u8(&mut self) -> u8 {
        self.array::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.array())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.array())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.array())
    }
}

// ---------------------------------------------------------------- stream decoding

/// A checked frame: version and CRC verified, payload not yet interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    pub kind: Kind,
    pub payload: &'a [u8],
}

/// Finds frames in a byte stream. Feed it whatever the UART delivered, in any chunking.
///
/// Bytes that cannot start a valid frame are dropped and counted, and a frame that fails its
/// CRC costs only its first byte: the search resumes one byte on, so a sync pair inside a
/// corrupt frame cannot hide the real frame after it.
pub struct FrameDecoder {
    buf: [u8; MAX_FRAME],
    len: usize,
    /// Frames rejected for CRC, version, kind or length since this decoder was made.
    pub errors: u32,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameDecoder {
    pub const fn new() -> Self {
        Self {
            buf: [0; MAX_FRAME],
            len: 0,
            errors: 0,
        }
    }

    /// Appends as much of `data` as fits and returns how many bytes it took. Call
    /// [`Self::next_frame`] until it returns `None`, then push the rest.
    pub fn push(&mut self, data: &[u8]) -> usize {
        let n = data.len().min(self.buf.len() - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&data[..n]);
        self.len += n;
        n
    }

    /// The next complete, checked frame in the buffer, if any. The frame borrows the decoder;
    /// its bytes are consumed on the following call.
    pub fn next_frame(&mut self) -> Option<Frame<'_>> {
        loop {
            // Drop everything before the next sync pair.
            let start = (0..self.len.saturating_sub(1))
                .find(|&i| self.buf[i] == SYNC[0] && self.buf[i + 1] == SYNC[1]);
            match start {
                Some(0) => {}
                Some(i) => self.consume(i),
                None => {
                    // Keep a trailing first sync byte: its partner may be in the next chunk.
                    let keep = usize::from(self.len > 0 && self.buf[self.len - 1] == SYNC[0]);
                    self.consume(self.len - keep);
                    return None;
                }
            }
            if self.len < HEADER_LEN {
                return None;
            }
            let payload_len = u16::from_le_bytes([self.buf[4], self.buf[5]]) as usize;
            let kind = Kind::from_u8(self.buf[3]);
            if self.buf[2] != PROTOCOL_VERSION || kind.is_none() || payload_len > MAX_PAYLOAD {
                self.reject();
                continue;
            }
            let total = HEADER_LEN + payload_len + CRC_LEN;
            if self.len < total {
                return None;
            }
            let crc = u32::from_le_bytes(self.buf[total - 4..total].try_into().unwrap());
            if crc != crc32fast::hash(&self.buf[2..total - 4]) {
                self.reject();
                continue;
            }
            return Some(self.take_frame(kind.unwrap(), total));
        }
    }

    /// Consumes the frame at the front and returns it. The returned slice has to outlive the
    /// consumption, so the frame is rotated to just past the remaining bytes rather than
    /// overwritten; the next `push` reclaims that space once the borrow has ended.
    fn take_frame(&mut self, kind: Kind, total: usize) -> Frame<'_> {
        let rest = self.len - total;
        self.buf[..self.len].rotate_left(total);
        self.len = rest;
        let payload_start = rest + HEADER_LEN;
        let payload_end = rest + total - CRC_LEN;
        Frame {
            kind,
            payload: &self.buf[payload_start..payload_end],
        }
    }

    fn reject(&mut self) {
        self.errors = self.errors.wrapping_add(1);
        self.consume(1);
    }

    fn consume(&mut self, n: usize) {
        self.buf.copy_within(n..self.len, 0);
        self.len -= n;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn sample_command() -> Command {
        let mut c = Command {
            seq: 7,
            host_time_us: 123_456_789_012,
            ..Default::default()
        };
        for (i, s) in c.servos.iter_mut().enumerate() {
            *s = ServoCommand {
                mode: if i % 2 == 0 {
                    ServoMode::Foc
                } else {
                    ServoMode::Stop
                },
                action: if i == 3 {
                    Action::ClearFaults
                } else {
                    Action::None
                },
                action_id: i as u8,
                position: i as f32 * 0.1 - 0.7,
                velocity: -(i as f32),
                kp: 2.0,
                kd: 0.05,
                torque: 0.01 * i as f32,
            };
        }
        c
    }

    fn sample_state() -> State {
        let mut s = State {
            seq: 99,
            command_seq: 7,
            request_host_time_us: 5,
            bridge_time_us: 9_000_000_000,
            command_age_us: 1234,
            flags: state_flags::COMMANDED,
            pack_mv: 24_100,
            rounds: 1_000_000,
            round_us: 1800,
            link_errors: 3,
            ..Default::default()
        };
        for (i, v) in s.servos.iter_mut().enumerate() {
            *v = ServoState {
                segment: (i / 5) as u8,
                mode: 1,
                flags: servo_flags::RECOVERING,
                temp_c: -5 + i as i8,
                winding_c: 40,
                volts_half: 48,
                warning: 1,
                error: 0xDEAD_0000 | i as u32,
                position: 1.5 * i as f32,
                speed: -0.25,
                velocity: 0.125,
                torque: 0.5,
                output_encoder: 8191,
                age_us: 2100,
                replies: 65535,
                failures: 2,
            };
        }
        s.servos[14].segment = ABSENT;
        s.imu = ImuState {
            status: imu_status::RUNNING,
            block: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            age_us: 4000,
            gravity_mg: [1.0, -2.0, 999.0],
            gbias_mdps: [-337.0, -96.0, 687.0],
            gyro_samples: 46644,
            rotation_samples: 46643,
            fifo_overruns: 1,
            int1_timeouts: 2,
        };
        s
    }

    fn frame_of<M: Message>(m: &M) -> Vec<u8> {
        let mut buf = [0u8; MAX_FRAME];
        let n = m.encode(&mut buf).unwrap();
        buf[..n].to_vec()
    }

    fn decode_one<M: Message>(bytes: &[u8]) -> M {
        let mut d = FrameDecoder::new();
        assert_eq!(d.push(bytes), bytes.len());
        let f = d.next_frame().expect("a frame");
        M::decode(&f).unwrap()
    }

    /// The CRC must be the standard CRC-32 so either end can be checked against any other tool.
    /// The check value for "123456789" is the published one for CRC-32/ISO-HDLC.
    #[test]
    fn crc_is_the_standard_crc32() {
        assert_eq!(crc32fast::hash(b"123456789"), 0xCBF4_3926);
    }

    /// Every message must survive the round trip field for field; a field written and read at
    /// different offsets would hand one servo another's numbers.
    #[test]
    fn every_message_round_trips() {
        let c = sample_command();
        assert_eq!(decode_one::<Command>(&frame_of(&c)), c);
        let s = sample_state();
        assert_eq!(decode_one::<State>(&frame_of(&s)), s);
        let r = StateRequest {
            host_time_us: u64::MAX - 1,
        };
        assert_eq!(decode_one::<StateRequest>(&frame_of(&r)), r);
        assert_eq!(
            decode_one::<InfoRequest>(&frame_of(&InfoRequest)),
            InfoRequest
        );
        let mut i = Info {
            protocol_version: 1,
            sysclk_mhz: 168,
            link_baud: 2_000_000,
            watchdog_ms: 100,
            ..Default::default()
        };
        i.firmware[..5].copy_from_slice(b"0.1.0");
        i.segments = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, ABSENT];
        let back = decode_one::<Info>(&frame_of(&i));
        assert_eq!(back, i);
        assert_eq!(back.firmware_str(), "0.1.0");
    }

    /// The link budget in bridge-protocol.md is worked from these sizes; if a layout grows, the
    /// doc's 100 Hz arithmetic has to be redone, so the sizes are pinned here.
    #[test]
    fn frame_sizes_match_the_link_budget() {
        assert_eq!(frame_of(&sample_command()).len(), 367);
        assert_eq!(frame_of(&sample_state()).len(), 628);
        assert_eq!(frame_of(&StateRequest::default()).len(), 18);
    }

    /// A stream arrives in arbitrary pieces; a frame split across reads must still decode.
    #[test]
    fn frames_decode_from_any_chunking() {
        let bytes = [frame_of(&sample_state()), frame_of(&sample_command())].concat();
        for chunk in [1, 2, 7, 64, 500, bytes.len()] {
            let mut d = FrameDecoder::new();
            let mut kinds = Vec::new();
            for piece in bytes.chunks(chunk) {
                assert_eq!(d.push(piece), piece.len());
                while let Some(f) = d.next_frame() {
                    kinds.push(f.kind);
                }
            }
            assert_eq!(kinds, [Kind::State, Kind::Command], "chunk {chunk}");
            assert_eq!(d.errors, 0);
        }
    }

    /// Line noise before a frame, and a corrupt frame between good ones, must cost only the
    /// damaged bytes: the decoder resynchronises on the next good frame.
    #[test]
    fn garbage_and_corrupt_frames_are_skipped() {
        let good = frame_of(&StateRequest { host_time_us: 42 });
        let mut bad = frame_of(&StateRequest { host_time_us: 43 });
        bad[8] ^= 0x40;
        let bytes = [
            &[0x00, SYNC[0], 0x12, SYNC[0], SYNC[1], 0xFF][..],
            &bad,
            &good,
        ]
        .concat();
        let mut d = FrameDecoder::new();
        d.push(&bytes);
        let f = d.next_frame().expect("the good frame");
        assert_eq!(StateRequest::decode(&f).unwrap().host_time_us, 42);
        assert!(d.next_frame().is_none());
        assert!(
            d.errors >= 2,
            "the bad header and the bad CRC are both counted"
        );
    }

    /// A frame from another protocol version must be refused, not misread with today's layout.
    #[test]
    fn another_protocol_version_is_rejected() {
        let mut bytes = frame_of(&StateRequest::default());
        bytes[2] = PROTOCOL_VERSION + 1;
        let mut d = FrameDecoder::new();
        d.push(&bytes);
        assert!(d.next_frame().is_none());
        assert_eq!(d.errors, 1);
    }

    /// An undefined mode must fail the decode: the bridge must never drive a servo in a mode it
    /// guessed.
    #[test]
    fn an_undefined_servo_mode_is_refused() {
        let mut bytes = frame_of(&sample_command());
        bytes[HEADER_LEN + 12] = 9;
        let end = bytes.len() - CRC_LEN;
        let crc = crc32fast::hash(&bytes[2..end]);
        bytes[end..].copy_from_slice(&crc.to_le_bytes());
        let mut d = FrameDecoder::new();
        d.push(&bytes);
        let f = d.next_frame().unwrap();
        assert_eq!(Command::decode(&f), Err(Error::BadValue));
    }
}
