//! The robot through the servo bridge: fifteen Unitree J288s and the trunk IMU behind an
//! STM32G474, over `duck-bridge-proto`.
//!
//! `docs/design/bridge-protocol.md` owns the protocol. In short: the bridge polls the servos on
//! its own schedule, so a [`RobotIo::read`] is one small request and one [`proto::Sample`],
//! answered after a fresh pass over the servos; a [`RobotIo::write`] is one [`proto::Targets`]
//! frame with no reply, or a full [`proto::Command`] when a mode, gain or action changed and about
//! once a second. The full [`proto::State`], with temperatures and diagnostics, is asked for about
//! once a second after a write, so it never sits in a read. The control rate is this side's
//! choice alone.
//!
//! Each J288's ID is its joint's index (`motor-setup.md`), so servo slot `j` in the protocol is
//! joint `j` in [`crate::model::JOINT_NAMES`].
//!
//! Joint angles. The bridge reports each J288's multi-turn position, which the servo seeds from
//! its absolute output encoder at power-up, so it reads that encoder's angle within the first
//! turn and follows motion smoothly from there. With [`BridgeConfig::zero_counts`] set (robotd's
//! `[bus.bridge_zero]`, from the calibration jig), a joint's angle is that position less the
//! joint's zero and a whole number of turns. The turns are chosen the first time each servo is
//! read, so the angle lands in the one turn centred on the joint's measured travel
//! ([`crate::model::travel_centre`]), and held from then on: a joint moved across the turn's
//! edge while running keeps counting rather than jumping a turn. Targets go back the same way.
//! Every joint turns the way the model's does (checked on Dukki 21/11/2026), so there is no
//! sign. Without zeros, positions pass through unchanged, in the servos' own frame.
//!
//! A servo that loses power while the bridge runs seeds its position afresh when it returns,
//! which can be a turn away from the count the turns were chosen against. Nothing here notices
//! that yet; it would show as a joint reading a turn out.

use std::io::{ErrorKind, Read, Write};
use std::time::{Duration, Instant};

use duck_bridge_proto::{self as proto, Action, FrameDecoder, Message, ServoMode};

use crate::imu::SflpDecoder;
use crate::io::{
    HostState, ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors,
};
use crate::model::{JOINT_IDS, JOINT_NAMES, NUM_JOINTS, travel_centre};

/// What the bridge firmware runs today (`link.rs` there). 4 Mbps waits on a scope check of the
/// neck harness.
pub const DEFAULT_BAUD: u32 = 2_000_000;

/// A sample takes a pass over the servos (under 1 ms) and 1.1 ms on the wire at 2 Mbps; a full
/// state frame 3.1 ms, answered from memory. Short enough to leave a 100 Hz tick most of its
/// budget when the bridge does not answer.
const REPLY_TIMEOUT: Duration = Duration::from_millis(8);
/// How often a full command goes out even when only targets changed, so a lost one is repaired.
const FULL_COMMAND_EVERY: Duration = Duration::from_secs(1);
/// How often a write also asks for a full state frame, for the slow sensors and diagnostics.
const FULL_STATE_EVERY: Duration = Duration::from_secs(1);
/// The info exchange happens once, at open, where a slow answer costs nothing.
const INFO_TIMEOUT: Duration = Duration::from_millis(250);
/// Granularity of the blocking reads inside a reply wait.
const PORT_READ_TIMEOUT: Duration = Duration::from_millis(2);
/// A servo whose last reply is older than this fails the read. The bridge polls every servo
/// several hundred times a second, so this is a servo that has stopped answering, not a slow
/// one — and a stale position handed to the policy is worse than a coasted tick.
const MAX_SERVO_AGE_US: u16 = 50_000;
/// J288 torque constant, output side, N·m/A (hardware.md §5.1). Only used to express torque in
/// the `currents_ma` field the rest of the stack already reads as load.
const TORQUE_PER_AMP: f64 = 0.554;

/// What turns the stack's gain and torque calls into J288 gains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BridgeConfig {
    /// J288 kp (N·m/rad, output side) per unit of [`RobotIo::set_gain`]'s XL330-style gain. The
    /// policy's gain of 200 at the default 0.01 is 2.0 N·m/rad, what the J288 sim-to-real runs
    /// used; the limp gain of 50 becomes 0.5.
    pub kp_per_gain: f64,
    /// J288 kd, N·m·s/rad, on every joint driven in FOC.
    pub kd: f64,
    /// Each joint's zero as its output encoder reads it, counts (8192 a turn), indexed as
    /// [`crate::model::JOINT_NAMES`]. `None` passes positions through in the servos' own frame.
    pub zero_counts: Option<[u16; NUM_JOINTS]>,
    /// The trunk IMU's sensor-to-trunk rotation, scalar-first. [`SflpDecoder::DUKKI_MOUNT`]
    /// unless a calibration says otherwise.
    pub imu_mount: [f64; 4],
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            kp_per_gain: 0.01,
            kd: 0.05,
            zero_counts: None,
            imu_mount: SflpDecoder::DUKKI_MOUNT,
        }
    }
}

/// The servo bridge over any byte transport; a serial port on the robot, a simulated bridge in
/// tests.
pub struct BridgeIo<T> {
    port: T,
    decoder: FrameDecoder,
    out: Vec<u8>,
    config: BridgeConfig,
    epoch: Instant,
    info: proto::Info,
    command: proto::Command,
    torque_on: bool,
    /// Whether `command` carries real targets yet. Torque is never enabled onto targets nobody
    /// set: that would drive every joint to position zero.
    targets_set: bool,
    kp: f64,
    /// The last full command sent and when, to know when targets alone will do.
    last_full: Option<(proto::Command, Instant)>,
    /// When the next write also asks for a full state frame.
    state_due: Instant,
    last_state: Option<proto::State>,
    last_sample: Option<proto::Sample>,
    /// State flags and IMU status from the newest sample or state frame, whichever came last.
    flags: u16,
    imu_status: u8,
    imu: SflpDecoder,
    stale_imu: ImuStale,
    last_rotation_samples: Option<u32>,
    /// What to subtract from each servo's position to get its joint angle: the joint's zero
    /// plus whole turns, chosen the first time the servo is read. All zero without
    /// calibration.
    offsets: [Option<f64>; NUM_JOINTS],
}

/// Output encoder counts in a turn.
pub const ENCODER_COUNTS: u16 = 8192;

/// The offset that puts a servo's `position` into the turn centred on `centre`, for a joint
/// whose zero is `zero` (both radians).
fn turn_offset(zero: f64, centre: f64, position: f64) -> f64 {
    let turns = ((position - zero - centre) / std::f64::consts::TAU).round();
    zero + turns * std::f64::consts::TAU
}

/// The bridge on a serial port: what the robot runs.
pub type SerialBridgeIo = BridgeIo<Box<dyn serialport::SerialPort>>;

impl BridgeIo<Box<dyn serialport::SerialPort>> {
    /// Open the bridge on a serial port and confirm it speaks this protocol version.
    pub fn open(path: &str, baud: u32, config: BridgeConfig) -> Result<Self> {
        let port = serialport::new(path, baud)
            .timeout(PORT_READ_TIMEOUT)
            .open()
            .map_err(|e| IoError::Port {
                path: path.to_owned(),
                source: std::io::Error::other(e),
            })?;
        Self::with_transport(port, config)
    }
}

impl<T: Read + Write> BridgeIo<T> {
    pub fn with_transport(port: T, config: BridgeConfig) -> Result<Self> {
        let mut io = Self {
            port,
            decoder: FrameDecoder::new(),
            out: vec![0; proto::MAX_FRAME],
            config,
            epoch: Instant::now(),
            info: proto::Info::default(),
            command: proto::Command::default(),
            torque_on: false,
            targets_set: false,
            kp: 0.0,
            last_full: None,
            state_due: Instant::now(),
            last_state: None,
            last_sample: None,
            flags: 0,
            imu_status: proto::imu_status::STARTING,
            imu: SflpDecoder::new(config.imu_mount),
            stale_imu: ImuStale::default(),
            last_rotation_samples: None,
            offsets: match config.zero_counts {
                None => [Some(0.0); NUM_JOINTS],
                Some(_) => [None; NUM_JOINTS],
            },
        };
        io.info = io.request_info()?;
        Ok(io)
    }

    /// What the bridge said it is when it was opened.
    pub fn info(&self) -> &proto::Info {
        &self.info
    }

    /// The last full state frame, with everything the bridge reports and [`Sensors`] has no room
    /// for. About a second old at most once writes are going.
    pub fn last_state(&self) -> Option<&proto::State> {
        self.last_state.as_ref()
    }

    /// The last sample, what the last read was made from.
    pub fn last_sample(&self) -> Option<&proto::Sample> {
        self.last_sample.as_ref()
    }

    /// The sequence number of the last command sent, as [`proto::State::command_seq`] echoes it.
    pub fn command_seq(&self) -> u32 {
        self.command.seq
    }

    fn host_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    fn send<M: Message>(&mut self, message: &M) -> Result<()> {
        let len = message
            .encode(&mut self.out)
            .map_err(|e| IoError::Bus(format!("encode {:?}: {e:?}", M::KIND)))?;
        self.port
            .write_all(&self.out[..len])
            .map_err(|e| IoError::Bus(format!("bridge write: {e}")))
    }

    /// Reads until a frame of `M`'s kind that `accept` takes arrives, or `timeout` passes. A full
    /// state frame met on the way, the answer to a request a write made, is kept; other frames,
    /// and ones `accept` refuses, are dropped.
    fn receive<M: Message>(&mut self, timeout: Duration, accept: impl Fn(&M) -> bool) -> Result<M> {
        let deadline = Instant::now() + timeout;
        let mut chunk = [0u8; 1024];
        loop {
            while let Some(frame) = self.decoder.next_frame() {
                if frame.kind == proto::Kind::State && M::KIND != proto::Kind::State {
                    if let Ok(state) = proto::State::decode(&frame) {
                        self.flags = state.flags;
                        self.imu_status = state.imu.status;
                        self.last_state = Some(state);
                    }
                    continue;
                }
                if frame.kind == M::KIND {
                    let message = M::decode(&frame)
                        .map_err(|e| IoError::Bus(format!("bridge {:?} frame: {e:?}", M::KIND)))?;
                    if accept(&message) {
                        return Ok(message);
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(IoError::Bus(format!(
                    "no {:?} from the bridge within {timeout:?} ({} frames rejected so far)",
                    M::KIND,
                    self.decoder.errors
                )));
            }
            match self.port.read(&mut chunk) {
                Ok(n) => {
                    let mut taken = 0;
                    while taken < n {
                        taken += self.decoder.push(&chunk[taken..n]);
                        if taken < n {
                            // The decoder is full of bytes that are not a frame yet; let it
                            // discard what it can before pushing the rest.
                            while self.decoder.next_frame().is_some() {}
                        }
                    }
                }
                Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {}
                Err(e) => return Err(IoError::Bus(format!("bridge read: {e}"))),
            }
        }
    }

    fn request_info(&mut self) -> Result<proto::Info> {
        self.send(&proto::InfoRequest)?;
        let info = self
            .receive::<proto::Info>(INFO_TIMEOUT, |_| true)
            .map_err(|e| {
                // Frames rejected but nothing decoded is the signature of a bridge running another
                // protocol version: every frame fails the version check.
                if self.decoder.errors > 0 {
                    IoError::Bus(format!(
                        "{e}; is the bridge firmware on protocol {}?",
                        proto::PROTOCOL_VERSION
                    ))
                } else {
                    e
                }
            })?;
        tracing::info!(
            firmware = info.firmware_str(),
            link_baud = info.link_baud,
            watchdog_ms = info.watchdog_ms,
            servos = info
                .segments
                .iter()
                .filter(|&&s| s != proto::ABSENT)
                .count(),
            "servo bridge answered"
        );
        Ok(info)
    }

    /// One state request and its answer. A reply to an earlier request that arrived after its
    /// wait gave up carries that request's timestamp, and is skipped rather than mistaken for
    /// this one.
    pub fn state(&mut self) -> Result<proto::State> {
        let host_time_us = self.host_us();
        self.send(&proto::StateRequest { host_time_us })?;
        let state = self
            .receive::<proto::State>(REPLY_TIMEOUT, |s| s.request_host_time_us == host_time_us)?;
        self.flags = state.flags;
        self.imu_status = state.imu.status;
        self.last_state = Some(state);
        Ok(state)
    }

    /// One sample request and its answer, after a fresh pass over the servos. A late answer to
    /// an earlier request is skipped, as for [`Self::state`].
    pub fn sample(&mut self) -> Result<proto::Sample> {
        let host_time_us = self.host_us();
        self.send(&proto::SampleRequest { host_time_us })?;
        let sample = self
            .receive::<proto::Sample>(REPLY_TIMEOUT, |s| s.request_host_time_us == host_time_us)?;
        self.flags = sample.flags;
        self.imu_status = sample.imu.status;
        self.last_sample = Some(sample);
        Ok(sample)
    }

    /// Whether the bridge's shutdown button was held, from the last state frame. Stays set until
    /// [`Self::send_host_status`] reports the compute module shutting down.
    pub fn shutdown_requested(&self) -> bool {
        self.flags & proto::state_flags::SHUTDOWN_REQUESTED != 0
    }

    /// Tells the bridge what the compute module is doing and how hot it is, for its display.
    pub fn send_host_status(
        &mut self,
        state: proto::HeadState,
        cpu_temp_c: Option<f64>,
    ) -> Result<()> {
        let cpu_temp_dc = cpu_temp_c.map_or(proto::TEMP_UNKNOWN, |t| {
            (t * 10.0).round().clamp(-3276.7, 3276.7) as i16
        });
        self.send(&proto::HostStatus { state, cpu_temp_dc })
    }

    /// A servo's position as its joint's angle, choosing the joint's turn the first time.
    fn joint_angle(&mut self, joint: usize, position: f64) -> f64 {
        if let Some(offset) = self.offsets[joint] {
            return position - offset;
        }
        let zero = self.config.zero_counts.map_or(0.0, |z| {
            z[joint] as f64 * std::f64::consts::TAU / ENCODER_COUNTS as f64
        });
        let offset = turn_offset(zero, travel_centre(joint), position);
        tracing::info!(
            joint = JOINT_NAMES[joint],
            position,
            zero,
            offset,
            angle = position - offset,
            "joint turn chosen"
        );
        self.offsets[joint] = Some(offset);
        position - offset
    }

    /// Fills every servo from the current gains and torque state.
    fn fill_command(&mut self) {
        let drive = self.torque_on && self.targets_set;
        for s in &mut self.command.servos {
            s.mode = if drive {
                ServoMode::Foc
            } else {
                ServoMode::Stop
            };
            s.kp = self.kp as f32;
            s.kd = self.config.kd as f32;
            s.velocity = 0.0;
            s.torque = 0.0;
        }
    }

    /// Sends the full command: targets, modes, gains and actions.
    fn send_command(&mut self) -> Result<()> {
        self.fill_command();
        self.command.seq = self.command.seq.wrapping_add(1);
        self.command.host_time_us = self.host_us();
        let command = self.command;
        self.send(&command)?;
        self.last_full = Some((command, Instant::now()));
        Ok(())
    }

    /// Whether everything but the positions still matches the last full command, recently
    /// enough that targets alone will do.
    fn targets_suffice(&mut self) -> bool {
        self.fill_command();
        let Some((sent, at)) = &self.last_full else {
            return false;
        };
        let rest = |s: &proto::ServoCommand| {
            (
                s.mode,
                s.action,
                s.action_id,
                s.velocity,
                s.kp,
                s.kd,
                s.torque,
            )
        };
        at.elapsed() < FULL_COMMAND_EVERY
            && sent
                .servos
                .iter()
                .zip(&self.command.servos)
                .all(|(a, b)| rest(a) == rest(b))
    }

    /// Sends the targets alone; the bridge keeps the rest from the last full command.
    fn send_targets(&mut self) -> Result<()> {
        self.command.seq = self.command.seq.wrapping_add(1);
        self.command.host_time_us = self.host_us();
        let targets = proto::Targets {
            seq: self.command.seq,
            host_time_us: self.command.host_time_us,
            positions: self.command.servos.map(|s| s.position),
        };
        self.send(&targets)
    }

    /// Joint angles to servo positions. Refused until every joint has been read once, since
    /// before that a joint's turn is not known and its target could be a turn out.
    fn set_targets(&mut self, positions: &[f64; NUM_JOINTS]) -> Result<()> {
        let mut servo_positions = [0.0f32; NUM_JOINTS];
        for (j, &p) in positions.iter().enumerate() {
            let offset = self.offsets[j].ok_or_else(|| {
                IoError::Bus(format!(
                    "no target for {} before it has been read: its turn is not known",
                    JOINT_NAMES[j]
                ))
            })?;
            servo_positions[j] = (p + offset) as f32;
        }
        for (servo, p) in self.command.servos.iter_mut().zip(servo_positions) {
            servo.position = p;
        }
        self.targets_set = true;
        Ok(())
    }

    /// Present positions from a fresh sample.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        let sample = self.sample()?;
        Ok(self.sensors_from(&sample)?.positions)
    }

    /// Ramp every joint from where it is now to `target`, linearly. The same contract as
    /// `DynamixelIo::interpolate_to`: only an explicit `robotd init` calls it, and nothing else
    /// talks to the bridge while it runs.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64())
            .ceil()
            .max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let next: [f64; NUM_JOINTS] =
                std::array::from_fn(|j| start[j] + (target[j] - start[j]) * t);
            self.write(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }
}

impl<T: Read + Write> BridgeIo<T> {
    /// The servos in [`Sensors`]' shape, as joint angles, or why not. The IMU is the caller's.
    fn sensors_from(&mut self, sample: &proto::Sample) -> Result<Sensors> {
        let mut sensors = check_servos(sample)?;
        for j in 0..NUM_JOINTS {
            sensors.positions[j] = self.joint_angle(j, sensors.positions[j]);
        }
        Ok(sensors)
    }
}

/// The servos in [`Sensors`]' shape, in the servos' own frame, or why not. The IMU is the
/// caller's.
fn check_servos(sample: &proto::Sample) -> Result<Sensors> {
    let present = sample
        .servos
        .iter()
        .filter(|s| s.segment != proto::ABSENT)
        .count();
    if present != NUM_JOINTS {
        return Err(IoError::ShortRead {
            what: "servos on the bridge",
            expected: NUM_JOINTS,
            got: present,
        });
    }
    let mut sensors = Sensors::default();
    for (j, servo) in sample.servos.iter().enumerate() {
        if servo.age_us > MAX_SERVO_AGE_US {
            return Err(IoError::Bus(format!(
                "servo {j} last answered {} µs ago",
                servo.age_us
            )));
        }
        sensors.positions[j] = servo.position as f64;
        // The bridge's position difference, not the servo's own filtered speed: hardware.md §5.1.
        sensors.velocities[j] = servo.velocity as f64;
        sensors.currents_ma[j] = (servo.torque as f64 / TORQUE_PER_AMP).abs() * 1000.0;
    }
    Ok(sensors)
}

impl<T: Read + Write> RobotIo for BridgeIo<T> {
    fn read(&mut self) -> Result<Sensors> {
        let sample = self.sample()?;
        let mut sensors = self.sensors_from(&sample)?;
        // Stale by the bridge's own sample counter rather than by comparing bytes: the IMU runs
        // at 120 Hz, so at a 100 Hz tick two identical blocks are rare but not wrong, and the
        // counter says exactly whether a new fused sample arrived.
        let samples = sample.imu.rotation_samples;
        if self.last_rotation_samples == Some(samples) {
            self.stale_imu.total += 1;
            self.stale_imu.run += 1;
        } else {
            self.stale_imu.run = 0;
        }
        self.last_rotation_samples = Some(samples);
        sensors.imu = self.imu.decode(&sample.imu.block);
        Ok(sensors)
    }

    /// Targets alone when nothing else changed, the full command otherwise; then, about once a
    /// second, a full state request, whose answer arrives while the tick has nothing else on
    /// the link and is kept by the next read.
    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        self.set_targets(&targets.positions)?;
        if self.targets_suffice() {
            self.send_targets()?;
        } else {
            self.send_command()?;
        }
        if Instant::now() >= self.state_due {
            self.state_due = Instant::now() + FULL_STATE_EVERY;
            let host_time_us = self.host_us();
            self.send(&proto::StateRequest { host_time_us })?;
        }
        Ok(())
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        self.kp = kp as f64 * self.config.kp_per_gain;
        self.send_command()
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        if on && !self.targets_set {
            // Hold where the robot is: the J288 drives to whatever position the frame carries,
            // and there is no goal register left over from before as there is on an XL330.
            let here = self.present_positions()?;
            self.set_targets(&here)?;
        }
        self.torque_on = on;
        self.send_command()
    }

    /// A J288 has no reboot instruction; clearing its faults (mode 6) is what gets a latched
    /// servo going again. Its reset (mode 7) is not used, because it is not known whether it
    /// keeps the multi-turn position the joint calibration will rest on.
    fn reboot(&mut self, id: u8) -> Result<()> {
        let joint = JOINT_IDS
            .iter()
            .position(|&j| j == id)
            .ok_or_else(|| IoError::Bus(format!("reboot: {id} is not a joint id")))?;
        let servo = &mut self.command.servos[joint];
        servo.action = Action::ClearFaults;
        // 0 never triggers an action on the bridge, so the count skips it on wrapping.
        servo.action_id = servo.action_id.wrapping_add(1).max(1);
        self.send_command()
    }

    /// From the last full state frame, which writes ask for about once a second; a fresh one
    /// only if none has arrived yet. Voltage is the bridge's ADC once it measures, otherwise the
    /// servos' mean.
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        let state = match self.last_state {
            Some(s) => s,
            None => self.state()?,
        };
        let mut temps_c = [0.0; NUM_JOINTS];
        let mut volts = Vec::with_capacity(NUM_JOINTS);
        for (j, servo) in state.servos.iter().enumerate() {
            if servo.segment == proto::ABSENT {
                continue;
            }
            temps_c[j] = servo.temp_c as f64;
            if servo.volts_half > 0 {
                volts.push(servo.volts_half as f64 / 2.0);
            }
        }
        let volts = if state.pack_mv > 0 {
            state.pack_mv as f64 / 1000.0
        } else if volts.is_empty() {
            return Err(IoError::ShortRead {
                what: "servo supply voltage",
                expected: NUM_JOINTS,
                got: 0,
            });
        } else {
            volts.iter().sum::<f64>() / volts.len() as f64
        };
        Ok(SlowSensors { volts, temps_c })
    }

    fn imu_stale(&self) -> ImuStale {
        self.stale_imu
    }

    fn imu_ready(&self) -> bool {
        self.imu.ready() && self.imu_status == proto::imu_status::RUNNING
    }

    fn shutdown_requested(&self) -> bool {
        BridgeIo::shutdown_requested(self)
    }

    fn report_host(&mut self, state: HostState, cpu_temp_c: Option<f64>) -> Result<()> {
        let state = match state {
            HostState::Running => proto::HeadState::Running,
            HostState::ShuttingDown => proto::HeadState::ShuttingDown,
        };
        self.send_host_status(state, cpu_temp_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A bridge that answers like the firmware: info on request, a state or a sample echoing the
    /// request's time, and it records every command, merging targets into the last full one as
    /// the firmware does. Servo positions follow the last FOC command, as a servo that tracks
    /// perfectly would.
    struct FakeBridge {
        rx: FrameDecoder,
        tx: VecDeque<u8>,
        state: proto::State,
        commands: Vec<proto::Command>,
        /// How many of `commands` came as targets alone.
        targets: usize,
        /// Answers queued ahead of the next reply, to model a late one.
        inject: Vec<u8>,
    }

    impl FakeBridge {
        fn new() -> Self {
            let mut state = proto::State::default();
            for (j, s) in state.servos.iter_mut().enumerate() {
                s.segment = (j / 5) as u8;
                s.position = j as f32 * 0.1;
                s.velocity = 0.5;
                s.torque = -0.277;
                s.volts_half = 47;
                s.temp_c = 30 + j as i8;
                s.age_us = 2000;
            }
            state.imu.status = proto::imu_status::RUNNING;
            Self {
                rx: FrameDecoder::new(),
                tx: VecDeque::new(),
                state,
                commands: Vec::new(),
                targets: 0,
                inject: Vec::new(),
            }
        }

        /// The sample the firmware would build from `state`.
        fn sample(&self, request_host_time_us: u64) -> proto::Sample {
            let s = &self.state;
            proto::Sample {
                request_host_time_us,
                flags: s.flags,
                servos: s.servos.map(|v| proto::ServoSample {
                    segment: v.segment,
                    position: v.position,
                    velocity: v.velocity,
                    torque: v.torque,
                    age_us: v.age_us,
                }),
                imu: proto::ImuSample {
                    status: s.imu.status,
                    block: s.imu.block,
                    age_us: s.imu.age_us,
                    rotation_samples: s.imu.rotation_samples,
                },
                ..Default::default()
            }
        }

        fn apply(&mut self, c: proto::Command) {
            for (servo, cmd) in self.state.servos.iter_mut().zip(&c.servos) {
                if cmd.mode == ServoMode::Foc {
                    servo.position = cmd.position;
                }
            }
            self.commands.push(c);
        }

        fn reply<M: Message>(&mut self, m: &M) {
            let mut buf = [0u8; proto::MAX_FRAME];
            let n = m.encode(&mut buf).unwrap();
            self.tx.extend(self.inject.drain(..));
            self.tx.extend(&buf[..n]);
        }
    }

    impl Write for FakeBridge {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            enum Request {
                Info,
                State(proto::StateRequest),
                Sample(proto::SampleRequest),
                Command(Box<proto::Command>),
                Targets(proto::Targets),
            }
            self.rx.push(data);
            loop {
                let request = match self.rx.next_frame() {
                    None => break,
                    Some(frame) => match frame.kind {
                        proto::Kind::InfoRequest => Request::Info,
                        proto::Kind::StateRequest => {
                            Request::State(proto::StateRequest::decode(&frame).unwrap())
                        }
                        proto::Kind::SampleRequest => {
                            Request::Sample(proto::SampleRequest::decode(&frame).unwrap())
                        }
                        proto::Kind::Command => {
                            Request::Command(Box::new(proto::Command::decode(&frame).unwrap()))
                        }
                        proto::Kind::Targets => {
                            Request::Targets(proto::Targets::decode(&frame).unwrap())
                        }
                        _ => continue,
                    },
                };
                match request {
                    Request::Info => {
                        let info = proto::Info {
                            protocol_version: proto::PROTOCOL_VERSION,
                            ..Default::default()
                        };
                        self.reply(&info);
                    }
                    Request::State(req) => {
                        let mut s = self.state;
                        s.request_host_time_us = req.host_time_us;
                        self.reply(&s);
                    }
                    Request::Sample(req) => {
                        let sample = self.sample(req.host_time_us);
                        self.reply(&sample);
                    }
                    Request::Command(c) => self.apply(*c),
                    Request::Targets(t) => {
                        let mut c = self.commands.last().copied().unwrap_or_default();
                        c.seq = t.seq;
                        c.host_time_us = t.host_time_us;
                        for (servo, p) in c.servos.iter_mut().zip(t.positions) {
                            servo.position = p;
                        }
                        self.targets += 1;
                        self.apply(c);
                    }
                }
            }
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Read for FakeBridge {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.tx.is_empty() {
                return Err(ErrorKind::TimedOut.into());
            }
            let n = out.len().min(self.tx.len());
            for b in out.iter_mut().take(n) {
                *b = self.tx.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    fn open() -> BridgeIo<FakeBridge> {
        BridgeIo::with_transport(FakeBridge::new(), BridgeConfig::default()).unwrap()
    }

    /// Servo slot j is joint j, velocity is the bridge's, and torque becomes the load figure the
    /// rest of the stack reads as `currents_ma`: getting the slot mapping wrong would hand the
    /// policy one joint's angle as another's.
    #[test]
    fn a_read_maps_slots_to_joints() {
        let mut io = open();
        let s = io.read().unwrap();
        for j in 0..NUM_JOINTS {
            assert!((s.positions[j] - j as f64 * 0.1).abs() < 1e-6);
            assert!((s.velocities[j] - 0.5).abs() < 1e-6);
            assert!(
                (s.currents_ma[j] - 500.0).abs() < 0.5,
                "{}",
                s.currents_ma[j]
            );
        }
    }

    /// Torque off must command every servo to stop whatever targets are written, and torque on
    /// must drive them with the gain the stack set, scaled to J288 units.
    #[test]
    fn writes_drive_only_with_torque_on_and_use_the_scaled_gain() {
        let mut io = open();
        io.set_gain(200).unwrap();
        io.write(&JointTargets::new([0.3; NUM_JOINTS])).unwrap();
        let last = io.port.commands.last().unwrap();
        assert!(last.servos.iter().all(|s| s.mode == ServoMode::Stop));

        io.set_torque(true).unwrap();
        let last = io.port.commands.last().unwrap();
        assert!(last.servos.iter().all(|s| s.mode == ServoMode::Foc));
        assert!(
            last.servos
                .iter()
                .all(|s| (s.kp - 2.0).abs() < 1e-6 && (s.kd - 0.05).abs() < 1e-6)
        );
        assert!(last.servos.iter().all(|s| (s.position - 0.3).abs() < 1e-6));
    }

    /// Enabling torque before any target was written must hold the robot where it is. Driving to
    /// whatever the frame happened to carry would snap every joint to position zero.
    #[test]
    fn torque_on_without_targets_holds_the_present_pose() {
        let mut io = open();
        io.set_gain(200).unwrap();
        io.set_torque(true).unwrap();
        let last = io.port.commands.last().unwrap();
        for (j, s) in last.servos.iter().enumerate() {
            assert_eq!(s.mode, ServoMode::Foc);
            assert!((s.position - j as f32 * 0.1).abs() < 1e-6);
        }
    }

    /// One silent servo must fail the read, as one silent Dynamixel does: a partial sample would
    /// leave a stale angle in the observation with nothing to say so.
    #[test]
    fn a_missing_or_silent_servo_fails_the_read() {
        let mut io = open();
        io.port.state.servos[4].segment = proto::ABSENT;
        assert!(matches!(io.read(), Err(IoError::ShortRead { got: 14, .. })));
        io.port.state.servos[4].segment = 0;
        io.port.state.servos[7].age_us = 60_000;
        assert!(io.read().is_err());
    }

    /// Stale means no new fused sample, by the bridge's counter; a fresh one resets the run.
    #[test]
    fn imu_staleness_follows_the_sample_counter() {
        let mut io = open();
        io.read().unwrap();
        io.read().unwrap();
        io.read().unwrap();
        assert_eq!(io.imu_stale(), ImuStale { total: 2, run: 2 });
        io.port.state.imu.rotation_samples += 3;
        io.read().unwrap();
        assert_eq!(io.imu_stale(), ImuStale { total: 2, run: 0 });
    }

    /// The stack reboots by Dynamixel-era joint id; the bridge needs the slot, an action that
    /// runs once, and an action id that never lands on 0, which the bridge treats as "none yet".
    #[test]
    fn reboot_clears_faults_on_the_right_slot_once() {
        let mut io = open();
        io.reboot(JOINT_IDS[3]).unwrap();
        let first = io.port.commands.last().unwrap().servos[3];
        assert_eq!(first.action, Action::ClearFaults);
        assert_eq!(first.action_id, 1);
        io.command.servos[3].action_id = 255;
        io.reboot(JOINT_IDS[3]).unwrap();
        assert_eq!(io.port.commands.last().unwrap().servos[3].action_id, 1);
        assert!(io.reboot(99).is_err());
    }

    /// A reply that missed its own wait must not be taken as the answer to the next request,
    /// or every tick after one slow reply would read data one request old.
    #[test]
    fn a_late_reply_is_not_mistaken_for_the_current_one() {
        let mut io = open();
        let mut late = io.port.sample(u64::MAX);
        late.servos[0].position = 99.0;
        let mut buf = [0u8; proto::MAX_FRAME];
        let n = late.encode(&mut buf).unwrap();
        io.port.inject.extend_from_slice(&buf[..n]);
        let s = io.read().unwrap();
        assert!((s.positions[0] - 0.0).abs() < 1e-6);
    }

    /// Voltage comes from the servos until the bridge's ADC reports, then from the ADC, read
    /// from the full state frame a write asks for once a second and the next read keeps.
    #[test]
    fn slow_sensors_prefer_the_adc_once_it_reports() {
        let mut io = open();
        io.read().unwrap();
        let slow = io.slow_sensors().unwrap();
        assert!((slow.volts - 23.5).abs() < 1e-9);
        assert_eq!(slow.temps_c[2], 32.0);
        io.port.state.pack_mv = 24_150;
        io.state_due = Instant::now();
        io.write(&JointTargets::new([0.0; NUM_JOINTS])).unwrap();
        io.read().unwrap();
        assert!((io.slow_sensors().unwrap().volts - 24.15).abs() < 1e-9);
    }

    /// Once a full command has set modes and gains, a write sends targets alone, and a change to
    /// anything but the positions sends the full command again.
    #[test]
    fn writes_send_targets_alone_until_something_else_changes() {
        let mut io = open();
        io.set_gain(200).unwrap();
        io.set_torque(true).unwrap();
        let full = io.port.commands.len();
        io.write(&JointTargets::new([0.4; NUM_JOINTS])).unwrap();
        io.write(&JointTargets::new([0.5; NUM_JOINTS])).unwrap();
        assert_eq!(io.port.targets, 2);
        let last = io.port.commands.last().unwrap();
        assert!(
            last.servos
                .iter()
                .all(|s| s.mode == ServoMode::Foc && (s.position - 0.5).abs() < 1e-6)
        );
        assert!(last.servos.iter().all(|s| (s.kp - 2.0).abs() < 1e-6));
        io.set_gain(100).unwrap();
        io.write(&JointTargets::new([0.6; NUM_JOINTS])).unwrap();
        assert_eq!(
            io.port.targets, 3,
            "the gain change went as a full command, then targets"
        );
        assert_eq!(io.port.commands.len(), full + 4);
        assert!(
            io.port
                .commands
                .last()
                .unwrap()
                .servos
                .iter()
                .all(|s| (s.kp - 1.0).abs() < 1e-6)
        );
    }

    fn open_calibrated(zero_counts: [u16; NUM_JOINTS]) -> BridgeIo<FakeBridge> {
        let config = BridgeConfig {
            zero_counts: Some(zero_counts),
            ..BridgeConfig::default()
        };
        BridgeIo::with_transport(FakeBridge::new(), config).unwrap()
    }

    fn counts(rad: f64) -> u16 {
        ((rad / TAU * ENCODER_COUNTS as f64).round() as i64).rem_euclid(ENCODER_COUNTS as i64)
            as u16
    }

    use std::f64::consts::TAU;

    /// With zeros, a joint reads its servo's position less its zero, and a servo a whole turn
    /// away reads the same: the turn is chosen, not inherited from the servo's count.
    #[test]
    fn a_calibrated_read_is_the_position_less_the_zero() {
        // Every joint's zero 0.25 rad below where its servo sits (FakeBridge: j * 0.1).
        let zeros: [u16; NUM_JOINTS] = std::array::from_fn(|j| counts(j as f64 * 0.1 - 0.25));
        let mut io = open_calibrated(zeros);
        io.port.state.servos[3].position += TAU as f32;
        let s = io.read().unwrap();
        // 0.25 rad lies inside every joint's turn, so every joint reads it, joint 3 included.
        // The tolerance is the zero's rounding to a whole count.
        for (name, angle) in JOINT_NAMES.iter().zip(s.positions) {
            assert!((angle - 0.25).abs() < 2e-3, "{name} read {angle}");
        }
    }

    /// The turn a joint is read in is centred on its travel, so head yaw near either end reads
    /// near that end rather than a turn round.
    #[test]
    fn the_turn_is_centred_on_the_joint_travel() {
        let yaw = 7;
        assert_eq!(JOINT_NAMES[yaw], "head_yaw");
        for angle_deg in [165.0_f64, -172.0] {
            let mut io = open_calibrated([0; NUM_JOINTS]);
            // Zero at count 0: the servo reads the angle itself, within the encoder's turn.
            io.port.state.servos[yaw].position = angle_deg.to_radians().rem_euclid(TAU) as f32;
            let s = io.read().unwrap();
            assert!(
                (s.positions[yaw].to_degrees() - angle_deg).abs() < 0.01,
                "{angle_deg} read {}",
                s.positions[yaw].to_degrees()
            );
        }
    }

    /// Targets go back through the same offsets: holding the present pose commands exactly the
    /// servo positions read, and a joint angle becomes that angle plus the joint's offset.
    #[test]
    fn calibrated_targets_map_back_to_servo_positions() {
        let zeros: [u16; NUM_JOINTS] = std::array::from_fn(|j| counts(j as f64 * 0.1 - 0.25));
        let mut io = open_calibrated(zeros);
        io.port.state.servos[3].position += TAU as f32;
        io.set_gain(200).unwrap();
        io.set_torque(true).unwrap();
        let held = io.port.commands.last().unwrap();
        for (j, s) in held.servos.iter().enumerate() {
            let servo = j as f32 * 0.1 + if j == 3 { TAU as f32 } else { 0.0 };
            assert!(
                (s.position - servo).abs() < 1e-4,
                "{} held {}",
                JOINT_NAMES[j],
                s.position
            );
        }
        io.write(&JointTargets::new([0.35; NUM_JOINTS])).unwrap();
        let moved = io.port.commands.last().unwrap();
        for (j, s) in moved.servos.iter().enumerate() {
            let servo = j as f32 * 0.1 + if j == 3 { TAU as f32 } else { 0.0 };
            assert!(
                (s.position - (servo + 0.1)).abs() < 2e-3,
                "{} sent {}",
                JOINT_NAMES[j],
                s.position
            );
        }
    }

    /// Before a joint has been read its turn is unknown, so a calibrated bridge refuses a target
    /// rather than guess one that could be a turn out.
    #[test]
    fn a_calibrated_target_before_any_read_is_refused() {
        let mut io = open_calibrated([0; NUM_JOINTS]);
        assert!(io.write(&JointTargets::new([0.0; NUM_JOINTS])).is_err());
        io.read().unwrap();
        assert!(io.write(&JointTargets::new([0.0; NUM_JOINTS])).is_ok());
    }

    /// The shutdown flag follows the newest frame, so a full state frame from before the
    /// acknowledgement cannot raise it again after a sample has cleared it.
    #[test]
    fn the_shutdown_flag_follows_the_newest_frame() {
        let mut io = open();
        io.port.state.flags = proto::state_flags::SHUTDOWN_REQUESTED;
        io.read().unwrap();
        assert!(io.shutdown_requested());
        io.port.state.flags = 0;
        io.read().unwrap();
        assert!(!io.shutdown_requested());
    }
}
