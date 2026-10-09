//! The robot through the servo bridge: fifteen Unitree J288s and the trunk IMU behind an
//! STM32G474, over `duck-bridge-proto`.
//!
//! `docs/design/bridge-protocol.md` owns the protocol. In short: the bridge polls the servos on
//! its own schedule and holds the latest state, so a [`RobotIo::read`] is one small request and
//! one state frame, answered without waiting on a servo; a [`RobotIo::write`] is one command frame
//! with no reply. The control rate is therefore this side's choice alone.
//!
//! Each J288's ID is its joint's index (`motor-setup.md`), so servo slot `j` in the protocol is
//! joint `j` in [`crate::model::JOINT_NAMES`]. Positions pass through unchanged: they are the
//! J288's multi-turn position relative to where it powered up, and joint zeros and directions
//! are a calibration step that does not exist yet.

use std::io::{ErrorKind, Read, Write};
use std::time::{Duration, Instant};

use duck_bridge_proto::{self as proto, Action, FrameDecoder, Message, ServoMode};

use crate::imu::SflpDecoder;
use crate::io::{ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{JOINT_IDS, NUM_JOINTS};

/// What the bridge firmware runs today (`link.rs` there). 4 Mbps waits on a scope check of the
/// neck harness.
pub const DEFAULT_BAUD: u32 = 2_000_000;

/// A state frame is 628 bytes, 3.1 ms at 2 Mbps; the bridge answers from memory, so the rest is
/// scheduling on either side. Short enough to leave a 100 Hz tick most of its budget when the
/// bridge does not answer.
const REPLY_TIMEOUT: Duration = Duration::from_millis(8);
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
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            kp_per_gain: 0.01,
            kd: 0.05,
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
    last_state: Option<proto::State>,
    imu: SflpDecoder,
    stale_imu: ImuStale,
    last_rotation_samples: Option<u32>,
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
            last_state: None,
            imu: SflpDecoder::default(),
            stale_imu: ImuStale::default(),
            last_rotation_samples: None,
        };
        io.info = io.request_info()?;
        Ok(io)
    }

    /// What the bridge said it is when it was opened.
    pub fn info(&self) -> &proto::Info {
        &self.info
    }

    /// The last state frame, with everything the bridge reports and [`Sensors`] has no room for.
    pub fn last_state(&self) -> Option<&proto::State> {
        self.last_state.as_ref()
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

    /// Reads until a frame of `M`'s kind that `accept` takes arrives, or `timeout` passes.
    /// Frames of other kinds, and ones `accept` refuses, are dropped.
    fn receive<M: Message>(&mut self, timeout: Duration, accept: impl Fn(&M) -> bool) -> Result<M> {
        let deadline = Instant::now() + timeout;
        let mut chunk = [0u8; 1024];
        loop {
            while let Some(frame) = self.decoder.next_frame() {
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
        self.last_state = Some(state);
        Ok(state)
    }

    /// Whether the bridge's shutdown button was held, from the last state frame. Stays set until
    /// [`Self::send_host_status`] reports the compute module shutting down.
    pub fn shutdown_requested(&self) -> bool {
        self.last_state
            .is_some_and(|s| s.flags & proto::state_flags::SHUTDOWN_REQUESTED != 0)
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

    /// Fills every servo from the current targets, gains and torque state and sends it.
    fn send_command(&mut self) -> Result<()> {
        self.command.seq = self.command.seq.wrapping_add(1);
        self.command.host_time_us = self.host_us();
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
        let command = self.command;
        self.send(&command)
    }

    fn set_targets(&mut self, positions: &[f64; NUM_JOINTS]) {
        for (servo, &p) in self.command.servos.iter_mut().zip(positions) {
            servo.position = p as f32;
        }
        self.targets_set = true;
    }

    /// Present positions from a fresh state frame.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        Ok(sensors_from(&self.state()?)?.positions)
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

/// The servos and IMU in [`Sensors`]' shape, or why not.
fn sensors_from(state: &proto::State) -> Result<Sensors> {
    let present = state
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
    for (j, servo) in state.servos.iter().enumerate() {
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
        let state = self.state()?;
        let mut sensors = sensors_from(&state)?;
        // Stale by the bridge's own sample counter rather than by comparing bytes: the IMU runs
        // at 120 Hz, so at a 100 Hz tick two identical blocks are rare but not wrong, and the
        // counter says exactly whether a new fused sample arrived.
        let samples = state.imu.rotation_samples;
        if self.last_rotation_samples == Some(samples) {
            self.stale_imu.total += 1;
            self.stale_imu.run += 1;
        } else {
            self.stale_imu.run = 0;
        }
        self.last_rotation_samples = Some(samples);
        sensors.imu = self.imu.decode(&state.imu.block);
        Ok(sensors)
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        self.set_targets(&targets.positions);
        self.send_command()
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
            self.set_targets(&here);
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

    /// From the last state frame, which every tick fetches anyway; a fresh one only if no tick
    /// has run yet. Voltage is the bridge's ADC once it measures, otherwise the servos' mean.
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
        self.imu.ready()
            && self
                .last_state
                .is_some_and(|s| s.imu.status == proto::imu_status::RUNNING)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A bridge that answers like the firmware: info on request, a state echoing the request's
    /// time, and it records every command. Servo positions follow the last FOC command, as a
    /// servo that tracks perfectly would.
    struct FakeBridge {
        rx: FrameDecoder,
        tx: VecDeque<u8>,
        state: proto::State,
        commands: Vec<proto::Command>,
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
                inject: Vec::new(),
            }
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
                Command(Box<proto::Command>),
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
                        proto::Kind::Command => {
                            Request::Command(Box::new(proto::Command::decode(&frame).unwrap()))
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
                    Request::Command(c) => {
                        for (servo, cmd) in self.state.servos.iter_mut().zip(&c.servos) {
                            if cmd.mode == ServoMode::Foc {
                                servo.position = cmd.position;
                            }
                        }
                        self.commands.push(*c);
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
        let mut late = io.port.state;
        late.request_host_time_us = u64::MAX;
        late.servos[0].position = 99.0;
        let mut buf = [0u8; proto::MAX_FRAME];
        let n = late.encode(&mut buf).unwrap();
        io.port.inject.extend_from_slice(&buf[..n]);
        let s = io.read().unwrap();
        assert!((s.positions[0] - 0.0).abs() < 1e-6);
    }

    /// Voltage comes from the servos until the bridge's ADC reports, then from the ADC.
    #[test]
    fn slow_sensors_prefer_the_adc_once_it_reports() {
        let mut io = open();
        io.read().unwrap();
        let slow = io.slow_sensors().unwrap();
        assert!((slow.volts - 23.5).abs() < 1e-9);
        assert_eq!(slow.temps_c[2], 32.0);
        io.port.state.pack_mv = 24_150;
        io.read().unwrap();
        assert!((io.slow_sensors().unwrap().volts - 24.15).abs() < 1e-9);
    }
}
