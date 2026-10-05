# The servo bridge protocol

How the compute module talks to the servo bridge on a J288 microduck: the link, the frames, what
each message carries, the timing and the watchdog. `duck-bridge-proto` is the one implementation
and is built by both ends; this page owns the reasoning, the crate owns the byte layout.

The bridge is an STM32G474 on a WeAct core board in the trunk. It owns the three J288 servo bus
segments (single-wire, 6 Mbps) and the trunk IMU (LSM6DSV16X with on-chip SFLP fusion), and its
firmware lives in the `dukki` repository (`src/bridge`). On this side, `duck-control`'s
`BridgeIo` implements `RobotIo` over the protocol, and robotd selects it with `[bus] backend =
"bridge"`. The hardware — pins, wiring, why there is a bridge at all — is `dukki`'s
`docs/hardware.md` §5.

## 1. The shape: the bridge runs freely, the compute module asks

The bridge polls every servo on its own schedule, about 450 times a second each, and always holds
the latest reply from each servo and the latest IMU sample. The compute module, once per control
tick:

1. sends a `StateRequest` and gets a `State` back, built from what the bridge already holds — the
   reply never waits on a servo;
2. runs the policy;
3. sends a `Command`, which the bridge stores and applies on its next pass over each segment.

The control rate is therefore the compute module's choice alone. `control.hz` can be 50 or 100 —
or anything the link budget below allows — with no change to the bridge or the protocol. The cost
is that a state is up to one bridge pass old (about 2 ms) and a command takes up to one pass to
reach a servo; both are measured and reported in every frame (§4), not guessed.

The alternative — one servo round per command, replying when it completes — makes the bridge's
bus time part of every tick and ties the control rate to the servo count. It was not chosen.

## 2. The link

USART1 on the bridge (PA9 TX, PA10 RX) to UART7 on the CM4-NANO-A header (pin 18 RX, pin 16 TX),
full duplex, 3.3 V both sides, 8N1, no flow control, **2 Mbps**. 168 MHz / 2 Mbps is an integer
divider on the bridge; 4 Mbps (also integer) is the planned step once the neck harness has been
checked on a scope. `[bus] bridge_baud` must match what the firmware runs.

The bridge receives into a circular DMA buffer. Its servo transactions run as polled register
loops with interrupts masked for about 120 µs each, and an interrupt-driven receiver would lose
link bytes in those windows; the DMA keeps receiving through them. It runs USART1 with the
hardware FIFO off, because DMA transmits with it on never reached the pin after boot.

On the compute module the vendor kernel receives UART7 in interrupt mode (its DMA receive lost
frames), with a 16-byte FIFO that leaves about 40 µs per interrupt at 2 Mbps. A boot service from
`dukki`'s `setup-bridge-link.sh` moves the interrupt off CPU 0, which takes nearly every other
one, to a little core of its own. On bench leads about 3 to 5 state requests in 6,000 went
unanswered at 100 Hz; `BridgeIo` reports each as a failed read, which robotd coasts over. `dukki`
hardware.md §5.5 has the measurements.

## 3. Frames

```
SYNC 0xD5 0x6B · version (1) · kind (1) · payload length (2, LE) · payload · CRC-32 (4, LE)
```

- **CRC-32** is the standard one (ISO-HDLC, as zlib and Ethernet use), over everything after the
  sync bytes. Any tool can check a captured frame.
- **Resynchronisation.** A receiver drops bytes up to the next sync pair, and a frame that fails
  its CRC costs only its first byte, so a sync pair inside a corrupt frame cannot hide the good
  frame after it. Bad frames are counted on both ends.
- **Version.** `PROTOCOL_VERSION` is bumped whenever a layout changes. A frame of another version
  is rejected, not misread. `BridgeIo::open` asks for `Info` first, so a bridge on another
  version fails the open with a message that says so, before anything is commanded.
- Fixed-length payloads, little-endian, `f32` for physical quantities.

## 4. Messages

| kind | direction | reply | sent |
|---|---|---|---|
| `Command` 0x01 | compute → bridge | none | every tick, after the policy |
| `StateRequest` 0x02 | compute → bridge | `State` 0x81 | every tick, at the start |
| `InfoRequest` 0x03 | compute → bridge | `Info` 0x83 | at open |

**Servo slots are servo IDs.** Every J288's ID is its joint's index (`dukki`'s `motor-setup.md`),
so slot `j` of a command or state is joint `j` of `JOINT_NAMES`. The bridge does not know which
segment a slot should be on; it reports which segment each ID answered on.

**Units** are output side and SI: rad, rad/s, N·m, N·m/rad, N·m·s/rad. Position is the J288's
multi-turn position relative to where it powered up. Joint zeros and directions belong to the
compute module and are a calibration step that does not exist yet.

### `Command` (367 bytes framed)

A sequence number, the compute module's clock, and per servo: mode (`Stop`, J288 mode 0, a
viscous brake; or `Foc`, mode 1), target position, target speed, kp, kd and feed-forward torque —
the J288's full hybrid control, not only position. Held until the next command.

Per servo there is also a one-shot **action** (clear faults, J288 mode 6; reset, mode 7) with an
`action_id`. The bridge performs the action when the id changes, and 0 never triggers, so a lost
or repeated frame neither skips nor repeats it. `BridgeIo` uses clear-faults for
`RobotIo::reboot`; reset is not used, because it is not known whether a J288 keeps its multi-turn
position across one.

### `State` (628 bytes framed)

- **Sequence and timing:** the bridge's own sequence number; the last command's sequence number
  (so the compute module sees when a command landed); the request's timestamp echoed (so a reply
  that arrives after its wait gave up is recognised and skipped, never taken as the next tick's);
  the bridge's clock; how long ago the last command arrived.
- **Flags:** `COMMANDED` (a command has arrived since the bridge started) and `WATCHDOG` (§5).
- **Per servo:** the segment it answered on, or absent; position; the servo's own filtered speed;
  a velocity the bridge computes from position differences over about 15 ms of polls (the one the
  policy gets, `dukki` hardware.md §5.1.1); torque; housing and winding temperature; supply
  voltage; mode; fault and warning bits; the 13-bit output-shaft encoder; how old the reply is;
  replies and failures counted.
- **IMU:** the 12-byte block `SflpDecoder` already reads (gyro at ±500 dps, SFLP game rotation as
  three half floats), unchanged from the `imu_to_dxl` board so the decoder is untouched; plus the
  chip's gravity and gyro-bias estimates, sample counters, FIFO overruns and the sample's age.
  `BridgeIo` judges staleness by the rotation-sample counter, not by comparing bytes.
- **Pack voltage** from the bridge's ADC in mV, 0 until it measures; `BridgeIo` falls back to the
  servos' mean supply voltage until then.
- **Bridge health:** passes over the segments, the last pass's duration, link frames rejected.

### `Info`

Protocol version, firmware version, system clock, link baud, watchdog period, and the segment
each servo ID was found on.

## 5. The watchdog

If no `Command` arrives for **100 ms** — five ticks at 50 Hz, ten at 100 Hz — the bridge stops
every servo (mode 0) and sets `WATCHDOG` in each state frame; the next command clears it. It
exists because the J288's own frame timeout can never fire on this robot: the bridge keeps every
servo's link alive by polling, so a crashed robotd or a cut link would otherwise leave the servos
holding the last command indefinitely.

**This conflicts with robotd's restart contract,** and the conflict is open. robotd is written so
that a restart mid-update leaves a standing robot standing: "the servos hold their last commanded
goal while this process is dead" (`adopt_startup_pose` in `robotd/src/main.rs`). With the watchdog as built, a robotd
restart stops every servo after 100 ms and the robot sags into the brake. The two options are a
watchdog that holds the last command (the Dynamixel behaviour, and the restart contract kept) or
one that stops (a dead robotd cannot leave a fallen robot pushing). Until it is settled the
firmware stops.

Before the first command since power-up, the bridge's servos follow its bench console's test modes
(stop by default), so a bridge with no compute module behaves as it did on the bench.

## 6. Link budget

Per tick: a state request (18 bytes) and command (367) one way, a state (628) the other. At 2 Mbps,
10 bits per byte:

| | 50 Hz | 100 Hz |
|---|---|---|
| compute → bridge | 1.9 ms of 20 ms | 1.9 ms of 10 ms |
| bridge → compute | 3.1 ms of 20 ms | 3.1 ms of 10 ms |
| state round trip | about 3.5 ms | about 3.5 ms |

The link is full duplex, so the two directions do not add. 100 Hz uses about a third of each
direction's time; 4 Mbps halves every figure. `BridgeIo` waits 8 ms for a state before failing the
read.

## 7. What is not here yet

- Joint zeros and directions (`dukki` hardware.md §12).
- The bridge's own pack-voltage ADC on PA0; the field is in the frame.
- Reflashing the bridge from the compute module over this same link (`dukki` hardware.md §5.5).
