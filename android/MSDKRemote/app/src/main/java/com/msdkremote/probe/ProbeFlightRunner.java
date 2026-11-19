package com.msdkremote.probe;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.msdkremote.vla.FrameSnapshotter;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import dji.sdk.keyvalue.key.FlightControllerKey;
import dji.sdk.keyvalue.key.GimbalKey;
import dji.sdk.keyvalue.key.KeyTools;
import dji.sdk.keyvalue.key.RemoteControllerKey;
import dji.sdk.keyvalue.value.common.Attitude;
import dji.sdk.keyvalue.value.common.EmptyMsg;
import dji.sdk.keyvalue.value.common.LocationCoordinate3D;
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem;
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode;
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode;
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam;
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode;
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation;
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode;
import dji.sdk.keyvalue.value.gimbal.GimbalMode;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.KeyManager;
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager;
import dji.v5.manager.interfaces.IVirtualStickManager;

/**
 * Bounded-authority flight from the flight server:
 *
 *   camera frame + telemetry -> POST /act on the server -> one short move
 *
 * A move is a vertical speed, a body-frame horizontal speed, a turn rate
 * or a camera tilt, depending on what the server's config allows. This
 * runner never takes off and never lands: the pilot takes off, flies into
 * the envelope and presses Start.
 *
 * The pilot takes over by moving either stick on the remote controller,
 * by pressing Stop, or with EMERGENCY LAND. A stick input ends the run
 * and returns control within one stick period.
 *
 * Safety, all enforced here regardless of what the server says:
 *  - every speed is clamped to the envelope;
 *  - no climb at or above the ceiling, no descend at or below the floor;
 *  - geofence: horizontal travel needs a GPS position; once the drone is
 *    at the radius from where Start was pressed, any travel with an
 *    outward component is dropped;
 *  - a command is held for two loop periods, then decays to hover, so a
 *    dead link or a slow server means hover;
 *  - a reply older than the stale limit is ignored;
 *  - leaving the envelope or the fence by more than the margin, losing
 *    the aircraft link, losing the GPS position while travel is allowed,
 *    landing, or three failed requests in a row ends the run.
 *
 * With sendSticks == false nothing is sent to the aircraft; the loop
 * runs and logs what it would have commanded.
 */
public class ProbeFlightRunner
{
    private static final String TAG = ProbeFlightRunner.class.getSimpleName();

    private static final int FRAME_MAX_WIDTH = 640;
    private static final int FRAME_JPEG_QUALITY = 80;
    private static final long FRAME_TIMEOUT_MS = 3000;

    private static final long STICK_PERIOD_MS = 100;
    private static final long ACTION_TIMEOUT_S = 10;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private static final int CLOCK_SYNC_SAMPLES = 5;

    /** Stick deflection (of 660) that counts as the pilot taking over. */
    private static final int TAKEOVER_STICK_THRESHOLD = 120;
    private static final double GIMBAL_ROTATE_DURATION_S = 0.8;

    private static final double METRES_PER_DEG_LAT = 110_540.0;
    private static final double METRES_PER_DEG_LON_EQUATOR = 111_320.0;

    /** App-side limits. The server's envelope must lie inside the chosen preset. */
    public static class Envelope
    {
        public final String name;
        public final double floorM;
        public final double ceilingM;
        public final double maxVerticalMps;
        /** Leaving [floor - margin, ceiling + margin] ends the run. */
        public final double marginM;
        public final double maxHorizontalMps;
        public final double maxYawDps;
        /** Largest geofence radius the app accepts; 0 = no horizontal travel at all. */
        public final double maxRadiusM;
        /** Being this far beyond the fence ends the run. */
        public final double fenceMarginM;

        public Envelope(String name, double floorM, double ceilingM, double maxVerticalMps,
                        double marginM, double maxHorizontalMps, double maxYawDps,
                        double maxRadiusM, double fenceMarginM)
        {
            this.name = name;
            this.floorM = floorM;
            this.ceilingM = ceilingM;
            this.maxVerticalMps = maxVerticalMps;
            this.marginM = marginM;
            this.maxHorizontalMps = maxHorizontalMps;
            this.maxYawDps = maxYawDps;
            this.maxRadiusM = maxRadiusM;
            this.fenceMarginM = fenceMarginM;
        }

        @NonNull
        @Override
        public String toString()
        {
            return String.format(Locale.ENGLISH,
                    "%s: %.1f to %.1f m, up to %.1f m/s vertical, %.1f m/s horizontal, fence up to %.0f m",
                    name, floorM, ceilingM, maxVerticalMps, maxHorizontalMps, maxRadiusM);
        }
    }

    public static final Envelope INDOOR =
            new Envelope("indoor", 0.5, 2.5, 0.2, 0.4, 0.0, 30.0, 0.0, 0.0);
    public static final Envelope OUTDOOR =
            new Envelope("outdoor", 15.0, 100.0, 1.0, 5.0, 3.0, 45.0, 100.0, 15.0);

    public interface Listener {
        void onInfo(String message);
        /** One line per server round trip. */
        void onStep(String line);
        /** Run is over. error == null means a normal end. */
        void onFinished(String summary, @Nullable String error);
    }

    private final FlightServerClient client;
    private final FrameSnapshotter snapshotter;
    private final Envelope preset;
    private final boolean sendSticks;
    private final File logDir;
    private final Listener listener;

    private Thread thread = null;
    private Thread stickThread = null;

    private volatile boolean running = false;
    private volatile boolean emergency = false;
    private volatile boolean pilotTookOver = false;

    // Telemetry mirrored by key listeners (listener object = this).
    private volatile boolean droneConnected = false;
    private volatile boolean connectionKnown = false;
    private volatile boolean isFlying = false;
    private volatile double altitude = 0.0;
    private volatile double heading = 0.0;
    private volatile double gimbalPitch = Double.NaN;
    private volatile int satelliteCount = 0;
    @Nullable private volatile LocationCoordinate3D location = null;

    // Effective envelope for this run (the server's, checked against the preset).
    private volatile double floorM;
    private volatile double ceilingM;
    private volatile double maxVerticalMps;
    private volatile double maxHorizontalMps = 0.0;
    private volatile double maxYawDps = 0.0;
    private volatile double fenceRadiusM = 0.0;
    private volatile double gimbalMinDeg = -90.0;
    private volatile double gimbalMaxDeg = 0.0;

    // Where Start was pressed: the centre of the geofence.
    private volatile boolean hasHome = false;
    private volatile double homeLat = 0.0;
    private volatile double homeLon = 0.0;

    // Current command and the time it stops being valid.
    private volatile double cmdUp = 0.0;
    private volatile double cmdForward = 0.0;
    private volatile double cmdRight = 0.0;
    private volatile double cmdYaw = 0.0;
    private volatile long commandExpiryMs = 0;

    // What the stick thread actually sent last.
    private volatile double appliedUp = 0.0;
    private volatile double appliedForward = 0.0;
    private volatile double appliedRight = 0.0;
    private volatile double appliedYaw = 0.0;
    private volatile String fenceNote = "";

    // Gimbal: absolute rotation first; some firmware ignores it, then relative is used.
    private boolean gimbalUseRelative = false;
    private double gimbalPitchBeforeLastMove = Double.NaN;
    private double gimbalLastTarget = Double.NaN;

    private boolean virtualStickEnabled = false;

    public ProbeFlightRunner(@NonNull FlightServerClient client,
                             @NonNull FrameSnapshotter snapshotter,
                             @NonNull Envelope preset,
                             boolean sendSticks,
                             @NonNull File logDir,
                             @NonNull Listener listener)
    {
        this.client = client;
        this.snapshotter = snapshotter;
        this.preset = preset;
        this.sendSticks = sendSticks;
        this.logDir = logDir;
        this.listener = listener;

        this.floorM = preset.floorM;
        this.ceilingM = preset.ceilingM;
        this.maxVerticalMps = preset.maxVerticalMps;
    }

    public synchronized boolean isRunning() {
        return running;
    }

    /** @return false if a run is already in progress. */
    public synchronized boolean start()
    {
        if (running)
            return false;

        running = true;
        emergency = false;
        pilotTookOver = false;

        thread = new Thread(this::runFlight, "ProbeFlightRunner");
        thread.start();
        return true;
    }

    /** Stop: the drone hovers and control returns to the remote controller. */
    public synchronized void requestStop()
    {
        running = false;
        if (thread != null)
            thread.interrupt();
    }

    /**
     * An emergency landing was triggered from outside. Stop without
     * touching the sticks again, so the run cannot fight the landing.
     */
    public synchronized void notifyEmergency()
    {
        emergency = true;
        requestStop();
    }

    // ==================== Main loop ====================

    private void runFlight()
    {
        String endError = null;
        String endSummary = "Run stopped by pilot.";
        FileWriter log = null;

        registerTelemetryListeners();
        seedTelemetry();

        try {
            // 1. Server reachable, envelope agreed, clocks aligned.
            FlightServerClient.Health health;
            double clockOffsetS;
            try {
                health = client.health();
                checkEnvelope(health);
                clockOffsetS = client.clockOffsetSeconds(CLOCK_SYNC_SAMPLES);
            }
            catch (RunAbort e) {
                throw e;
            }
            catch (Exception e) {
                throw new RunAbort("Flight server not reachable at " + client.getBaseUrl()
                        + ": " + e.getMessage());
            }

            long periodMs = Math.max(500, Math.round(1000.0 / Math.max(0.1, health.commandRateHz)));
            long staleMs = Math.round(health.holdOnStaleS * 1000.0);

            listener.onInfo(String.format(Locale.ENGLISH,
                    "Server ok: policy %s, session %s, envelope %.1f to %.1f m at %.2f m/s, "
                            + "round trip %d ms, clock offset %+.2f s.",
                    health.model, health.session, floorM, ceilingM, maxVerticalMps,
                    health.rttMs, clockOffsetS));
            listener.onInfo(String.format(Locale.ENGLISH,
                    "Authority: travel %.1f m/s inside %.0f m, turn %.0f deg/s, camera tilt %s, %d task(s).",
                    maxHorizontalMps, fenceRadiusM, maxYawDps,
                    health.gimbalStepDeg > 0 ? "yes" : "no", health.taskCount));

            // 2. Home point for the geofence and for the distance telemetry.
            LocationCoordinate3D start = location;
            if (hasGpsFix(start)) {
                homeLat = start.getLatitude();
                homeLon = start.getLongitude();
                hasHome = true;
            }

            // 3. Aircraft state.
            if (sendSticks) {
                if (!droneConnected)
                    throw new RunAbort("Drone not connected.");
                if (!isFlying)
                    throw new RunAbort("Drone is not flying. Take off and climb into the "
                            + "envelope by hand first.");
                if (altitude < floorM || altitude > ceilingM)
                    throw new RunAbort(String.format(Locale.ENGLISH,
                            "Altitude %.1f m is outside the envelope (%.1f to %.1f m). "
                                    + "Fly into it by hand, then start.", altitude, floorM, ceilingM));
                if (maxHorizontalMps > 0 && !hasHome)
                    throw new RunAbort("No GPS position, so there can be no geofence. Wait for "
                            + "a GPS lock, or use a config without horizontal travel.");

                if (health.gimbalStepDeg > 0)
                    setGimbalModeQuietly();

                registerTakeoverListeners();
                enableVirtualStick();
                startStickThread();
                listener.onInfo("Virtual stick enabled: the server is flying. Move a stick on the "
                        + "controller, or press Stop, to take over.");
            }
            else {
                listener.onInfo("Observe mode: nothing is sent to the aircraft.");
            }

            log = openLog(health.session);

            // 4. Loop.
            int failures = 0;
            long seq = 0;
            long runStartMs = System.currentTimeMillis();

            while (running) {
                long loopStart = System.currentTimeMillis();
                seq++;

                if (pilotTookOver)
                    break;

                if (connectionKnown && !droneConnected && sendSticks)
                    throw new RunAbort("Drone disconnected.");

                if (sendSticks) {
                    if (!isFlying)
                        throw new RunAbort("Drone is no longer flying.");
                    if (altitude > ceilingM + preset.marginM || altitude < floorM - preset.marginM)
                        throw new RunAbort(String.format(Locale.ENGLISH,
                                "Altitude %.1f m left the envelope (%.1f to %.1f m).",
                                altitude, floorM, ceilingM));
                    if (maxHorizontalMps > 0) {
                        double dist = homeDistanceM();
                        if (Double.isNaN(dist))
                            throw new RunAbort("GPS position lost.");
                        if (dist > fenceRadiusM + preset.fenceMarginM)
                            throw new RunAbort(String.format(Locale.ENGLISH,
                                    "Drone is %.0f m from the start point, beyond the %.0f m fence.",
                                    dist, fenceRadiusM));
                    }
                }

                String action = "hold";
                String chosen = "";
                String taskId = "";
                String note = "";
                double serverMs = Double.NaN;
                double probe = Double.NaN;
                long rttMs = -1;
                String raw = "";

                long captureMs = System.currentTimeMillis();
                double altAtCapture = altitude;
                byte[] jpeg = snapshotter.captureJpeg(
                        FRAME_MAX_WIDTH, FRAME_JPEG_QUALITY, FRAME_TIMEOUT_MS);

                if (jpeg == null) {
                    failures++;
                    note = "no_frame";
                    setCommand(0, 0, 0, 0, periodMs);
                }
                else {
                    captureMs = System.currentTimeMillis();
                    altAtCapture = altitude;

                    try {
                        FlightServerClient.Act act = client.act(jpeg,
                                telemetryJson(seq, captureMs, clockOffsetS, runStartMs), null);

                        failures = 0;
                        action = act.action;
                        chosen = act.chosen;
                        taskId = act.taskId;
                        note = act.note;
                        serverMs = act.serverMs;
                        probe = act.probeReadout;
                        rttMs = act.rttMs;
                        raw = act.rawText;

                        long ageMs = System.currentTimeMillis() - captureMs;
                        if (ageMs > staleMs) {
                            note = "late_reply";
                            setCommand(0, 0, 0, 0, periodMs);
                        }
                        else {
                            setCommand(act.verticalMps, act.forwardMps, act.rightMps,
                                    act.yawDps, periodMs);
                            if (sendSticks && act.gimbalDeltaDeg != 0.0)
                                tiltGimbal(act.gimbalDeltaDeg);
                        }
                    }
                    catch (Exception e) {
                        failures++;
                        note = "request_failed: " + e.getMessage();
                        setCommand(0, 0, 0, 0, periodMs);
                    }
                }

                if (failures >= MAX_CONSECUTIVE_FAILURES)
                    throw new RunAbort("Three failed requests in a row (" + note + ").");

                String fence = fenceNote;
                if (!fence.isEmpty())
                    note = note.isEmpty() ? fence : note + " " + fence;

                String shownAction = chosen.isEmpty() || chosen.equals(action)
                        ? action : chosen + ">" + action;

                listener.onStep(String.format(Locale.ENGLISH,
                        "#%d %s alt %.1f m  %s  rtt %d ms%s",
                        seq, taskId.isEmpty() ? "" : "[" + taskId + "]", altAtCapture, shownAction,
                        rttMs, note.isEmpty() ? "" : "  [" + note + "]"));

                writeLog(log, seq, captureMs, altAtCapture, taskId, chosen, action,
                        rttMs, serverMs, probe, note, raw);

                long sleepMs = periodMs - (System.currentTimeMillis() - loopStart);
                if (sleepMs > 0)
                    Thread.sleep(sleepMs);
            }

            if (pilotTookOver)
                endSummary = "Pilot took over with the sticks. You have control.";
        }
        catch (RunAbort e) {
            endError = e.getMessage();
        }
        catch (InterruptedException e) {
            endSummary = pilotTookOver
                    ? "Pilot took over with the sticks. You have control."
                    : "Run stopped by pilot.";
        }
        catch (Exception e) {
            Log.e(TAG, "Unexpected error", e);
            endError = "Unexpected error: " + e;
        }
        finally {
            cleanup();

            if (log != null) {
                try {
                    log.close();
                }
                catch (IOException ignored) { }
            }

            synchronized (this) {
                running = false;
                thread = null;
            }

            listener.onFinished(endError != null ? "Run aborted." : endSummary, endError);
        }
    }

    /** The server's envelope must sit inside the preset chosen in the app. */
    private void checkEnvelope(FlightServerClient.Health health) throws RunAbort
    {
        if (Double.isNaN(health.floorM) || Double.isNaN(health.ceilingM)
                || Double.isNaN(health.maxVerticalMps))
            throw new RunAbort("Server /health reports no envelope - update "
                    + "scripts/80_flight_server.py on the server.");

        boolean travel = health.maxHorizontalMps > 0;
        double radius = Double.isNaN(health.geofenceRadiusM) ? 0.0 : health.geofenceRadiusM;

        boolean inside = health.floorM >= preset.floorM - 1e-6
                && health.ceilingM <= preset.ceilingM + 1e-6
                && health.maxVerticalMps <= preset.maxVerticalMps + 1e-6
                && health.floorM < health.ceilingM
                && health.maxHorizontalMps <= preset.maxHorizontalMps + 1e-6
                && health.yawRateDps <= preset.maxYawDps + 1e-6
                && (!travel || (radius > 0 && radius <= preset.maxRadiusM + 1e-6));

        if (!inside)
            throw new RunAbort(String.format(Locale.ENGLISH,
                    "Envelope mismatch. Server: %.1f to %.1f m, %.2f m/s vertical, %.1f m/s "
                            + "horizontal, fence %.0f m, turn %.0f deg/s. App preset %s. Pick the "
                            + "other preset, or start the server with a matching config.",
                    health.floorM, health.ceilingM, health.maxVerticalMps, health.maxHorizontalMps,
                    radius, health.yawRateDps, preset));

        floorM = health.floorM;
        ceilingM = health.ceilingM;
        maxVerticalMps = health.maxVerticalMps;
        maxHorizontalMps = health.maxHorizontalMps;
        maxYawDps = health.yawRateDps;
        fenceRadiusM = radius;
        gimbalMinDeg = health.gimbalMinDeg;
        gimbalMaxDeg = health.gimbalMaxDeg;
    }

    private JSONObject telemetryJson(long seq, long captureMs, double clockOffsetS, long runStartMs)
            throws Exception
    {
        JSONObject json = new JSONObject();
        json.put("ts", captureMs / 1000.0 + clockOffsetS);   // server clock
        json.put("t_phone_ms", captureMs);
        json.put("run_t", (captureMs - runStartMs) / 1000.0);
        json.put("seq", seq);
        json.put("rel_alt", altitude);
        json.put("yaw", heading);
        json.put("flying", isFlying);
        json.put("sats", satelliteCount);
        json.put("sticks", sendSticks);
        json.put("applied_mps", appliedUp);
        json.put("applied_forward_mps", appliedForward);
        json.put("applied_right_mps", appliedRight);
        json.put("applied_yaw_dps", appliedYaw);
        json.put("envelope", preset.name);

        double pitch = gimbalPitch;
        if (Double.isFinite(pitch))
            json.put("gimbal_pitch", pitch);

        LocationCoordinate3D loc = location;
        if (hasGpsFix(loc)) {
            json.put("lat", loc.getLatitude());
            json.put("lon", loc.getLongitude());

            if (hasHome) {
                json.put("home_dist_m", homeDistanceM());
                json.put("home_bearing_deg", homeBearingDeg());
            }
        }
        return json;
    }

    // ==================== Geofence geometry ====================

    /** Metres north (index 0) and east (index 1) of the start point, or null without a position. */
    @Nullable
    private double[] offsetFromHome()
    {
        LocationCoordinate3D loc = location;
        if (!hasHome || !hasGpsFix(loc))
            return null;

        double north = (loc.getLatitude() - homeLat) * METRES_PER_DEG_LAT;
        double east = (loc.getLongitude() - homeLon) * METRES_PER_DEG_LON_EQUATOR
                * Math.cos(Math.toRadians(homeLat));
        return new double[] { north, east };
    }

    private double homeDistanceM()
    {
        double[] offset = offsetFromHome();
        return (offset == null) ? Double.NaN : Math.hypot(offset[0], offset[1]);
    }

    /** Compass bearing from the start point to the drone. */
    private double homeBearingDeg()
    {
        double[] offset = offsetFromHome();
        return (offset == null) ? Double.NaN : Math.toDegrees(Math.atan2(offset[1], offset[0]));
    }

    // ==================== Command handling ====================

    private static double finite(double value) {
        return Double.isFinite(value) ? value : 0.0;
    }

    private static double clamp(double value, double limit) {
        return Math.max(-limit, Math.min(limit, value));
    }

    private void setCommand(double up, double forward, double right, double yaw, long periodMs)
    {
        cmdUp = finite(up);
        cmdForward = finite(forward);
        cmdRight = finite(right);
        cmdYaw = finite(yaw);
        commandExpiryMs = System.currentTimeMillis() + 2 * periodMs;
    }

    /** Apply the envelope and the geofence to the current command and send it. */
    private void sendClampedCommand()
    {
        boolean expired = System.currentTimeMillis() > commandExpiryMs;

        double up = expired ? 0.0 : clamp(cmdUp, maxVerticalMps);
        double forward = expired ? 0.0 : clamp(cmdForward, maxHorizontalMps);
        double right = expired ? 0.0 : clamp(cmdRight, maxHorizontalMps);
        double yaw = expired ? 0.0 : clamp(cmdYaw, maxYawDps);
        String note = "";

        if (up > 0 && altitude >= ceilingM)
            up = 0.0;
        if (up < 0 && altitude <= floorM)
            up = 0.0;

        if (forward != 0.0 || right != 0.0) {
            double[] offset = offsetFromHome();

            if (maxHorizontalMps <= 0 || offset == null) {
                forward = 0.0;
                right = 0.0;
                note = "no_position";
            }
            else {
                double dist = Math.hypot(offset[0], offset[1]);
                if (dist >= fenceRadiusM) {
                    // Velocity in the north/east frame; heading is clockwise from north,
                    // "right" points 90 degrees clockwise of "forward".
                    double h = Math.toRadians(heading);
                    double north = forward * Math.cos(h) - right * Math.sin(h);
                    double east = forward * Math.sin(h) + right * Math.cos(h);
                    double outward = (north * offset[0] + east * offset[1]) / Math.max(dist, 1e-6);

                    if (outward > 0) {
                        forward = 0.0;
                        right = 0.0;
                        note = "app_fence";
                    }
                }
            }
        }

        appliedUp = up;
        appliedForward = forward;
        appliedRight = right;
        appliedYaw = yaw;
        fenceNote = note;

        sendVelocity(forward, right, up, yaw);
    }

    private void startStickThread()
    {
        stickThread = new Thread(() -> {
            try {
                while (running && !emergency && !pilotTookOver) {
                    sendClampedCommand();
                    Thread.sleep(STICK_PERIOD_MS);
                }
            }
            catch (InterruptedException ignored) { }
            catch (Exception e) {
                Log.e(TAG, "Stick thread failed", e);
            }
        }, "ProbeFlightSticks");
        stickThread.start();
    }

    /**
     * Send one body-frame velocity setpoint.
     * DJI naming: roll = forward velocity, pitch = rightward velocity,
     * yaw = clockwise angular velocity, verticalThrottle = upward velocity.
     */
    private void sendVelocity(double forward, double right, double up, double yawRate)
    {
        VirtualStickFlightControlParam param = new VirtualStickFlightControlParam();
        param.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        param.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        param.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);
        param.setVerticalControlMode(VerticalControlMode.VELOCITY);
        param.setRoll(forward);
        param.setPitch(right);
        param.setYaw(yawRate);
        param.setVerticalThrottle(up);

        VirtualStickManager.getInstance().sendVirtualStickAdvancedParam(param);
    }

    // ==================== Pilot takeover ====================

    /** Any clear stick deflection on the remote controller ends the run and returns control. */
    private void registerTakeoverListeners()
    {
        KeyManager.getInstance().listen(
                KeyTools.createKey(RemoteControllerKey.KeyStickLeftHorizontal), this,
                (oldValue, newValue) -> onStick(newValue));
        KeyManager.getInstance().listen(
                KeyTools.createKey(RemoteControllerKey.KeyStickLeftVertical), this,
                (oldValue, newValue) -> onStick(newValue));
        KeyManager.getInstance().listen(
                KeyTools.createKey(RemoteControllerKey.KeyStickRightHorizontal), this,
                (oldValue, newValue) -> onStick(newValue));
        KeyManager.getInstance().listen(
                KeyTools.createKey(RemoteControllerKey.KeyStickRightVertical), this,
                (oldValue, newValue) -> onStick(newValue));
    }

    private void onStick(@Nullable Integer value)
    {
        if (value != null && Math.abs(value) > TAKEOVER_STICK_THRESHOLD)
            pilotTakeover();
    }

    private void pilotTakeover()
    {
        if (!running || pilotTookOver || !sendSticks)
            return;

        pilotTookOver = true;

        // Hand control back straight away; the run thread tidies up afterwards.
        disableVirtualStick();
        requestStop();
    }

    // ==================== Gimbal ====================

    /** FPV mode ignores pitch commands; make sure the gimbal is in a pitch-controllable mode. */
    private void setGimbalModeQuietly()
    {
        KeyManager.getInstance().setValue(
                KeyTools.createKey(GimbalKey.KeyGimbalMode),
                GimbalMode.YAW_FOLLOW,
                new CommonCallbacks.CompletionCallback() {
                    @Override
                    public void onSuccess() { }

                    @Override
                    public void onFailure(@NonNull IDJIError idjiError) {
                        Log.w(TAG, "Setting gimbal mode failed: " + idjiError);
                    }
                });
    }

    /**
     * Tilt the camera by delta degrees (negative = down), clamped to the
     * envelope. Does not block: the result shows up in the gimbal pitch
     * telemetry of the following frames. If an absolute rotation did not
     * move the gimbal last time, relative rotations are used from then on.
     */
    private void tiltGimbal(double deltaDeg)
    {
        double current = gimbalPitch;
        if (!Double.isFinite(current))
            current = 0.0;

        if (!gimbalUseRelative && Double.isFinite(gimbalPitchBeforeLastMove)
                && Math.abs(gimbalLastTarget - gimbalPitchBeforeLastMove) > 5.0
                && Math.abs(current - gimbalPitchBeforeLastMove) < 2.0) {
            gimbalUseRelative = true;
            listener.onInfo("Gimbal ignored an absolute rotation - using relative rotations.");
        }

        double target = Math.max(gimbalMinDeg, Math.min(gimbalMaxDeg, current + deltaDeg));
        if (Math.abs(target - current) < 1.0)
            return;

        gimbalPitchBeforeLastMove = current;
        gimbalLastTarget = target;

        GimbalAngleRotation rotation = new GimbalAngleRotation();
        if (gimbalUseRelative) {
            rotation.setMode(GimbalAngleRotationMode.RELATIVE_ANGLE);
            rotation.setPitch(target - current);
        }
        else {
            rotation.setMode(GimbalAngleRotationMode.ABSOLUTE_ANGLE);
            rotation.setPitch(target);
        }
        rotation.setRollIgnored(true);
        rotation.setYawIgnored(true);
        rotation.setDuration(GIMBAL_ROTATE_DURATION_S);

        KeyManager.getInstance().performAction(
                KeyTools.createKey(GimbalKey.KeyRotateByAngle),
                rotation,
                new CommonCallbacks.CompletionCallbackWithParam<EmptyMsg>() {
                    @Override
                    public void onSuccess(EmptyMsg emptyMsg) { }

                    @Override
                    public void onFailure(@NonNull IDJIError idjiError) {
                        Log.w(TAG, "Gimbal rotation rejected: " + idjiError);
                    }
                });
    }

    // ==================== Virtual stick ====================

    private void enableVirtualStick() throws InterruptedException, RunAbort
    {
        IVirtualStickManager manager = VirtualStickManager.getInstance();

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> error = new AtomicReference<>(null);

        manager.enableVirtualStick(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onSuccess() {
                latch.countDown();
            }

            @Override
            public void onFailure(@NonNull IDJIError idjiError) {
                error.set(String.valueOf(idjiError));
                latch.countDown();
            }
        });

        if (!latch.await(ACTION_TIMEOUT_S, TimeUnit.SECONDS))
            throw new RunAbort("Enabling virtual stick timed out.");

        if (error.get() != null)
            throw new RunAbort("Could not enable virtual stick: " + error.get());

        manager.setVirtualStickAdvancedModeEnabled(true);
        virtualStickEnabled = true;
    }

    private synchronized void disableVirtualStick()
    {
        if (!virtualStickEnabled)
            return;

        virtualStickEnabled = false;

        IVirtualStickManager manager = VirtualStickManager.getInstance();
        manager.setVirtualStickAdvancedModeEnabled(false);
        manager.disableVirtualStick(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onSuccess() {
                Log.i(TAG, "Virtual stick disabled, control returned to RC.");
            }

            @Override
            public void onFailure(@NonNull IDJIError idjiError) {
                Log.w(TAG, "Failed disabling virtual stick: " + idjiError);
            }
        });
    }

    private void cleanup()
    {
        Thread sticks = stickThread;
        stickThread = null;
        if (sticks != null) {
            sticks.interrupt();
            try {
                sticks.join(500);
            }
            catch (InterruptedException ignored) { }
        }

        // On emergency the activity owns the aircraft (landing), and after a
        // pilot takeover the pilot does: do not touch the sticks from here.
        if (!emergency && !pilotTookOver && virtualStickEnabled) {
            try {
                for (int i = 0; i < 3; i++)
                    sendVelocity(0, 0, 0, 0);
            }
            catch (Exception e) {
                Log.w(TAG, "Failed zeroing sticks on cleanup", e);
            }
        }
        if (!emergency)
            disableVirtualStick();

        appliedUp = 0.0;
        appliedForward = 0.0;
        appliedRight = 0.0;
        appliedYaw = 0.0;
        KeyManager.getInstance().cancelListen(this);
    }

    // ==================== Phone-side log ====================

    @Nullable
    private FileWriter openLog(String session)
    {
        try {
            if (!logDir.exists() && !logDir.mkdirs())
                throw new IOException("cannot create " + logDir);

            File file = new File(logDir, "probe_" + session + "_" + System.currentTimeMillis() + ".csv");
            FileWriter writer = new FileWriter(file);
            writer.write("t_phone_ms,seq,rel_alt,yaw,gimbal_pitch,lat,lon,sats,home_dist_m,task_id,chosen,"
                    + "action,applied_up_mps,applied_forward_mps,applied_right_mps,applied_yaw_dps,"
                    + "rtt_ms,server_ms,probe_readout,sticks,note,raw_text\n");
            listener.onInfo("Phone log: " + file.getAbsolutePath());
            return writer;
        }
        catch (IOException e) {
            listener.onInfo("No phone log (" + e.getMessage() + "); the server log is unaffected.");
            return null;
        }
    }

    private static String num(double value, String format)
    {
        return Double.isFinite(value) ? String.format(Locale.ENGLISH, format, value) : "";
    }

    private void writeLog(@Nullable FileWriter log, long seq, long captureMs, double alt,
                          String taskId, String chosen, String action, long rttMs,
                          double serverMs, double probe, String note, String raw)
    {
        if (log == null)
            return;

        LocationCoordinate3D loc = location;
        boolean fix = hasGpsFix(loc);

        try {
            log.write(String.format(Locale.ENGLISH,
                    "%d,%d,%.2f,%.1f,%s,%s,%s,%d,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.2f,%d,%s,%s,%b,%s,%s\n",
                    captureMs, seq, alt, heading,
                    num(gimbalPitch, "%.1f"),
                    fix ? String.format(Locale.ENGLISH, "%.7f", loc.getLatitude()) : "",
                    fix ? String.format(Locale.ENGLISH, "%.7f", loc.getLongitude()) : "",
                    satelliteCount,
                    num(homeDistanceM(), "%.1f"),
                    clean(taskId), clean(chosen), clean(action),
                    appliedUp, appliedForward, appliedRight, appliedYaw,
                    rttMs,
                    num(serverMs, "%.1f"),
                    num(probe, "%.4f"),
                    sendSticks, clean(note), clean(raw)));
            log.flush();
        }
        catch (IOException e) {
            Log.w(TAG, "Phone log write failed", e);
        }
    }

    private static String clean(@Nullable String text)
    {
        if (text == null)
            return "";
        return text.replace(',', ';').replace('\n', ' ').replace('\r', ' ');
    }

    // ==================== Telemetry ====================

    private static boolean hasGpsFix(@Nullable LocationCoordinate3D loc)
    {
        return loc != null && loc.getLatitude() != null && loc.getLongitude() != null
                && Double.isFinite(loc.getLatitude()) && Double.isFinite(loc.getLongitude())
                && !(loc.getLatitude() == 0.0 && loc.getLongitude() == 0.0);
    }

    private void seedTelemetry()
    {
        // A cached 'true' is trustworthy; a cached 'false' may just be an
        // empty cache, so it does not mark the connection state as known.
        Boolean connected = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyConnection), Boolean.FALSE);
        if (Boolean.TRUE.equals(connected)) {
            droneConnected = true;
            connectionKnown = true;
        }

        Boolean flying = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyIsFlying), Boolean.FALSE);
        isFlying = (flying != null && flying);

        Double cachedAltitude = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyAltitude), 0.0);
        altitude = (cachedAltitude != null) ? cachedAltitude : 0.0;

        Double cachedHeading = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyCompassHeading), 0.0);
        heading = (cachedHeading != null) ? cachedHeading : 0.0;

        location = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyAircraftLocation3D),
                (LocationCoordinate3D) null);

        Integer cachedSatellites = KeyManager.getInstance().getValue(
                KeyTools.createKey(FlightControllerKey.KeyGPSSatelliteCount), 0);
        satelliteCount = (cachedSatellites != null) ? cachedSatellites : 0;

        Attitude cachedGimbal = KeyManager.getInstance().getValue(
                KeyTools.createKey(GimbalKey.KeyGimbalAttitude), (Attitude) null);
        if (cachedGimbal != null && cachedGimbal.getPitch() != null
                && Double.isFinite(cachedGimbal.getPitch()))
            gimbalPitch = cachedGimbal.getPitch();
    }

    private void registerTelemetryListeners()
    {
        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyConnection),
                this,
                (oldValue, newValue) -> {
                    if (newValue != null) {
                        droneConnected = newValue;
                        connectionKnown = true;
                    }
                });

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyIsFlying),
                this,
                (oldValue, newValue) ->
                        isFlying = (newValue != null && newValue));

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyAltitude),
                this,
                (oldValue, newValue) ->
                        altitude = (newValue != null) ? newValue : 0.0);

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyCompassHeading),
                this,
                (oldValue, newValue) ->
                        heading = (newValue != null) ? newValue : 0.0);

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyAircraftLocation3D),
                this,
                (oldValue, newValue) -> {
                    if (newValue != null)
                        location = newValue;
                });

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyGPSSatelliteCount),
                this,
                (oldValue, newValue) ->
                        satelliteCount = (newValue != null) ? newValue : 0);

        KeyManager.getInstance().listen(
                KeyTools.createKey(GimbalKey.KeyGimbalAttitude),
                this,
                (oldValue, newValue) -> {
                    if (newValue != null && newValue.getPitch() != null
                            && Double.isFinite(newValue.getPitch()))
                        gimbalPitch = newValue.getPitch();
                });
    }

    /** Internal: ends the run with a user-readable error. */
    private static class RunAbort extends Exception {
        RunAbort(String message) {
            super(message);
        }
    }
}
