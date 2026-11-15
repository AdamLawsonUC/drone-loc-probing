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
import dji.sdk.keyvalue.value.common.Attitude;
import dji.sdk.keyvalue.value.common.LocationCoordinate3D;
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem;
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode;
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode;
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam;
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.KeyManager;
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager;
import dji.v5.manager.interfaces.IVirtualStickManager;

/**
 * Bounded-authority vertical control from the flight server:
 *
 *   camera frame + telemetry -> POST /act on the server -> vertical velocity
 *
 * The server only ever commands climb / hold / descend. This runner never
 * takes off, never lands and never commands horizontal motion or yaw: the
 * pilot takes off, positions the drone inside the envelope and presses
 * Start. Pressing Stop zeroes the command and returns control to the
 * remote controller.
 *
 * Safety, all enforced here regardless of what the server says:
 *  - vertical speed clamped to the envelope maximum;
 *  - no climb at or above the ceiling, no descend at or below the floor;
 *  - a command is only held for two loop periods, then it decays to hover,
 *    so a dead link or a slow server means hover, not a runaway;
 *  - a reply older than the stale limit is ignored;
 *  - leaving the envelope by more than the margin, losing the aircraft
 *    link, landing, or three failed requests in a row ends the run.
 *
 * With sendSticks == false (bench mode) nothing is sent to the aircraft;
 * the loop runs and logs what it would have commanded.
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

    /** App-side limits. The server's envelope must lie inside the chosen preset. */
    public static class Envelope
    {
        public final String name;
        public final double floorM;
        public final double ceilingM;
        public final double maxVerticalMps;
        /** Leaving [floor - margin, ceiling + margin] ends the run. */
        public final double marginM;

        public Envelope(String name, double floorM, double ceilingM,
                        double maxVerticalMps, double marginM)
        {
            this.name = name;
            this.floorM = floorM;
            this.ceilingM = ceilingM;
            this.maxVerticalMps = maxVerticalMps;
            this.marginM = marginM;
        }

        @NonNull
        @Override
        public String toString()
        {
            return String.format(Locale.ENGLISH, "%s: %.1f to %.1f m, max %.2f m/s",
                    name, floorM, ceilingM, maxVerticalMps);
        }
    }

    public static final Envelope INDOOR = new Envelope("indoor", 0.5, 2.5, 0.2, 0.4);
    public static final Envelope OUTDOOR = new Envelope("outdoor", 15.0, 60.0, 0.5, 3.0);

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

    // Current vertical command and the time it stops being valid.
    private volatile double commandMps = 0.0;
    private volatile long commandExpiryMs = 0;
    private volatile double appliedMps = 0.0;

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

            // 2. Aircraft state.
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

                enableVirtualStick();
                startStickThread();
                listener.onInfo("Virtual stick enabled: the server now commands the vertical axis.");
            }
            else {
                listener.onInfo("Bench mode: nothing is sent to the aircraft.");
            }

            log = openLog(health.session);

            // 3. Loop.
            int failures = 0;
            long seq = 0;

            while (running) {
                long loopStart = System.currentTimeMillis();
                seq++;

                if (connectionKnown && !droneConnected && sendSticks)
                    throw new RunAbort("Drone disconnected.");

                if (sendSticks) {
                    if (!isFlying)
                        throw new RunAbort("Drone is no longer flying.");
                    if (altitude > ceilingM + preset.marginM || altitude < floorM - preset.marginM)
                        throw new RunAbort(String.format(Locale.ENGLISH,
                                "Altitude %.1f m left the envelope (%.1f to %.1f m).",
                                altitude, floorM, ceilingM));
                }

                String action = "hold";
                String note = "";
                double serverMps = 0.0;
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
                    setCommand(0.0, periodMs);
                }
                else {
                    captureMs = System.currentTimeMillis();
                    altAtCapture = altitude;

                    try {
                        FlightServerClient.Act act = client.act(
                                jpeg, telemetryJson(seq, captureMs, clockOffsetS), null);

                        failures = 0;
                        action = act.action;
                        note = act.note;
                        serverMps = act.verticalMps;
                        serverMs = act.serverMs;
                        probe = act.probeReadout;
                        rttMs = act.rttMs;
                        raw = act.rawText;

                        long ageMs = System.currentTimeMillis() - captureMs;
                        if (ageMs > staleMs) {
                            note = "late_reply";
                            setCommand(0.0, periodMs);
                        }
                        else {
                            setCommand(serverMps, periodMs);
                        }
                    }
                    catch (Exception e) {
                        failures++;
                        note = "request_failed: " + e.getMessage();
                        setCommand(0.0, periodMs);
                    }
                }

                if (failures >= MAX_CONSECUTIVE_FAILURES)
                    throw new RunAbort("Three failed requests in a row (" + note + ").");

                double shown = sendSticks ? appliedMps : clampToEnvelope(commandMps);

                listener.onStep(String.format(Locale.ENGLISH,
                        "#%d alt %.1f m  %s  cmd %+.2f m/s  rtt %d ms%s",
                        seq, altAtCapture, action, shown, rttMs,
                        note.isEmpty() ? "" : "  [" + note + "]"));

                writeLog(log, seq, captureMs, altAtCapture, action, serverMps, shown,
                        rttMs, serverMs, probe, note, raw);

                long sleepMs = periodMs - (System.currentTimeMillis() - loopStart);
                if (sleepMs > 0)
                    Thread.sleep(sleepMs);
            }
        }
        catch (RunAbort e) {
            endError = e.getMessage();
        }
        catch (InterruptedException e) {
            endSummary = "Run stopped by pilot.";
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

        boolean inside = health.floorM >= preset.floorM - 1e-6
                && health.ceilingM <= preset.ceilingM + 1e-6
                && health.maxVerticalMps <= preset.maxVerticalMps + 1e-6
                && health.floorM < health.ceilingM;

        if (!inside)
            throw new RunAbort(String.format(Locale.ENGLISH,
                    "Envelope mismatch. Server: %.1f to %.1f m at %.2f m/s. App preset %s. "
                            + "Pick the other preset, or start the server with the matching "
                            + "config (configs/flight_indoor.yaml or configs/flight.yaml).",
                    health.floorM, health.ceilingM, health.maxVerticalMps, preset));

        floorM = health.floorM;
        ceilingM = health.ceilingM;
        maxVerticalMps = health.maxVerticalMps;
    }

    private JSONObject telemetryJson(long seq, long captureMs, double clockOffsetS) throws Exception
    {
        JSONObject json = new JSONObject();
        json.put("ts", captureMs / 1000.0 + clockOffsetS);   // server clock
        json.put("t_phone_ms", captureMs);
        json.put("seq", seq);
        json.put("rel_alt", altitude);
        json.put("yaw", heading);
        json.put("flying", isFlying);
        json.put("sats", satelliteCount);
        json.put("sticks", sendSticks);
        json.put("applied_mps", appliedMps);
        json.put("envelope", preset.name);

        double pitch = gimbalPitch;
        if (Double.isFinite(pitch))
            json.put("gimbal_pitch", pitch);

        LocationCoordinate3D loc = location;
        if (hasGpsFix(loc)) {
            json.put("lat", loc.getLatitude());
            json.put("lon", loc.getLongitude());
        }
        return json;
    }

    // ==================== Command handling ====================

    private void setCommand(double mps, long periodMs)
    {
        commandMps = Double.isFinite(mps) ? mps : 0.0;
        commandExpiryMs = System.currentTimeMillis() + 2 * periodMs;
    }

    /** Envelope applied to a vertical command at the current altitude. */
    private double clampToEnvelope(double mps)
    {
        double v = Math.max(-maxVerticalMps, Math.min(maxVerticalMps, mps));

        if (System.currentTimeMillis() > commandExpiryMs)
            v = 0.0;
        if (v > 0 && altitude >= ceilingM)
            v = 0.0;
        if (v < 0 && altitude <= floorM)
            v = 0.0;

        return v;
    }

    private void startStickThread()
    {
        stickThread = new Thread(() -> {
            try {
                while (running && !emergency) {
                    double v = clampToEnvelope(commandMps);
                    appliedMps = v;
                    sendVelocity(v);
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

    /** Vertical velocity only; horizontal velocity and yaw rate are always zero. */
    private void sendVelocity(double up)
    {
        VirtualStickFlightControlParam param = new VirtualStickFlightControlParam();
        param.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        param.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        param.setYawControlMode(YawControlMode.ANGULAR_VELOCITY);
        param.setVerticalControlMode(VerticalControlMode.VELOCITY);
        param.setRoll(0.0);
        param.setPitch(0.0);
        param.setYaw(0.0);
        param.setVerticalThrottle(up);

        VirtualStickManager.getInstance().sendVirtualStickAdvancedParam(param);
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

    private void disableVirtualStick()
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

        // On emergency the activity owns the aircraft (landing);
        // do not touch the sticks from here.
        if (!emergency && virtualStickEnabled) {
            try {
                for (int i = 0; i < 3; i++)
                    sendVelocity(0.0);
            }
            catch (Exception e) {
                Log.w(TAG, "Failed zeroing sticks on cleanup", e);
            }
            disableVirtualStick();
        }

        appliedMps = 0.0;
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
            writer.write("t_phone_ms,seq,rel_alt,yaw,gimbal_pitch,lat,lon,sats,action,server_mps,"
                    + "applied_mps,rtt_ms,server_ms,probe_readout,sticks,note,raw_text\n");
            listener.onInfo("Phone log: " + file.getAbsolutePath());
            return writer;
        }
        catch (IOException e) {
            listener.onInfo("No phone log (" + e.getMessage() + "); the server log is unaffected.");
            return null;
        }
    }

    private void writeLog(@Nullable FileWriter log, long seq, long captureMs, double alt,
                          String action, double serverMps, double applied, long rttMs,
                          double serverMs, double probe, String note, String raw)
    {
        if (log == null)
            return;

        LocationCoordinate3D loc = location;
        boolean fix = hasGpsFix(loc);

        try {
            log.write(String.format(Locale.ENGLISH,
                    "%d,%d,%.2f,%.1f,%s,%s,%s,%d,%s,%.3f,%.3f,%d,%s,%s,%b,%s,%s\n",
                    captureMs, seq, alt, heading,
                    Double.isFinite(gimbalPitch) ? String.format(Locale.ENGLISH, "%.1f", gimbalPitch) : "",
                    fix ? String.format(Locale.ENGLISH, "%.7f", loc.getLatitude()) : "",
                    fix ? String.format(Locale.ENGLISH, "%.7f", loc.getLongitude()) : "",
                    satelliteCount, action, serverMps, applied, rttMs,
                    Double.isFinite(serverMs) ? String.format(Locale.ENGLISH, "%.1f", serverMs) : "",
                    Double.isFinite(probe) ? String.format(Locale.ENGLISH, "%.4f", probe) : "",
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
