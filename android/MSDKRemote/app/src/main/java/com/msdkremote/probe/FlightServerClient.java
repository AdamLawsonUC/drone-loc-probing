package com.msdkremote.probe;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Client for the drone-loc-probing flight server (scripts/80_flight_server.py).
 *
 *   GET  /health  -> server state, server clock and the flight envelope
 *   POST /act     -> multipart image + telemetry JSON, returns the action
 *
 * All calls block; run them off the UI thread.
 */
public class FlightServerClient
{
    private static final int CONNECT_TIMEOUT_MS = 3_000;
    private static final int READ_TIMEOUT_MS = 6_000;

    /** Reply of GET /health. */
    public static class Health
    {
        public String model = "?";
        public String session = "?";
        public double steerDelta = 0.0;
        public double serverTimeS = Double.NaN;
        public double floorM = Double.NaN;
        public double ceilingM = Double.NaN;
        public double maxVerticalMps = Double.NaN;
        public double commandRateHz = 1.0;
        public double holdOnStaleS = 3.0;
        public double targetAltitudeM = Double.NaN;
        /** 0 when the server never commands travel, turns or camera tilt. */
        public double maxHorizontalMps = 0.0;
        public double yawRateDps = 0.0;
        public double gimbalStepDeg = 0.0;
        public double gimbalMinDeg = -90.0;
        public double gimbalMaxDeg = 0.0;
        /** NaN when the server sets no geofence. */
        public double geofenceRadiusM = Double.NaN;
        public String actionVocab = "";
        public int taskCount = 0;
        public boolean tokenRequired = false;
        public long rttMs = 0;
    }

    /** Reply of POST /act. */
    public static class Act
    {
        public String action = "hold";
        /** What the model asked for, before the server's envelope. */
        public String chosen = "";
        public String taskId = "";
        public double verticalMps = 0.0;
        public double forwardMps = 0.0;
        public double rightMps = 0.0;
        public double yawDps = 0.0;
        public double gimbalDeltaDeg = 0.0;
        public String rawText = "";
        public String note = "";
        public double probeReadout = Double.NaN;
        public double steerDelta = 0.0;
        public double serverMs = Double.NaN;
        public long rttMs = 0;
    }

    private final String baseUrl;
    private final String token;

    /**
     * @param baseUrl e.g. "http://wolfgang.ise.canberra.edu.au:8765" (no trailing slash needed).
     * @param token   value of FLIGHT_TOKEN on the server, or empty.
     */
    public FlightServerClient(@NonNull String baseUrl, @Nullable String token)
    {
        String url = baseUrl.trim();
        while (url.endsWith("/"))
            url = url.substring(0, url.length() - 1);
        if (!url.startsWith("http://") && !url.startsWith("https://"))
            url = "http://" + url;

        this.baseUrl = url;
        this.token = (token == null) ? "" : token.trim();
    }

    @NonNull
    public String getBaseUrl() {
        return baseUrl;
    }

    @NonNull
    public Health health() throws IOException, JSONException
    {
        long t0 = System.currentTimeMillis();
        HttpURLConnection connection = open("/health");

        try {
            connection.setRequestMethod("GET");
            String body = readReply(connection);
            JSONObject json = new JSONObject(body);

            Health health = new Health();
            health.rttMs = System.currentTimeMillis() - t0;
            health.model = json.optString("model", "?");
            health.session = json.optString("session", "?");
            health.steerDelta = json.optDouble("steer_delta", 0.0);
            health.serverTimeS = json.optDouble("server_time", Double.NaN);
            health.targetAltitudeM = json.optDouble("target_altitude_m", Double.NaN);
            health.tokenRequired = json.optBoolean("token_required", false);
            health.taskCount = json.optInt("n_tasks", 0);
            health.actionVocab = String.valueOf(json.opt("action_vocab"));

            JSONObject envelope = json.optJSONObject("envelope");
            if (envelope != null) {
                health.floorM = envelope.optDouble("floor_m", Double.NaN);
                health.ceilingM = envelope.optDouble("ceiling_m", Double.NaN);
                health.maxVerticalMps = envelope.optDouble("max_vertical_mps", Double.NaN);
                health.commandRateHz = envelope.optDouble("command_rate_hz", 1.0);
                health.holdOnStaleS = envelope.optDouble("hold_on_stale_s", 3.0);
                health.maxHorizontalMps = envelope.optDouble("max_horizontal_mps", 0.0);
                health.yawRateDps = envelope.optDouble("yaw_rate_dps", 0.0);
                health.gimbalStepDeg = envelope.optDouble("gimbal_step_deg", 0.0);
                health.gimbalMinDeg = envelope.optDouble("gimbal_min_deg", -90.0);
                health.gimbalMaxDeg = envelope.optDouble("gimbal_max_deg", 0.0);
                health.geofenceRadiusM = envelope.optDouble("geofence_radius_m", Double.NaN);
            }
            return health;
        }
        finally {
            connection.disconnect();
        }
    }

    /**
     * Estimate (server clock - phone clock) in seconds from a few /health
     * round trips, keeping the sample with the shortest round trip.
     * The phone adds this to its own time so the "ts" it sends is on the
     * server clock, and the server's frame age and stale-frame guard work
     * even when the two clocks disagree.
     */
    public double clockOffsetSeconds(int samples) throws IOException, JSONException
    {
        double bestOffset = 0.0;
        long bestRtt = Long.MAX_VALUE;

        for (int i = 0; i < samples; i++) {
            long t0 = System.currentTimeMillis();
            Health health = health();
            long t1 = System.currentTimeMillis();

            if (Double.isNaN(health.serverTimeS))
                throw new IOException("Server /health has no server_time - update scripts/80_flight_server.py");

            long rtt = t1 - t0;
            if (rtt < bestRtt) {
                bestRtt = rtt;
                bestOffset = health.serverTimeS - (t0 + t1) / 2000.0;
            }
        }
        return bestOffset;
    }

    /**
     * Send one frame with its telemetry and get the action back.
     *
     * @param jpeg        camera frame as JPEG.
     * @param telemetry   JSON object; "ts" must be seconds on the server clock.
     * @param instruction optional instruction overriding the server's prompt.
     */
    @NonNull
    public Act act(@NonNull byte[] jpeg, @NonNull JSONObject telemetry, @Nullable String instruction)
            throws IOException, JSONException
    {
        String boundary = "----dlp" + System.nanoTime();
        ByteArrayOutputStream body = new ByteArrayOutputStream(jpeg.length + 1024);

        writeField(body, boundary, "telemetry", telemetry.toString());
        if (instruction != null && !instruction.trim().isEmpty())
            writeField(body, boundary, "instruction", instruction.trim());

        body.write(("--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
                + "Content-Type: image/jpeg\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        body.write(jpeg);
        body.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));

        byte[] payload = body.toByteArray();

        long t0 = System.currentTimeMillis();
        HttpURLConnection connection = open("/act");

        try {
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
            connection.setFixedLengthStreamingMode(payload.length);

            try (OutputStream out = connection.getOutputStream()) {
                out.write(payload);
            }

            JSONObject json = new JSONObject(readReply(connection));

            Act act = new Act();
            act.rttMs = System.currentTimeMillis() - t0;
            act.action = json.optString("action", "hold");
            act.chosen = json.optString("chosen", act.action);
            act.taskId = json.optString("task_id", "");
            act.verticalMps = finite(json.optDouble("vertical_mps", 0.0));
            act.forwardMps = finite(json.optDouble("forward_mps", 0.0));
            act.rightMps = finite(json.optDouble("right_mps", 0.0));
            act.yawDps = finite(json.optDouble("yaw_dps", 0.0));
            act.gimbalDeltaDeg = finite(json.optDouble("gimbal_delta_deg", 0.0));
            act.rawText = json.optString("raw_text", "");
            act.note = json.optString("note", "");
            act.probeReadout = json.optDouble("probe_readout", Double.NaN);
            act.steerDelta = json.optDouble("steer_delta", 0.0);
            act.serverMs = json.optDouble("server_ms", Double.NaN);

            return act;
        }
        finally {
            connection.disconnect();
        }
    }

    private static double finite(double value) {
        return Double.isFinite(value) ? value : 0.0;
    }

    private HttpURLConnection open(String path) throws IOException
    {
        HttpURLConnection connection = (HttpURLConnection) new URL(baseUrl + path).openConnection();
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setUseCaches(false);
        if (!token.isEmpty())
            connection.setRequestProperty("X-Flight-Token", token);
        return connection;
    }

    private static void writeField(ByteArrayOutputStream body, String boundary, String name, String value)
            throws IOException
    {
        body.write(("--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n"
                + value + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    private static String readReply(HttpURLConnection connection) throws IOException
    {
        int status = connection.getResponseCode();
        if (status / 100 != 2) {
            String error = readStream(connection.getErrorStream());
            if (error.length() > 300)
                error = error.substring(0, 300);
            throw new IOException("Flight server HTTP " + status + ": " + error);
        }
        return readStream(connection.getInputStream());
    }

    private static String readStream(@Nullable InputStream stream) throws IOException
    {
        if (stream == null)
            return "";

        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            char[] buffer = new char[4096];
            int read;
            while ((read = reader.read(buffer)) != -1)
                builder.append(buffer, 0, read);
        }
        return builder.toString();
    }
}
