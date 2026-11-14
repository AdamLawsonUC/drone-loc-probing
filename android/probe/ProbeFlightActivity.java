package com.msdkremote.probe;

import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.msdkremote.vla.FrameSnapshotter;

import java.io.File;
import java.util.Locale;

import dji.sdk.keyvalue.key.FlightControllerKey;
import dji.sdk.keyvalue.key.KeyTools;
import dji.sdk.keyvalue.value.common.ComponentIndexType;
import dji.sdk.keyvalue.value.common.EmptyMsg;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.KeyManager;
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager;
import dji.v5.manager.datacenter.camera.CameraStreamManager;
import dji.v5.manager.interfaces.ICameraStreamManager;

/**
 * Screen for the drone-loc-probing flight server link.
 *
 * Enter the server address, pick the envelope (indoor or outdoor), test the
 * connection, then start. With "Send stick commands" off the loop runs as a
 * bench test and nothing reaches the aircraft. The layout is built in code,
 * so this screen needs no layout or string resources.
 */
public class ProbeFlightActivity extends AppCompatActivity implements ProbeFlightRunner.Listener
{
    private static final String TAG = ProbeFlightActivity.class.getSimpleName();
    private static final String PREFS = "probe_flight";
    private static final String DEFAULT_SERVER = "http://wolfgang.ise.canberra.edu.au:8765";
    private static final int MAX_LOG_CHARS = 20_000;

    private final FrameSnapshotter snapshotter = new FrameSnapshotter();
    private ProbeFlightRunner runner = null;

    private volatile boolean emergencyLanding = false;

    private EditText editServer;
    private EditText editToken;
    private RadioButton radioIndoor;
    private CheckBox checkSticks;
    private Button buttonTest;
    private Button buttonStart;
    private Button buttonStop;
    private TextView textLog;
    private ScrollView scrollLog;
    private SurfaceView videoView;

    private final ICameraStreamManager.AvailableCameraUpdatedListener cameraListener =
            availableCameraList -> {
                if (!availableCameraList.contains(ComponentIndexType.LEFT_OR_MAIN))
                    return;

                Surface surface = videoView.getHolder().getSurface();

                ICameraStreamManager manager = CameraStreamManager.getInstance();
                manager.removeCameraStreamSurface(surface);
                manager.putCameraStreamSurface(
                        ComponentIndexType.LEFT_OR_MAIN,
                        surface,
                        videoView.getWidth(),
                        videoView.getHeight(),
                        ICameraStreamManager.ScaleType.CENTER_INSIDE
                );
            };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(buildLayout());
        setTitle("Probe flight (server link)");

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        editServer.setText(prefs.getString("server", DEFAULT_SERVER));
        editToken.setText(prefs.getString("token", ""));

        buttonTest.setOnClickListener(v -> testConnection());
        buttonStart.setOnClickListener(v -> confirmAndStart());
        buttonStop.setOnClickListener(v -> stopRun());

        setRunningUi(false);
        appendLog("1. Start the flight server on wolfgang.\n"
                + "2. Test connection.\n"
                + "3. Bench test with stick commands off.\n"
                + "4. Take off by hand, fly into the envelope, tick stick commands, Start.");
    }

    private View buildLayout()
    {
        int pad = (int) (12 * getResources().getDisplayMetrics().density);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        editServer = new EditText(this);
        editServer.setHint("Server, e.g. " + DEFAULT_SERVER);
        editServer.setSingleLine(true);
        editServer.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        root.addView(editServer, matchWrap());

        editToken = new EditText(this);
        editToken.setHint("Token (FLIGHT_TOKEN on the server, blank if not set)");
        editToken.setSingleLine(true);
        editToken.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        root.addView(editToken, matchWrap());

        RadioGroup group = new RadioGroup(this);
        group.setOrientation(RadioGroup.HORIZONTAL);
        radioIndoor = new RadioButton(this);
        radioIndoor.setId(View.generateViewId());
        radioIndoor.setText("Indoor (" + describe(ProbeFlightRunner.INDOOR) + ")");
        RadioButton radioOutdoor = new RadioButton(this);
        radioOutdoor.setId(View.generateViewId());
        radioOutdoor.setText("Outdoor (" + describe(ProbeFlightRunner.OUTDOOR) + ")");
        group.addView(radioIndoor);
        group.addView(radioOutdoor);
        group.check(radioIndoor.getId());
        root.addView(group, matchWrap());

        checkSticks = new CheckBox(this);
        checkSticks.setText("Send stick commands (off = bench test, nothing reaches the aircraft)");
        checkSticks.setChecked(false);
        root.addView(checkSticks, matchWrap());

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        buttonTest = new Button(this);
        buttonTest.setText("Test connection");
        buttonStart = new Button(this);
        buttonStart.setText("Start");
        buttonStop = new Button(this);
        buttonStop.setText("Stop (hover)");
        row.addView(buttonTest, weighted());
        row.addView(buttonStart, weighted());
        row.addView(buttonStop, weighted());
        root.addView(row, matchWrap());

        Button buttonEmergency = new Button(this);
        buttonEmergency.setText("EMERGENCY LAND");
        buttonEmergency.setTextColor(Color.WHITE);
        buttonEmergency.setBackgroundColor(Color.rgb(198, 40, 40));
        buttonEmergency.setTypeface(Typeface.DEFAULT_BOLD);
        buttonEmergency.setOnClickListener(v -> emergencyStop());
        root.addView(buttonEmergency, matchWrap());

        textLog = new TextView(this);
        textLog.setTypeface(Typeface.MONOSPACE);
        textLog.setTextSize(12);
        scrollLog = new ScrollView(this);
        scrollLog.addView(textLog);
        root.addView(scrollLog, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        videoView = new SurfaceView(this);
        root.addView(videoView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        return root;
    }

    private static String describe(ProbeFlightRunner.Envelope envelope)
    {
        return String.format(Locale.ENGLISH, "%.1f-%.1f m", envelope.floorM, envelope.ceilingM);
    }

    private static LinearLayout.LayoutParams matchWrap()
    {
        return new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private static LinearLayout.LayoutParams weighted()
    {
        return new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
    }

    @Override
    protected void onResume()
    {
        super.onResume();
        CameraStreamManager.getInstance().addAvailableCameraUpdatedListener(cameraListener);
        snapshotter.start();
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        // Leaving the screen ends the run: hover, control back to the RC.
        stopRun();
        snapshotter.stop();

        ICameraStreamManager manager = CameraStreamManager.getInstance();
        manager.removeAvailableCameraUpdatedListener(cameraListener);
        manager.removeCameraStreamSurface(videoView.getHolder().getSurface());
    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();
        KeyManager.getInstance().cancelListen(this);
    }

    // ==================== Actions ====================

    private FlightServerClient makeClient()
    {
        String server = editServer.getText().toString().trim();
        String token = editToken.getText().toString().trim();

        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString("server", server)
                .putString("token", token)
                .apply();

        return new FlightServerClient(server, token);
    }

    private void testConnection()
    {
        final FlightServerClient client = makeClient();
        appendLog("Testing " + client.getBaseUrl() + " ...");
        buttonTest.setEnabled(false);

        new Thread(() -> {
            String result;
            try {
                FlightServerClient.Health health = client.health();
                double offset = client.clockOffsetSeconds(5);
                result = String.format(Locale.ENGLISH,
                        "OK. Policy %s, session %s.\nServer envelope %.1f to %.1f m at %.2f m/s, "
                                + "target %.1f m.\nRound trip %d ms, clock offset %+.2f s, token %s.",
                        health.model, health.session, health.floorM, health.ceilingM,
                        health.maxVerticalMps, health.targetAltitudeM, health.rttMs, offset,
                        health.tokenRequired ? "required" : "not required");
            }
            catch (Exception e) {
                result = "FAILED: " + e.getMessage()
                        + "\nCheck the phone can reach the server (same network or VPN), the "
                        + "port, and that the server is running.";
            }

            final String message = result;
            runOnUiThread(() -> {
                appendLog(message);
                buttonTest.setEnabled(true);
            });
        }, "ProbeFlightTest").start();
    }

    private void confirmAndStart()
    {
        if (runner != null && runner.isRunning())
            return;

        final ProbeFlightRunner.Envelope preset =
                radioIndoor.isChecked() ? ProbeFlightRunner.INDOOR : ProbeFlightRunner.OUTDOOR;
        final boolean sticks = checkSticks.isChecked();

        if (!sticks) {
            startRun(preset, false);
            return;
        }

        new AlertDialog.Builder(this)
                .setTitle("Hand the vertical axis to the server?")
                .setMessage("Preset " + preset + ".\n\nThe drone must already be flying inside the "
                        + "envelope. The server will command climb, hold or descend. Stop returns "
                        + "control to the remote controller.")
                .setPositiveButton("Start", (dialog, which) -> startRun(preset, true))
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void startRun(ProbeFlightRunner.Envelope preset, boolean sticks)
    {
        emergencyLanding = false;

        File logDir = new File(getExternalFilesDir(null), "probe_flights");
        runner = new ProbeFlightRunner(makeClient(), snapshotter, preset, sticks, logDir, this);

        if (runner.start()) {
            setRunningUi(true);
            appendLog("--- run started (" + preset.name + (sticks ? ", sticks ON" : ", bench") + ") ---");
        }
    }

    private void stopRun()
    {
        if (runner != null && runner.isRunning()) {
            runner.requestStop();
            appendLog("Stop requested - drone hovers, control returns to the RC.");
        }
    }

    /**
     * Abort everything and land: stop the run without letting it touch the
     * sticks again, give control back to the remote controller, and start
     * an automatic landing (auto-confirming it if the aircraft asks).
     */
    private void emergencyStop()
    {
        appendLog(">>> EMERGENCY - landing <<<");
        emergencyLanding = true;

        if (runner != null)
            runner.notifyEmergency();

        VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false);
        VirtualStickManager.getInstance().disableVirtualStick(
                new CommonCallbacks.CompletionCallback() {
                    @Override
                    public void onSuccess() {
                        Log.i(TAG, "Emergency: control returned to RC.");
                    }

                    @Override
                    public void onFailure(@NonNull IDJIError error) {
                        Log.e(TAG, "Emergency: failed returning control: " + error);
                    }
                });

        KeyManager.getInstance().listen(
                KeyTools.createKey(FlightControllerKey.KeyIsLandingConfirmationNeeded),
                this,
                (oldValue, newValue) -> {
                    if (emergencyLanding && newValue != null && newValue)
                        confirmLanding();
                });

        KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyStartAutoLanding),
                new CommonCallbacks.CompletionCallbackWithParam<EmptyMsg>() {
                    @Override
                    public void onSuccess(EmptyMsg emptyMsg) {
                        runOnUiThread(() -> appendLog("Auto landing started."));
                    }

                    @Override
                    public void onFailure(@NonNull IDJIError error) {
                        runOnUiThread(() -> appendLog("Landing command failed: " + error
                                + " (drone may already be on the ground)"));
                    }
                });
    }

    private void confirmLanding()
    {
        KeyManager.getInstance().performAction(
                KeyTools.createKey(FlightControllerKey.KeyConfirmLanding),
                new CommonCallbacks.CompletionCallbackWithParam<EmptyMsg>() {
                    @Override
                    public void onSuccess(EmptyMsg emptyMsg) {
                        Log.i(TAG, "Emergency landing confirmed.");
                    }

                    @Override
                    public void onFailure(@NonNull IDJIError error) {
                        Log.w(TAG, "Landing confirmation failed: " + error);
                    }
                });
    }

    // ==================== Runner callbacks (background thread) ====================

    @Override
    public void onInfo(String message) {
        runOnUiThread(() -> appendLog(message));
    }

    @Override
    public void onStep(String line) {
        runOnUiThread(() -> appendLog(line));
    }

    @Override
    public void onFinished(String summary, @Nullable String error) {
        runOnUiThread(() -> {
            appendLog("--- " + summary + (error != null ? " " + error : "") + " ---");
            setRunningUi(false);
        });
    }

    // ==================== UI helpers ====================

    private void setRunningUi(boolean running)
    {
        buttonStart.setEnabled(!running);
        buttonTest.setEnabled(!running);
        buttonStop.setEnabled(running);
        editServer.setEnabled(!running);
        editToken.setEnabled(!running);
        checkSticks.setEnabled(!running);
    }

    private void appendLog(String line)
    {
        textLog.append(line + "\n");

        CharSequence text = textLog.getText();
        if (text.length() > MAX_LOG_CHARS)
            textLog.setText(text.subSequence(text.length() - MAX_LOG_CHARS / 2, text.length()));

        scrollLog.post(() -> scrollLog.fullScroll(View.FOCUS_DOWN));
    }
}
