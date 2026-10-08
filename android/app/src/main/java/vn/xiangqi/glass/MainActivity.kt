package vn.xiangqi.glass

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.KeyEvent
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import vn.xiangqi.glass.databinding.ActivityMainBinding
import java.util.Locale
import java.util.UUID

/**
 * Màn hình chính, dùng được trên kính AR (Android) lẫn điện thoại.
 *
 * Điều khiển:
 *  - Chạm màn hình / OK / Enter / phím camera / giảm âm lượng: chụp và phân tích
 *  - Tăng âm lượng: đổi lượt đi (tự đoán -> Đỏ -> Đen)
 *  - Lên: bật/tắt tự chụp mỗi vài giây
 *  - Xuống: ván mới
 *  - Nhấn giữ màn hình / phím Menu: đặt địa chỉ máy chủ
 */
class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private var tts: TextToSpeech? = null
    private var busy = false
    private var autoJob: Job? = null
    private val turns = listOf("auto", "w", "b")
    private var turnIndex = 0

    private val prefs by lazy { getSharedPreferences("xiangqi", Context.MODE_PRIVATE) }
    private var serverUrl: String
        get() = prefs.getString("server", "http://192.168.1.10:8000")!!
        set(v) = prefs.edit().putString("server", v).apply()
    private var sessionId: String
        get() = prefs.getString("session", null) ?: newSession()
        set(v) = prefs.edit().putString("session", v).apply()

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else show("Cần quyền camera", "", "")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)
        ui.root.setOnClickListener { captureAndAnalyze() }
        ui.root.setOnLongClickListener { editServer(); true }

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) tts?.language = Locale("vi", "VN")
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        tts?.shutdown()
        super.onDestroy()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(ui.preview.surfaceProvider) }
            // ~2MP là đủ để nhận dạng, gửi qua Wi-Fi nhanh hơn ảnh gốc
            val resolution = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1920, 1440), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                )
                .build()
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(resolution)
                .setJpegQuality(88)
                .build()
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                imageCapture = capture
            } catch (e: Exception) {
                Log.e(TAG, "Không mở được camera", e)
                show("Không mở được camera", e.message.orEmpty(), "")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndAnalyze() {
        val capture = imageCapture ?: return
        if (busy) return
        busy = true
        ui.moveText.text = "Đang phân tích…"
        capture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val jpeg = image.use { toJpegBytes(it) }
                analyze(jpeg)
            }

            override fun onError(exception: ImageCaptureException) {
                busy = false
                show("Lỗi chụp ảnh", exception.message.orEmpty(), "")
            }
        })
    }

    private fun analyze(jpeg: ByteArray) {
        val turn = turns[turnIndex]
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    ApiClient(serverUrl).analyze(jpeg, sessionId, turn, movetimeMs = 1500)
                }
                show(result.display, result.detail, result.board)
                tts?.speak(result.speech, TextToSpeech.QUEUE_FLUSH, null, "move")
            } catch (e: Exception) {
                Log.w(TAG, "Phân tích lỗi", e)
                show("Lỗi: " + (e.message ?: e.javaClass.simpleName), "Máy chủ: $serverUrl", "")
            } finally {
                busy = false
            }
        }
    }

    /** ImageCapture mặc định trả về JPEG: lấy nguyên byte, server tự xử lý mọi góc xoay. */
    private fun toJpegBytes(image: ImageProxy): ByteArray {
        val buffer = image.planes[0].buffer
        return ByteArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun show(main: String, detail: String, board: String) {
        ui.moveText.text = main
        val mode = "Lượt: " + when (turns[turnIndex]) { "w" -> "Đỏ"; "b" -> "Đen"; else -> "tự đoán" } +
            if (autoJob != null) " • tự chụp" else ""
        ui.detailText.text = listOf(detail, mode).filter { it.isNotBlank() }.joinToString("\n")
        ui.boardText.text = board
    }

    private fun toggleAuto() {
        autoJob?.let { it.cancel(); autoJob = null; show("Tắt tự chụp", "", ""); return }
        autoJob = lifecycleScope.launch {
            while (isActive) {
                captureAndAnalyze()
                delay(AUTO_INTERVAL_MS)
            }
        }
    }

    private fun newSession(): String {
        val id = UUID.randomUUID().toString()
        prefs.edit().putString("session", id).apply()
        return id
    }

    private fun editServer() {
        val input = EditText(this).apply { setText(serverUrl) }
        AlertDialog.Builder(this)
            .setTitle(R.string.server_url)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ -> serverUrl = input.text.toString().trim() }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_CAMERA,
            KeyEvent.KEYCODE_VOLUME_DOWN -> captureAndAnalyze()
            KeyEvent.KEYCODE_VOLUME_UP -> {
                turnIndex = (turnIndex + 1) % turns.size
                show(ui.moveText.text.toString(), "", ui.boardText.text.toString())
            }
            KeyEvent.KEYCODE_DPAD_UP -> toggleAuto()
            KeyEvent.KEYCODE_DPAD_DOWN -> {
                val old = sessionId
                sessionId = UUID.randomUUID().toString()
                lifecycleScope.launch(Dispatchers.IO) { runCatching { ApiClient(serverUrl).resetSession(old) } }
                Toast.makeText(this, "Ván mới", Toast.LENGTH_SHORT).show()
            }
            KeyEvent.KEYCODE_MENU -> editServer()
            else -> return super.onKeyDown(keyCode, event)
        }
        return true
    }

    companion object {
        private const val TAG = "XiangqiGlass"
        private const val AUTO_INTERVAL_MS = 4000L
    }
}
