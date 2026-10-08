package vn.xiangqi.glass

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/** Gọi API của server Python (xiangqi_assistant.api). */
class ApiClient(private val baseUrl: String) {

    data class Result(
        val display: String,   // dòng chính hiện trên kính, vd "Đỏ: Pháo 2 bình 5 | +0.3"
        val detail: String,    // diễn biến, cảnh báo
        val board: String,     // bàn cờ dạng chữ
        val speech: String,    // câu đọc bằng giọng nói
    )

    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    @Throws(IOException::class)
    fun analyze(jpeg: ByteArray, sessionId: String, turn: String, movetimeMs: Int): Result {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("image", "board.jpg", jpeg.toRequestBody("image/jpeg".toMediaType()))
            .addFormDataPart("turn", turn)
            .addFormDataPart("session_id", sessionId)
            .addFormDataPart("movetime_ms", movetimeMs.toString())
            .build()
        val request = Request.Builder().url(baseUrl.trimEnd('/') + "/api/analyze").post(body).build()
        http.newCall(request).execute().use { resp ->
            val text = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                val msg = runCatching { JSONObject(text).optString("detail", text) }.getOrDefault(text)
                throw IOException(msg.ifBlank { "HTTP ${resp.code}" })
            }
            return parse(JSONObject(text))
        }
    }

    @Throws(IOException::class)
    fun resetSession(sessionId: String) {
        val request = Request.Builder()
            .url(baseUrl.trimEnd('/') + "/api/session/$sessionId/reset")
            .post(ByteArray(0).toRequestBody(null))
            .build()
        http.newCall(request).execute().close()
    }

    private fun parse(json: JSONObject): Result {
        val s = json.getJSONObject("suggestion")
        val notes = mutableListOf<String>()
        json.optJSONObject("last_move")?.let { notes.add("Vừa đi: " + it.optString("text")) }
        val pv = s.optJSONArray("pv")
        if (pv != null && pv.length() > 1) {
            notes.add("Diễn biến: " + (0 until pv.length()).joinToString("  ") { pv.getString(it) })
        }
        val warnings = json.optJSONObject("recognition")?.optJSONArray("warnings")
        if (warnings != null) for (i in 0 until warnings.length()) notes.add("⚠ " + warnings.getString(i))
        val board = json.optString("board")

        if (!s.optBoolean("ok")) {
            val errs = s.optJSONArray("errors")
            val msg = if (errs != null && errs.length() > 0) errs.getString(0) else "Thế cờ không hợp lệ"
            return Result("Nhận dạng sai: chụp lại", (listOf(msg) + notes).joinToString("\n"), board,
                "Nhận dạng chưa đúng, hãy chụp lại")
        }
        val display = s.optString("display", "Hết nước đi")
        val speech = s.optString("move_text", display)
        return Result(display, notes.joinToString("\n"), board, speech)
    }
}
