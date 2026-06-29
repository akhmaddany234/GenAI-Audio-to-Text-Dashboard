# src/summarizer.py

import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


class MeetingSummarizer:

    def __init__(self):

        genai.configure(
            api_key=GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel(
            MODEL_NAME
        )

    def summarize(self, transcript):

        prompt = f"""
Anda adalah sekretaris profesional yang ahli dalam membuat notulensi rapat.

## Konteks
Berikut adalah transkrip rapat yang telah dikoreksi. Buat notulensi resmi dan terstruktur
berdasarkan isi transkrip tersebut.

## Format Notulensi

### 📋 INFORMASI RAPAT
- **Tanggal:** [ambil dari transkrip jika ada, jika tidak tulis "Tidak disebutkan"]
- **Peserta:** [daftar nama/jabatan yang muncul dalam transkrip]
- **Topik Utama:** [topik utama rapat]

### 📌 AGENDA YANG DIBAHAS
1. [agenda 1]
2. [agenda 2]
...

### 📝 RINGKASAN PEMBAHASAN
#### [Topik 1]
[Ringkasan poin-poin yang dibahas]

#### [Topik 2]
[Ringkasan poin-poin yang dibahas]

### ✅ KEPUTUSAN & KESEPAKATAN
1. [keputusan 1]
2. [keputusan 2]
...

### 📌 ACTION ITEMS
| No | Tugas | Penanggung Jawab | Tenggat Waktu |
|----|-------|-----------------|---------------|
| 1  | [tugas] | [nama/divisi] | [deadline] |
...

### 🔖 CATATAN TAMBAHAN
[Hal-hal penting lain yang perlu dicatat tapi tidak masuk kategori di atas]

### ⚠️ DISCLAIMER
Notulensi ini dibuat secara otomatis oleh sistem AI berdasarkan transkrip rekaman rapat
dan bersifat sebagai **draft awal**. Mohon dilakukan **verifikasi dan persetujuan** oleh
pihak yang berwenang sebelum notulensi ini dijadikan dokumen resmi.

---
PENTING:
- Hanya gunakan informasi yang ADA dalam transkrip, jangan menambahkan asumsi
- Jika suatu informasi tidak disebutkan, tulis "Tidak disebutkan dalam rekaman"
- Pertahankan akurasi angka, tanggal, dan nama yang muncul dalam transkrip
- Tetap objektif dan netral dalam merangkum setiap pembahasan

## Transkrip Rapat:
{transcript}
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text.strip()