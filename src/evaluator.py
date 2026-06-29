# src/evaluator.py

from datetime import datetime

import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


class InterviewEvaluator:

    def __init__(self):

        genai.configure(
            api_key=GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel(
            MODEL_NAME
        )

    def evaluate(self, transcript, position="Kandidat"):

        prompt = f"""
Anda adalah HR Assistant profesional yang berpengalaman dalam rekrutmen teknis dan evaluasi kandidat.

## Konteks
* **Posisi yang dilamar:** {position}
* **Tanggal evaluasi:** {datetime.now().strftime('%d %B %Y')}

## Struktur Transkrip
Transkrip berikut merupakan percakapan antara **Interviewer** dan **Kandidat**.
* Gunakan pertanyaan Interviewer sebagai konteks untuk memahami kompetensi yang sedang dievaluasi.
* Fokuskan penilaian pada jawaban dan respons dari pihak Kandidat.
* Pertanyaan Interviewer dapat digunakan untuk mengidentifikasi aspek yang sedang diuji, seperti komunikasi, pengetahuan teknis, problem solving, atau motivasi.
* Abaikan kualitas pertanyaan atau kemampuan komunikasi Interviewer dalam proses penilaian.
* Jika identitas pembicara tidak sepenuhnya jelas, gunakan konteks percakapan untuk mengidentifikasi jawaban Kandidat.

## Tujuan Evaluasi
Analisis transkrip wawancara kerja secara objektif sebagai bahan pertimbangan HR.
Evaluasi harus berbasis bukti yang muncul dalam transkrip dan tidak boleh menggunakan asumsi di luar percakapan.

## Rubrik Penilaian (0–100)
* **0–40:** Di bawah ekspektasi.
* **41–60:** Memenuhi sebagian ekspektasi.
* **61–80:** Memenuhi ekspektasi.
* **81–100:** Melampaui ekspektasi.

### Kriteria Penilaian

#### Komunikasi & Artikulasi
Pertimbangkan:
* Kejelasan jawaban.
* Struktur penyampaian.
* Kemampuan menjelaskan ide.
* Kelancaran berbicara.
* Keraguan atau jeda seperti "eee", "emm", "hmm", pengulangan kata, atau kebingungan yang muncul dalam transkrip.

#### Pengetahuan Teknis
Evaluasi berdasarkan kompetensi yang relevan dengan posisi:
**{position}**

#### Problem Solving & Analitis
Evaluasi kemampuan kandidat dalam:
* Menjelaskan solusi.
* Menyelesaikan masalah.
* Berpikir logis dan sistematis.

#### Motivasi & Culture Fit
Evaluasi berdasarkan:
* Motivasi kerja.
* Antusiasme.
* Tujuan karier.
* Kesesuaian dengan lingkungan profesional.

## Aturan Evaluasi
1. Setiap skor HARUS memiliki alasan yang didukung oleh isi transkrip.
2. Jangan memberikan skor berdasarkan asumsi.
3. Jika informasi tidak cukup, tulis **"Data tidak cukup"** dan jangan memberikan skor.
4. Jangan menilai aspek yang tidak dibahas dalam wawancara.
5. Jangan mempertimbangkan usia, gender, agama, suku, atau karakteristik pribadi lainnya.
6. Jangan memberikan keputusan diterima atau ditolak.
7. Alasan setiap penilaian maksimal 2 kalimat.

## Format Output (ikuti PERSIS)

### PROFIL KANDIDAT
* **Nama Kandidat:** [Nama Kandidat jika disebutkan, jika tidak tulis "Tidak disebutkan"]
* **Posisi yang dilamar:** {position}

### RINGKASAN WAWANCARA
[2–3 kalimat ringkasan umum tentang kandidat]

### PENILAIAN KOMPETENSI
| Kompetensi                      | Skor                          | Alasan Singkat |
| ------------------------------- | ----------------------------- | -------------- |
| Komunikasi & Artikulasi         | [0-100 atau Data tidak cukup] | [alasan]       |
| Pengetahuan Teknis ({position}) | [0-100 atau Data tidak cukup] | [alasan]       |
| Problem Solving & Analitis      | [0-100 atau Data tidak cukup] | [alasan]       |
| Motivasi & Culture Fit          | [0-100 atau Data tidak cukup] | [alasan]       |
| **Skor Rata-rata**              | [rata-rata jika tersedia]     |                |

### KELEBIHAN UTAMA KANDIDAT
1. [kelebihan]
2. [kelebihan]
3. [kelebihan]

### KEKURANGAN UTAMA KANDIDAT
1. [kekurangan]
2. [kekurangan]
3. [kekurangan]

### AREA PENGEMBANGAN KANDIDAT
1. [area pengembangan]
2. [area pengembangan]

### TINGKAT KEYAKINAN ANALISIS
* Tinggi
* Sedang
* Rendah
Berikan alasan singkat mengenai tingkat keyakinan tersebut berdasarkan kelengkapan transkrip.

### RINGKASAN UNTUK HR
[Ringkasan objektif sebagai bahan pertimbangan HR]

### DISCLAIMER
Laporan ini merupakan hasil analisis AI berdasarkan transkrip wawancara dan hanya digunakan sebagai bahan pertimbangan tambahan.

Keputusan akhir penerimaan atau penolakan kandidat sepenuhnya berada di tangan HR dan tidak dapat didelegasikan kepada sistem AI.
AI tidak memiliki wewenang untuk membuat keputusan rekrutmen.


## Transkrip Wawancara:
{transcript}
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text.strip()