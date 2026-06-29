import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


class TranscriptCorrectorInterview:

    def __init__(self):

        genai.configure(
            api_key=GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel(
            MODEL_NAME
        )

    def correct(self, transcript):

        prompt = f"""
Anda adalah editor transkrip profesional yang ahli dalam bahasa Indonesia dan transkripsi wawancara kerja.

## Konteks
Transkrip berikut berasal dari rekaman wawancara kerja (job interview) yang ditranskripsi secara otomatis menggunakan model Whisper.
Percakapan melibatkan dua pihak:
* **Interviewer**: Pewawancara dari pihak perusahaan.
* **Kandidat**: Pelamar kerja.

Tujuan Anda adalah memperbaiki kesalahan hasil transkripsi otomatis tanpa mengubah isi percakapan.

## Tugas Koreksi
Perbaiki kesalahan transkripsi berikut.

### YANG BOLEH DIPERBAIKI
1. Kesalahan ejaan bahasa Indonesia.
2. Typo akibat kesalahan transkripsi otomatis.
3. Nama kota, provinsi, dan wilayah yang jelas salah.
4. Istilah profesional atau teknis yang salah penulisan.
5. Tanda baca dasar yang diperlukan agar kalimat mudah dipahami.
6. Kata yang terpotong atau tidak lengkap akibat proses transkripsi.
7. Istilah campuran Indonesia-Inggris yang umum digunakan dalam dunia kerja seperti:
   * meeting
   * deadline
   * interview
   * leadership
   * teamwork
8. Kata pengisi seperti "eee", "hmm", atau jeda yang tidak memiliki makna jangan dihapus.

### YANG TIDAK BOLEH DIUBAH
1. Makna atau isi pernyataan.
2. Informasi faktual seperti angka, tanggal, nama orang, nama perusahaan, atau pengalaman kerja.
3. Label speaker seperti:
   * Interviewer:
   * Kandidat:
4. Urutan percakapan.
5. Gaya bicara alami kandidat maupun interviewer.
6. Bahasa campuran yang memang digunakan oleh pembicara.
7. Bagian percakapan apa pun.

### ATURAN KHUSUS
1. Jika tidak yakin terhadap suatu koreksi, pertahankan teks asli.
2. Jangan menebak nama orang, perusahaan, jabatan, atau istilah yang tidak jelas.
3. Jangan menambahkan informasi baru.
4. Jangan membuat kesimpulan atau ringkasan.
5. Jangan mengubah percakapan menjadi bahasa yang lebih formal.
6. Jangan memperbaiki sesuatu yang masih mungkin benar.

## Format Output
* Kembalikan HANYA teks transkrip yang sudah diperbaiki.
* Pertahankan label speaker dan struktur percakapan.
* Jangan menambahkan komentar atau penjelasan.
* Jangan menggunakan markdown.
* Jangan menjelaskan perubahan yang dilakukan.

## Transkrip:
{transcript}
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text.strip()