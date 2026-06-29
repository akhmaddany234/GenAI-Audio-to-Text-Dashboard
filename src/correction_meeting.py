import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME


class TranscriptCorrectorMeeting:

    def __init__(self):

        genai.configure(
            api_key=GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel(
            MODEL_NAME
        )

    def correct(self, transcript):

        prompt = f"""
Anda adalah editor transkrip profesional yang ahli dalam bahasa Indonesia.

## Konteks
Transkrip ini berasal dari rekaman audio rapat/meeting yang ditranskripsi secara otomatis 
menggunakan model Whisper. Rekaman rapat dapat melibatkan lebih dari dua peserta dan 
membahas topik-topik bisnis/organisasi.

## Karakteristik Umum Rekaman Rapat
- Terdapat beberapa pembicara yang mungkin bergantian bicara
- Kemungkinan ada suara tumpang tindih atau noise latar belakang
- Terdapat istilah bisnis, teknis, atau organisasi yang spesifik
- Campuran bahasa Indonesia dan Inggris dalam konteks profesional adalah hal wajar

## Tugas Koreksi
Perbaiki kesalahan transkripsi otomatis berikut ini:

✅ YANG BOLEH DIPERBAIKI:
1. Kesalahan ejaan bahasa Indonesia (typo dari Whisper)
2. Nama kota, provinsi, dan wilayah yang salah transkripsi
3. Istilah bisnis dan profesional yang salah (contoh: "anggaran", "divisi", "kuorum")
4. Singkatan umum yang salah (contoh: "RAB", "RUPS", "MOU", "KPI", "OKR", "SOP")
5. Tanda baca dasar yang jelas salah (titik, koma, tanda tanya)
6. Kata yang terpotong atau tidak lengkap akibat noise/overlap suara
7. Nama jabatan yang salah (contoh: "Direktur", "Manajer", "Kepala Divisi")
8. Angka, persentase, atau nilai yang salah dibaca 
   (contoh: "lima belas persen" → "15%")

❌ YANG TIDAK BOLEH DIUBAH:
1. Makna atau isi dari setiap pernyataan
2. Label pembicara jika tersedia (contoh: "Pembicara 1:", "Pak Budi:", "[SPEAKER_01]:")
3. Keputusan, kesepakatan, atau angka yang disebutkan dalam rapat
4. Nama orang, nama perusahaan, nama proyek yang spesifik
5. Gaya bicara alami masing-masing pembicara
6. Jangan menambahkan kalimat, kata, atau informasi baru apapun
7. Jangan menghapus atau meringkas bagian percakapan manapun

## Penanganan Kasus Khusus
- Jika ada bagian yang tidak jelas atau tidak terbaca: tandai dengan [tidak jelas]
- Jika ada bagian yang terpotong karena noise: tandai dengan [noise]
- Jika ada pergantian pembicara yang tidak jelas: pertahankan apa adanya
- Jangan menebak atau mengasumsikan konten yang tidak ada dalam transkrip

## Format Output
- Kembalikan HANYA teks transkrip yang sudah diperbaiki
- Pertahankan seluruh struktur percakapan asli termasuk label pembicara
- Jangan tambahkan komentar, catatan, atau penjelasan perubahan
- Jangan gunakan markdown atau formatting tambahan

## Transkrip:
{transcript}
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text.strip()