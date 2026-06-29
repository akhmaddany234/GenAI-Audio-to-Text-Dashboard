import streamlit as st
import os
import sys
import tempfile
import soundfile as sf
import subprocess
from datetime import datetime

# Add workspace path to sys.path to resolve src imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import config
from src.transcriber import AudioTranscriber
from src.correction_interview import TranscriptCorrectorInterview
from src.correction_meeting import TranscriptCorrectorMeeting
from src.summarizer import MeetingSummarizer
from src.evaluator import InterviewEvaluator
from src.exporter import DocumentExporter
from src.processing_audio import AudioProcessor

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="GenAI Audio to Text Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    /* Styling headers */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .title-container h1 {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .title-container p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Section style */
    .section-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Custom button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Hover micro-animation */
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.2);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .status-pending { background-color: #f39c12; }
    .status-success { background-color: #2ecc71; }
    .status-error { background-color: #e74c3c; }
    </style>
""", unsafe_allow_html=True)

# ----------------- App Header -----------------
st.markdown("""
    <div class="title-container">
        <h1>🎙️ GenAI Audio to Text & Analysis</h1>
        <p>Transkripsikan audio secara cerdas, koreksi tata bahasa, dan buat laporan notulensi atau evaluasi wawancara otomatis menggunakan Gemini AI</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- Initialize Session States -----------------
if 'raw_transcript' not in st.session_state:
    st.session_state['raw_transcript'] = ""
if 'corrected_transcript' not in st.session_state:
    st.session_state['corrected_transcript'] = ""
if 'final_report' not in st.session_state:
    st.session_state['final_report'] = ""
if 'processed_audio_info' not in st.session_state:
    st.session_state['processed_audio_info'] = None
if 'audio_file_path' not in st.session_state:
    st.session_state['audio_file_path'] = ""
if 'reduced_audio_bytes' not in st.session_state:
    st.session_state['reduced_audio_bytes'] = None
if 'reduced_audio_path' not in st.session_state:
    st.session_state['reduced_audio_path'] = ""
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 0  # 0: Uploaded, 1: Transcribed, 2: Corrected, 3: Completed

# ----------------- Sidebar Configuration -----------------
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=90)
st.sidebar.title("Menu Settings")


# 1. Mode Selection
st.sidebar.markdown("### Pilih Mode Alur Kerja")
mode = st.sidebar.selectbox(
    "Pilih Mode Alur Kerja",
    options=["Wawancara (Interview)", "Rapat (Meeting)"],
    index=0
)
is_interview = mode.startswith("Wawancara")

# Conditional input for Wawancara mode
position = "Kandidat"
if is_interview:
    position = st.sidebar.text_input(
        "Target Posisi Kandidat",
        value="Supervisor Farm",
        placeholder="Enter target job position..."
    )


st.sidebar.markdown("### Transcription Settings")
whisper_model_size = st.sidebar.selectbox(
    "Whisper Model Size",
    options=["tiny", "base", "small", "medium", "large"],
    index=2, # default small
    help="Higher sizes yield better accuracy but require more time and system memory."
)
whisper_device = st.sidebar.selectbox(
    "Device",
    options=["cpu", "cuda"],
    index=0
)
whisper_compute_type = st.sidebar.selectbox(
    "Compute Type",
    options=["int8", "float16", "float32"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n1. Pastikan audio jelas dan berisik latar belakang minim.\n2. Jika proses transkripsi terasa lambat pada CPU, pilih model 'base' atau 'tiny' di pengaturan.")

# ----------------- Main Layout -----------------
col_main, col_info = st.columns([7, 3])

with col_info:
    st.markdown("""
        <div class="section-card">
            <h3>Pipeline Alur Kerja</h3>
            <ol>
                <li><b>Upload Audio:</b> Unggah file audio rekaman dalam format WAV/MP3.</li>
                <li><b>Transkripsi:</b> AI akan menyalin ucapan ke teks mentah.</li>
                <li><b>Koreksi Teks:</b> AI memperbaiki kesalahan ejaan (typo) & tanda baca.</li>
                <li><b>Analisis Laporan:</b> Pembuatan notulensi rapat (Meeting Mode) atau evaluasi kandidat (Interview Mode).</li>
                <li><b>Ekspor Word:</b> Unduh hasil laporan dalam format Word (.docx).</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Progress visualization based on state
    st.markdown("### Status Pipeline")
    step = st.session_state['current_step']
    if step == 0:
        st.markdown('<div class="status-badge status-pending">Ready to Transcribe</div>', unsafe_allow_html=True)
        st.progress(10)
    elif step == 1:
        st.markdown('<div class="status-badge status-pending">Transcribed (Needs Correction)</div>', unsafe_allow_html=True)
        st.progress(40)
    elif step == 2:
        st.markdown('<div class="status-badge status-pending">Corrected (Ready for Report)</div>', unsafe_allow_html=True)
        st.progress(70)
    elif step == 3:
        st.markdown('<div class="status-badge status-success">Completed & Exported</div>', unsafe_allow_html=True)
        st.progress(100)

with col_main:
    st.markdown("### Upload File Audio")
    uploaded_file = st.file_uploader(
        "Pilih file audio (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"],
        help="Unggah rekaman rapat atau wawancara Anda di sini."
    )
    
    # Audio slicing tool inside an expander
    enable_audio_slicing = False
    start_sec = 0
    end_sec = 0
    audio_dur = 0.0
    
    if uploaded_file is not None:
        # Save temp file to read audio info
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Reset session states if this is a new file
        if st.session_state.get('audio_file_path') != temp_audio_path:
            st.session_state['audio_file_path'] = temp_audio_path
            st.session_state['reduced_audio_bytes'] = None
            st.session_state['reduced_audio_path'] = ""
            st.session_state['raw_transcript'] = ""
            st.session_state['corrected_transcript'] = ""
            st.session_state['final_report'] = ""
            st.session_state['current_step'] = 0
        
        # Audio metadata & player
        try:
            # Convert to a temporary WAV to read metadata safely
            temp_wav = os.path.join(temp_dir, "metadata_temp.wav")
            cmd = ['ffmpeg', '-y', '-i', temp_audio_path, temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            try:
                info = sf.info(temp_wav)
                audio_dur = info.duration
                channels = info.channels
                frame_rate = info.samplerate
            finally:
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
                    
            st.session_state['processed_audio_info'] = {
                "duration": audio_dur,
                "channels": channels,
                "frame_rate": frame_rate,
                "file_name": uploaded_file.name
            }
            
            # Original Audio Player
            st.markdown("##### 📁 Audio Asli")
            st.audio(uploaded_file)
            
            # Processed Audio Player (if exists)
            if st.session_state.get('reduced_audio_bytes') is not None:
                st.markdown("##### 🧼 Audio Hasil Pembersihan Noise (Noise-Reduced)")
                st.audio(st.session_state['reduced_audio_bytes'], format="audio/wav")
                
            st.markdown(f"**Nama file:** `{uploaded_file.name}` | **Durasi:** `{audio_dur:.2f} detik` ({audio_dur/60:.2f} menit) | **Sample Rate:** `{frame_rate} Hz`")
            
            # 📊 Audio Statistics & Visualization
            with st.expander("📊 Analisis & Visualisasi Audio (Deteksi Noise)"):
                st.write("Analisis kenyaringan dan frekuensi audio untuk mendeteksi potensi adanya background noise.")
                with st.spinner("Menganalisis properti audio..."):
                    try:
                        processor = AudioProcessor()
                        stats = processor.get_audio_stats(temp_audio_path)
                        
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Peak Amplitude", f"{stats['peak_amplitude']:.4f}")
                        m_col2.metric("RMS Energy (Volume)", f"{stats['rms_energy']:.4f}")
                        m_col3.metric("Noise Floor (RMS)", f"{stats['noise_floor']:.4f}")
                        
                        snr_val = stats['snr']
                        if snr_val >= 20:
                            snr_status = "Sangat Jelas"
                            help_msg = "Sinyal suara sangat dominan dibanding kebisingan."
                        elif snr_val >= 12:
                            snr_status = "Cukup Bersih"
                            help_msg = "Ada sedikit kebisingan latar belakang tetapi ucapan masih terdengar jelas."
                        else:
                            snr_status = "Bising/Banyak Gangguan"
                            help_msg = "Kebisingan latar belakang tinggi. Disarankan membersihkan noise."
                            
                        m_col4.metric("Estimated SNR", f"{snr_val:.2f} dB", help=f"Signal-to-Noise Ratio. Status: {snr_status}. {help_msg}")
                        
                        if snr_val < 15:
                            st.warning(f"⚠️ **Audio terdeteksi cukup bising (SNR: {snr_val:.2f} dB - {snr_status}).** Silakan klik tombol **Pembersihan Noise** di bawah untuk menjernihkan audio.")
                        else:
                            st.success(f"✅ **Kualitas audio bagus (SNR: {snr_val:.2f} dB - {snr_status}).** Sinyal audio cukup bersih untuk ditranskripsi langsung.")
                        
                        # Generate and display plot
                        fig = processor.plot_audio_properties(temp_audio_path)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Gagal melakukan analisis audio: {str(e)}")
            
            # 🧼 Noise Reduction Button
            if st.button("🧼 Jalankan Pembersihan Noise", key="btn_reduce_noise", use_container_width=True):
                with st.spinner("Membersihkan background noise..."):
                    try:
                        processor = AudioProcessor()
                        reduced_filename = f"reduced_{uploaded_file.name}"
                        reduced_path = os.path.join(temp_dir, reduced_filename)
                        
                        processor.reduce_noise(temp_audio_path, reduced_path)
                        
                        # Save path and read bytes to play in UI
                        st.session_state['reduced_audio_path'] = reduced_path
                        with open(reduced_path, "rb") as f_reduced:
                            st.session_state['reduced_audio_bytes'] = f_reduced.read()
                            
                        st.success("Pembersihan noise selesai! Audio hasil pemrosesan siap digunakan.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal melakukan pembersihan noise: {str(e)}")
            
            # Slicing feature
            with st.expander("✂️ Potong Audio (Opsional)"):
                st.write("Potong durasi audio untuk mempercepat pemrosesan atau fokus pada segmen tertentu.")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    start_min = st.number_input("Menit Mulai", min_value=0, max_value=int(audio_dur // 60), value=0)
                    start_sec_inp = st.number_input("Detik Mulai", min_value=0, max_value=59, value=0)
                    start_sec = start_min * 60 + start_sec_inp
                with col_s2:
                    end_min = st.number_input("Menit Selesai", min_value=0, max_value=int(audio_dur // 60) + 1, value=int(audio_dur // 60))
                    end_sec_inp = st.number_input("Detik Selesai", min_value=0, max_value=59, value=int(audio_dur % 60))
                    end_sec = end_min * 60 + end_sec_inp
                
                if start_sec >= end_sec:
                    st.warning("⚠️ Waktu mulai harus lebih kecil dari waktu selesai.")
                elif end_sec > audio_dur:
                    st.warning(f"⚠️ Waktu selesai melebihi durasi audio ({audio_dur:.1f} detik).")
                else:
                    enable_audio_slicing = st.checkbox("Aktifkan pemotongan audio saat proses")
                    if enable_audio_slicing:
                        st.info(f"Audio akan dipotong dari **{start_min:02d}:{start_sec_inp:02d}** sampai **{end_min:02d}:{end_sec_inp:02d}**.")
                        
        except Exception as e:
            st.error(f"Gagal membaca metadata audio: {str(e)}")
            st.warning("Pastikan ffmpeg terinstal jika Anda mengunggah file non-WAV.")
            st.audio(uploaded_file)
            
    # Tabs for Workflow steps
    st.markdown("---")
    tab_trans, tab_correct, tab_report = st.tabs([
        "🎙️ Transkripsi",
        "✨ Koreksi & Sunting",
        "📄 Laporan Hasil Analisis"
    ])
    
    # ----------------- TAB 1: Transcription -----------------
    with tab_trans:
        st.markdown("### Proses Transkripsi Audio")
        if uploaded_file is None:
            st.info("Silakan unggah file audio terlebih dahulu untuk memulai transkripsi.")
        else:
            if st.button("🚀 Jalankan Transkripsi", key="btn_transcribe", use_container_width=True):
                with st.spinner("Mentranskripsikan audio... Mohon tunggu (proses ini bergantung pada durasi audio)"):
                    try:
                        # 1. Tentukan audio sumber (gunakan hasil pembersihan jika ada)
                        base_audio_path = st.session_state['audio_file_path']
                        using_reduced = False
                        
                        if st.session_state.get('reduced_audio_path') and os.path.exists(st.session_state['reduced_audio_path']):
                            base_audio_path = st.session_state['reduced_audio_path']
                            using_reduced = True
                            st.write(" menggunakan audio hasil pembersihan noise...")
                        else:
                            st.write(" menggunakan audio asli...")
                        
                        # 2. Tangani pemotongan (slicing) jika diaktifkan
                        input_audio_path = base_audio_path
                        sliced_path = ""
                        if enable_audio_slicing:
                            st.write("Sedang memotong audio...")
                            sliced_path = os.path.join(temp_dir, f"sliced_{uploaded_file.name}")
                            cmd = ['ffmpeg', '-y', '-ss', str(start_sec), '-to', str(end_sec), '-i', base_audio_path, sliced_path]
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            input_audio_path = sliced_path
                        
                        # 3. Inisialisasi transcriber
                        transcriber = AudioTranscriber(
                            model_size=whisper_model_size,
                            device=whisper_device,
                            compute_type=whisper_compute_type
                        )
                        
                        # Setup progress indicators
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        def update_progress(current, total, chunk_text):
                            percent = float(current) / float(total)
                            progress_bar.progress(percent)
                            status_text.info(f"Mengolah bagian {current} dari {total}...")
                        
                        raw_text = transcriber.transcribe(
                            input_audio_path,
                            language="id",
                            progress_callback=update_progress
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.session_state['raw_transcript'] = raw_text
                        st.session_state['corrected_transcript'] = "" # Reset downstream
                        st.session_state['final_report'] = "" # Reset downstream
                        st.session_state['current_step'] = 1
                        
                        # Cleanup temp sliced file
                        if sliced_path and os.path.exists(sliced_path):
                            os.unlink(sliced_path)
                            
                        st.success("Transkripsi selesai!")
                    except Exception as e:
                        st.error(f"Error saat transkripsi: {str(e)}")
            
            # Show raw transcript if available
            if st.session_state['raw_transcript']:
                st.markdown("#### Hasil Transkrip Mentah")
                st.text_area(
                    "Teks hasil transkripsi otomatis (dapat disalin/diedit):",
                    value=st.session_state['raw_transcript'],
                    height=250,
                    key="raw_transcript_area"
                )
                
    # ----------------- TAB 2: Correction & Editing -----------------
    with tab_correct:
        st.markdown("### Koreksi & Penyuntingan Teks")
        if not st.session_state['raw_transcript']:
            st.info("Silakan jalankan langkah transkripsi terlebih dahulu atau isi transkrip mentah di Tab 1.")
        else:
            col_corr1, col_corr2 = st.columns(2)
            with col_corr1:
                st.write("Gunakan LLM Gemini untuk mengoreksi typo, tata bahasa, nama tempat, dan memformat percakapan tanpa mengubah makna asli.")
                
            with col_corr2:
                if st.button("✨ Jalankan Koreksi Otomatis", key="btn_correct", use_container_width=True):
                    with st.spinner("Mengoreksi teks transkrip menggunakan Gemini AI..."):
                        try:
                            raw_text_to_correct = st.session_state['raw_transcript']
                            if is_interview:
                                corrector = TranscriptCorrectorInterview()
                            else:
                                corrector = TranscriptCorrectorMeeting()
                            
                            # Call correct
                            corrected_text = corrector.correct(raw_text_to_correct)
                            st.session_state['corrected_transcript'] = corrected_text
                            st.session_state['current_step'] = 2
                            st.success("Koreksi otomatis selesai!")
                        except Exception as e:
                            st.error(f"Error saat melakukan koreksi via API: {str(e)}")
                            st.warning("⚠️ Menggunakan teks transkrip mentah sebagai fallback karena kendala koneksi API.")
                            st.session_state['corrected_transcript'] = raw_text_to_correct
                            st.session_state['current_step'] = 2
            
            # Show corrected text and allow editing
            if st.session_state['corrected_transcript']:
                st.markdown("#### Hasil Koreksi Teks")
                edited_corrected = st.text_area(
                    "Anda dapat mengedit transkrip hasil koreksi ini secara manual sebelum membuat laporan:",
                    value=st.session_state['corrected_transcript'],
                    height=300,
                    key="corrected_transcript_area"
                )
                # Keep state updated if manual edits occur
                st.session_state['corrected_transcript'] = edited_corrected
                
                # Save to file output/corrected/corrected_transkrip_final.txt
                try:
                    os.makedirs(os.path.join(project_root, "output", "corrected"), exist_ok=True)
                    corrected_file_path = os.path.join(project_root, "output", "corrected", "corrected_transkrip_final.txt")
                    with open(corrected_file_path, "w", encoding="utf-8") as f:
                        f.write(edited_corrected)
                except Exception as e:
                    st.warning(f"Gagal menyimpan salinan file transkrip lokal: {str(e)}")

    # ----------------- TAB 3: Generation & Report Exporter -----------------
    with tab_report:
        st.markdown("### Pembuatan Laporan Analisis & Ekspor")
        if not st.session_state['corrected_transcript']:
            st.info("Selesaikan langkah koreksi transkrip terlebih dahulu.")
        else:
            st.write(f"Sistem akan memproses transkrip hasil koreksi untuk mode: **{mode}**")
            
            if st.button("📊 Buat Laporan & Ekspor Word", key="btn_report", use_container_width=True):
                with st.spinner("Menganalisis teks dan membuat laporan..."):
                    clean_text = st.session_state['corrected_transcript']
                    os.makedirs(os.path.join(project_root, "output", "docx"), exist_ok=True)
                    try:
                        if is_interview:
                            # Evaluate Interview
                            evaluator = InterviewEvaluator()
                            result = evaluator.evaluate(clean_text, position)
                            title = "EVALUASI WAWANCARA"
                            filename = os.path.join(project_root, "output", "docx", "evaluasi_wawancara.docx")
                        else:
                            # Summarize Meeting
                            summarizer = MeetingSummarizer()
                            result = summarizer.summarize(clean_text)
                            title = "NOTULENSI RAPAT"
                            filename = os.path.join(project_root, "output", "docx", "notulensi_rapat.docx")
                    except Exception as e:
                        st.error(f"Error saat membuat laporan via API: {str(e)}")

                    
                    try:
                        # Export to Docx
                        exporter = DocumentExporter()
                        exporter.export(content=result, filename=filename, title=title)
                        
                        st.session_state['final_report'] = result
                        st.session_state['export_filename'] = filename
                        st.session_state['export_title'] = title
                        st.session_state['current_step'] = 3
                        st.success("Laporan berhasil dibuat (Draft / Hasil AI)!")
                    except Exception as export_err:
                        st.error(f"Gagal mengekspor laporan ke Word: {str(export_err)}")
            
            # Show final report and download buttons
            if st.session_state['final_report']:
                st.markdown("#### 📝 Laporan Hasil AI")
                st.markdown(st.session_state['final_report'])
                
                st.markdown("---")
                st.markdown("#### 📥 Unduh Hasil")
                
                # Check if file exists to read for download button
                filename = st.session_state.get('export_filename', '')
                if filename and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        docx_bytes = f.read()
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="⬇️ Unduh Laporan (.docx)",
                            data=docx_bytes,
                            file_name=os.path.basename(filename),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    with col_dl2:
                        corrected_txt = st.session_state['corrected_transcript']
                        st.download_button(
                            label="⬇️ Unduh Transkrip (.txt)",
                            data=corrected_txt,
                            file_name="transkrip_koreksi.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.error("File Word hasil ekspor tidak ditemukan di penyimpanan server lokal.")
