import sys
import os
import cv2
import mysql.connector
import numpy as np
import torch
from difflib import get_close_matches

BANK_BARANG = [
    "aqua",
    "le mineral",
    "crystaline",
    "coca cola",
    "beng beng",
    "nextar",
    "oreo",
    "qtela",
    "milku",
    "floridina",
    "nutriboost",
    "nipis madu",
    "pizza",
    "pizza mini",
    "donat",
    "gorengan",
    "sate usus",
    "pisang coklat",
    "nasi goreng",
    "kentaki",
    "rames ayam",
    "roti bread shop banana",
    "stopmap"
]
CONFIDENCE_THRESHOLD = 85

from datetime import datetime
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QFrame,
    QHBoxLayout,
    QGraphicsDropShadowEffect,
    QLineEdit,
    QComboBox,
    QScrollArea,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QAbstractItemView
)

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD MODEL
# =========================================================
model_path = "best_kfold_model"

processor = TrOCRProcessor.from_pretrained(model_path)

model = VisionEncoderDecoderModel.from_pretrained(model_path)

model.to(device)
model.eval()

# =========================================================
# DATASET RETRAINING
# =========================================================

TEMP_CROP_DIR = "temp_crop"

os.makedirs(
    TEMP_CROP_DIR,
    exist_ok=True
)

# =========================================================
# DATABASE MYSQL LARAGON
# =========================================================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="htr_transaksi"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transaksi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tanggal VARCHAR(100),
    barang VARCHAR(255),
    jumlah INT,
    total INT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS koreksi_barang (
    id INT AUTO_INCREMENT PRIMARY KEY,
    salah VARCHAR(255),
    benar VARCHAR(255)
)
""")

conn.commit()

conn.commit()
# =========================================================
# SHADOW EFFECT
# =========================================================
def add_shadow(widget):

    shadow = QGraphicsDropShadowEffect()

    shadow.setBlurRadius(35)

    shadow.setXOffset(3)

    shadow.setYOffset(5)

    shadow.setColor(QColor(0, 0, 0, 90))

    widget.setGraphicsEffect(shadow)

# =========================================================
# CEK BANK BARANG
# =========================================================

def check_barang_bank(barang):

    barang = barang.lower().strip()

    if barang in BANK_BARANG:

        return barang, True

    match = get_close_matches(
        barang,
        BANK_BARANG,
        n=1,
        cutoff=0.75
    )

    if match:

        return match[0], True

    return barang, False

# Cek Database 
def check_database_koreksi(barang):

    cursor.execute(
        """
        SELECT benar
        FROM koreksi_barang
        WHERE LOWER(salah)=LOWER(%s)
        """,
        (barang,)
    )

    result = cursor.fetchone()

    if result:
        return result[0], True

    return barang, False

# =========================================================
# PARSING TEXT
# =========================================================
def parse_text(text):

    parts = text.split()

    barang_parts = []
    numbers = []

    for p in parts:

        clean = p.replace(".", "").replace(",", "")

        if clean.isdigit():
            numbers.append(int(clean))
        else:
            barang_parts.append(p)

    barang = " ".join(barang_parts)

    barang_bank, valid_bank = check_barang_bank(barang)

    if valid_bank:

        barang = barang_bank
        valid = True

    else:

        barang_db, valid_db = check_database_koreksi(barang)

        if valid_db:

            barang = barang_db
            valid = True

        else:

            valid = False

    jumlah = 0
    total = 0

    if len(numbers) >= 2:

        jumlah = numbers[-2]
        total = numbers[-1]

    return {
        "barang": barang,
        "jumlah": jumlah,
        "total": total,
        "valid_barang": valid
    }

# =========================================================
# CROP BARIS DAN PEMBCAAN GAMBAR
# =========================================================
def crop_lines(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8
    )

    kernel_line = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (120,1)
    )

    remove_lines = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel_line
    )

    text = cv2.subtract(thresh, remove_lines)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40,6)
    )

    text = cv2.dilate(text, kernel, iterations=2)

    #Proses Segmentasi

    proj = np.sum(text, axis=1)

    proj = np.convolve(
        proj,
        np.ones(5)/5,
        mode='same'
    )

    #batas threshold jika diatas maka dianggap memiliki tulisan

    threshold = np.max(proj) * 0.2

    lines = []

    start = None

    for i, val in enumerate(proj):

        if val > threshold and start is None:

            start = i

        elif val <= threshold and start is not None:

            end = i

            if end - start > 20:

                lines.append((start, end))

            start = None

    crops = []

    for y1, y2 in lines:

        crop = img[y1:y2, :]

        crops.append(crop)

    return crops

# =========================================================
# OCR PREDICT
# =========================================================
def predict_crop(crop):

    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image).convert("RGB")

    image = image.resize((512,128))

    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8
    )

    image = Image.fromarray(binary).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True
        )

    pred_text = processor.batch_decode(
        generated_ids.sequences,
        skip_special_tokens=True
    )[0]

    token_scores = torch.stack(
        generated_ids.scores
    )

    token_probs = token_scores.softmax(-1)

    max_probs = token_probs.max(-1).values

    confidence = (
        max_probs.mean().item() * 100
    )

    return pred_text, confidence


# =========================================================
# FULLSCREEN PREVIEW WINDOW
# =========================================================
class FullScreenPreview(QWidget):

    def __init__(self, title, content_widget):

        super().__init__()

        self.setWindowTitle(title)

        self.resize(1400, 900)

        layout = QVBoxLayout()

        close_btn = QPushButton("❌ Tutup")

        close_btn.clicked.connect(self.close)

        layout.addWidget(close_btn)

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        scroll.setWidget(content_widget)

        layout.addWidget(scroll)

        self.setLayout(layout)

# =========================================================
# DATABASE WINDOW
# =========================================================
class DatabaseWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Database Transaksi"
        )

        self.resize(1200, 700)

        layout = QVBoxLayout()

        # =========================================
        # TITLE
        # =========================================
        title = QLabel(
            "📋 Histori Database Transaksi"
        )

        title.setAlignment(Qt.AlignCenter)

        title.setFont(
            QFont(
                "Segoe UI",
                22,
                QFont.Bold
            )
        )

        layout.addWidget(title)

        # =========================================
        # SEARCH
        # =========================================
        self.search_box = QLineEdit()

        self.search_box.setPlaceholderText(
            "Cari barang..."
        )

        self.search_box.textChanged.connect(
            self.filter_table
        )

        layout.addWidget(self.search_box)

        # =========================================
        # TABLE
        # =========================================
        self.table = QTableWidget()

        self.table.setColumnCount(6)

        self.table.setHorizontalHeaderLabels([
            "Pilih",
            "ID",
            "Tanggal",
            "Barang",
            "Jumlah",
            "Total"
        ])

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        # =========================================
        # BUTTON AREA
        # =========================================
        button_layout = QHBoxLayout()

        self.select_btn = QPushButton(
            "☑ Select All"
        )

        self.add_btn = QPushButton(
            "➕ Tambah"
        )

        self.edit_btn = QPushButton(
            "✏ Simpan Edit"
        )

        self.delete_btn = QPushButton(
            "🗑 Hapus"
        )

        self.refresh_btn = QPushButton(
            "🔄 Refresh"
        )

        self.select_btn.clicked.connect(
            self.select_all_rows
        )

        self.add_btn.clicked.connect(
            self.add_data
        )

        self.edit_btn.clicked.connect(
            self.edit_data
        )

        self.delete_btn.clicked.connect(
            self.delete_data
        )

        self.refresh_btn.clicked.connect(
            self.load_data
        )

        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.edit_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.refresh_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.load_data()

    # =========================================
    # LOAD DATA
    # =========================================
    def load_data(self):

        cursor.execute(
            "SELECT * FROM transaksi"
        )

        rows = cursor.fetchall()

        self.table.setRowCount(len(rows))

        for i, row_data in enumerate(rows):

            checkbox = QTableWidgetItem()

            checkbox.setCheckState(Qt.Unchecked)

            self.table.setItem(i, 0, checkbox)

            for j, data in enumerate(row_data):

                item = QTableWidgetItem(str(data))

                # Kolom ID
                if j == 0:
                    item.setFlags(
                        item.flags() & ~Qt.ItemIsEditable
                    )

                self.table.setItem(i, j + 1, item)

    # =========================================
    # FILTER
    # =========================================
    def filter_table(self):

        keyword = self.search_box.text().lower()

        for row in range(self.table.rowCount()):

            item = self.table.item(row, 3)

            if item:

                self.table.setRowHidden(
                    row,
                    keyword not in item.text().lower()
                )

    # =========================================
    # SELECT ALL
    # =========================================
    def select_all_rows(self):

        for row in range(self.table.rowCount()):

            item = self.table.item(row, 0)

            if item.checkState() == Qt.Checked:

                item.setCheckState(Qt.Unchecked)

            else:

                item.setCheckState(Qt.Checked)

    # =========================================
    # ADD DATA
    # =========================================
    def add_data(self):

        tanggal = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cursor.execute("""
        INSERT INTO transaksi (
            tanggal,
            barang,
            jumlah,
            total
        )
        VALUES (%s,%s,%s,%s)
        """, (
            tanggal,
            "Barang Baru",
            1,
            0
        ))

        conn.commit()

        self.load_data()

    # =========================================
    # EDIT DATA
    # =========================================
    def edit_data(self):

        try:

            for row in range(self.table.rowCount()):

                data_id = self.table.item(row, 1).text()

                tanggal = self.table.item(row, 2).text()

                barang = self.table.item(row, 3).text()

                jumlah = self.table.item(row, 4).text()

                total = self.table.item(row, 5).text()

                cursor.execute("""
                UPDATE transaksi
                SET
                    tanggal=%s,
                    barang=%s,
                    jumlah=%s,
                    total=%s
                WHERE id=%s
                """, (
                    tanggal,
                    barang,
                    jumlah,
                    total,
                    data_id
                ))

            conn.commit()

            QMessageBox.information(
                self,
                "Success",
                "Data berhasil diupdate"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                str(e)
            )

    # =========================================
    # DELETE DATA
    # =========================================
    def delete_data(self):

        selected_ids = []

        for row in range(self.table.rowCount()):

            checkbox_item = self.table.item(row, 0)

            if checkbox_item.checkState() == Qt.Checked:

                id_item = self.table.item(row, 1)

                selected_ids.append(
                    id_item.text()
                )

        if len(selected_ids) == 0:

            QMessageBox.warning(
                self,
                "Warning",
                "Pilih data terlebih dahulu"
            )

            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi",
            f"Hapus {len(selected_ids)} data?"
        )

        if confirm == QMessageBox.Yes:

            for data_id in selected_ids:

                cursor.execute(
                    "DELETE FROM transaksi WHERE id=%s",
                    (data_id,)
                )

            conn.commit()

            self.load_data()

            QMessageBox.information(
                self,
                "Success",
                "Data berhasil dihapus"
            )

# =========================================================
# MAIN WINDOW
# =========================================================
class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "HTR Digitalisasi Transaksi"
        )

        self.resize(1400, 900)

        self.parsed_results = []

        self.crop_zoom_width = 700
        self.crop_zoom_height = 180
        self.ocr_font_size = 16

        # =================================================
        # DARK THEME
        # =================================================
        self.setStyleSheet("""
        QWidget{
            background-color:#0F172A;
            color:#F8FAFC;
            font-family:'Segoe UI';
            font-size:15px;
        }

        QLabel{
            color:#F8FAFC;
        }

        QFrame{
            background-color:#1E293B;
            border:none;
            border-radius:20px;
        }

        QPushButton{
            background-color:#556B2F;
            color:white;
            border:1px solid #6B8E23;

            border-top:2px solid #7FA34D;
            border-left:2px solid #7FA34D;

            border-bottom:2px solid #324018;
            border-right:2px solid #324018;

            border-radius:14px;
            padding:14px;
            font-size:15px;
            font-weight:600;
        }

        QPushButton:hover{
            background-color:#667D38;
        }

        QPushButton:pressed{
            background-color:#465824;

            border-top:2px solid #324018;
            border-left:2px solid #324018;

            border-bottom:2px solid #7FA34D;
            border-right:2px solid #7FA34D;

            padding-top:16px;
            padding-left:16px;
        }

        QLineEdit,
        QTextEdit,
        QComboBox{
            background-color:#22352C;
            border:1px solid #334155;
            border-radius:12px;
            padding:10px;
            color:white;
        }

        QTableWidget{
            background-color:#182A22;
            border:1px solid #556B2F;
            border-radius:14px;
            color:white;
            gridline-color:#355847;
            alternate-background-color:#22352C;
        }

        QHeaderView::section{
            background-color:#556B2F;
            color:white;
            padding:12px;
            border:none;
            font-weight:600;
        }

        QProgressBar{
            background:#22352C;
            border-radius:10px;
            text-align:center;
        }

        QProgressBar::chunk{
            background:#556B2F;
            border-radius:14px;
        }
        """)

        # =================================================
        # MAIN LAYOUT
        # =================================================
        main_layout = QVBoxLayout()

        main_layout.setSpacing(20)

        main_layout.setContentsMargins(
            25,
            25,
            25,
            25
        )

        # =================================================
        # TITLE CARD
        # =================================================
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")

        add_shadow(title_frame)

        title_layout = QVBoxLayout()

        title = QLabel(
            "HTR Otomatisasi Digitalisasi Transaksi"
        )

        title.setAlignment(Qt.AlignCenter)

        title.setFont(
            QFont(
                "Segoe UI Semibold",
                26,
                QFont.DemiBold
            )
        )


        title_layout.addWidget(title)

        title_frame.setLayout(title_layout)

        main_layout.addWidget(title_frame)


        # =================================================
        # INFO CARD
        # =================================================
        info_frame = QFrame()

        add_shadow(info_frame)

        info_layout = QHBoxLayout()

        self.status_label = QLabel(
            "Status : Menunggu Upload"
        )

        self.total_label = QLabel(
            "Total Data : 0"
        )

        info_layout.addWidget(
            self.status_label
        )

        info_layout.addStretch()

        info_layout.addWidget(
            self.total_label
        )

        info_layout.addSpacing(20)

        info_frame.setLayout(info_layout)

        main_layout.addWidget(info_frame)

        # =================================================
        # TOP BUTTON
        # =================================================
        top_layout = QHBoxLayout()

        self.upload_btn = QPushButton(
            "📂 Upload Gambar"
        )

        self.upload_btn.clicked.connect(
            self.load_image
        )

        self.reset_btn = QPushButton(
            "🔄 Reset"
        )

        self.help_btn = QPushButton(
            "📖 Petunjuk"
        )

        self.help_btn.clicked.connect(
            self.show_help
        )

        self.reset_btn.clicked.connect(
            self.reset_all
        )

        self.search_box = QLineEdit()

        self.search_box.setPlaceholderText(
            "Cari barang..."
        )

        self.search_box.textChanged.connect(
            self.filter_table
        )


        top_layout.addWidget(
            self.upload_btn
        )

        top_layout.addWidget(
            self.reset_btn
        )

        top_layout.addWidget(
            self.help_btn
        )

        self.full_preview_btn = QPushButton(
            "🖥 Full Preview"
        )

        self.full_preview_btn.clicked.connect(
            self.open_review_window
        )

        top_layout.addWidget(
            self.full_preview_btn
        )

        top_layout.addWidget(
            self.search_box
        )

        main_layout.addLayout(top_layout)

        # =================================================
        # OCR RESULT CARD AREA
        # =================================================
        self.result_scroll = QScrollArea()

        self.result_scroll.setWidgetResizable(True)

        self.result_widget = QWidget()

        self.result_layout = QVBoxLayout()

        self.result_layout.setSpacing(15)

        self.result_widget.setLayout(
            self.result_layout
        )

        self.result_scroll.setWidget(
            self.result_widget
        )

        main_layout.addWidget(
            self.result_scroll
        )

        # =================================================
        # TABLE
        # =================================================
        table_frame = QFrame()

        add_shadow(table_frame)

        table_layout = QVBoxLayout()

        table_label = QLabel(
            "Parsing Otomatis"
        )

        table_label.setFont(
            QFont(
                "Segoe UI",
                15,
                QFont.Bold
            )
        )

        self.table = QTableWidget()

        self.table.setColumnCount(4)

        self.table.setHorizontalHeaderLabels([
            "Barang",
            "Jumlah",
            "Total",
            "Confidence"
        ])

        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.table.setAlternatingRowColors(True)

        self.table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        table_layout.addWidget(
            table_label
        )

        table_layout.addWidget(
            self.table
        )

        table_frame.setLayout(
            table_layout
        )

        main_layout.addWidget(
            table_frame
        )

        # =================================================
        # FOOTER BUTTONS
        # =================================================
        footer_layout = QHBoxLayout()

        self.save_btn = QPushButton(
            "💾 Simpan Database"
        )

        self.save_btn.clicked.connect(
            self.save_database
        )

        self.export_btn = QPushButton(
            "📄 Export TXT"
        )

        self.view_btn = QPushButton(
            "📋 Lihat Database"
        )

        self.view_btn.clicked.connect(
            self.view_database
        )

        footer_layout.addWidget(
            self.view_btn
        )

        self.export_btn.clicked.connect(
            self.export_txt
        )

        footer_layout.addWidget(
            self.save_btn
        )

        footer_layout.addWidget(
            self.export_btn
        )

        main_layout.addLayout(
            footer_layout
        )

        self.setLayout(
            main_layout
        )

    # =====================================================
    # PETUNJUK PENGGUNAAN
    # =====================================================
    def show_help(self):

        QMessageBox.information(
            self,
            "Petunjuk Penggunaan",
            """
    1. Klik Upload Gambar untuk memilih gambar transaksi.

    2. Sistem akan melakukan segmentasi dan OCR otomatis.

    3. Hasil OCR akan tampil pada halaman utama.

    4. Klik Full Preview untuk melihat seluruh hasil OCR.

    5. Periksa hasil melalui menu Review OCR.

    6. Lakukan koreksi jika terdapat kesalahan.

    7. Klik Simpan Hasil Validasi.

    8. Klik Simpan Database untuk menyimpan data transaksi.

    9. Gunakan menu Lihat Database untuk melihat histori transaksi.
            """
        )

    # =====================================================
    # RESET
    # =====================================================
    def reset_all(self):

        for i in reversed(range(self.result_layout.count())):
            widget = self.result_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.table.setRowCount(0)

        self.parsed_results = []

        self.total_label.setText(
            "Total Data : 0"
        )

        self.status_label.setText(
            "Status : Reset selesai"
        )

    # =====================================================
    # FILTER
    # =====================================================
    def filter_table(self):

        keyword = self.search_box.text().lower()

        for row in range(
            self.table.rowCount()
        ):

            item = self.table.item(row, 0)

            if item:

                self.table.setRowHidden(
                    row,
                    keyword not in item.text().lower()
                )

    # =====================================================
    # VIEW DATABASE
    # =====================================================
    def view_database(self):

        self.db_window = DatabaseWindow()

        self.db_window.show()

    # =====================================================
    # FULL PREVIEW
    # =====================================================

    def open_full_preview(self):

        window = QWidget()

        window.setWindowTitle(
            "Fullscreen OCR & Crop"
        )

        window.resize(1600, 900)

        main_layout = QHBoxLayout()

        # =================================================
        # OCR AREA
        # =================================================
        ocr_box = QTextEdit()

        ocr_box.setReadOnly(True)

        ocr_box.setFont(
            QFont("Consolas", 11)
        )

        ocr_box.setPlainText(
            self.result_box.toPlainText()
        )

        # =================================================
        # CROP AREA
        # =================================================
        crop_scroll = QScrollArea()

        crop_scroll.setWidgetResizable(True)

        crop_widget = QWidget()

        crop_layout = QVBoxLayout()

        for i, crop in enumerate(self.current_crops):

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            h, w, ch = crop_rgb.shape

            bytes_per_line = ch * w

            qt_image = QImage(
                crop_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qt_image)

            pixmap = pixmap.scaled(
                1200,
                220,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            img_label = QLabel()

            img_label.setPixmap(pixmap)

            text_label = QLabel(
                f"Baris {i+1}"
            )

            text_label.setStyleSheet("""
                font-size:16px;
                font-weight:bold;
                margin-top:15px;
                margin-bottom:5px;
            """)

            crop_layout.addWidget(text_label)

            crop_layout.addWidget(img_label)

        crop_widget.setLayout(crop_layout)

        crop_scroll.setWidget(crop_widget)

        # =================================================
        # GABUNG
        # =================================================
        main_layout.addWidget(ocr_box, 1)

        main_layout.addWidget(crop_scroll, 2)

        window.setLayout(main_layout)

        self.full_window = window

        self.full_window.showMaximized()
    
    # =====================================================
    # REVIEW PARSING WINDOW
    # =====================================================
    def open_review_window(self):

        self.review_window = QWidget()

        self.review_window.setStyleSheet("""

        QWidget{
            background-color:#0F1F17;
            color:#f8fafc;
            font-family:Segoe UI;
            font-size:16px;
        }

        QFrame{
            background:#182A22;
            border-radius:18px;
            border:1px solid #355847;
        }

        QLineEdit#reviewInput{
            background-color:#1e293b;
            border:1px solid #334155;
            border-radius:12px;
            padding:10px;
            color:white;
            font-size:16px;
            font-weight:bold;
        }

        QPushButton{
            background-color:#556B2F;
            color:white;
            border:none;
            border-radius:12px;
            padding:10px;
            font-weight:bold;
        }

        QPushButton:hover{
            background-color:#6B8E23;
        }
        QScrollArea{
            border:none;
        }

        """)

        self.review_window.setWindowTitle(
            "Review OCR & Validasi"
        )

        self.review_window.resize(1800, 950)

        main_layout = QVBoxLayout()

        # =====================================================
        # TITLE
        # =====================================================
        title = QLabel(
            "🔍 Review OCR Sebelum Disimpan"
        )

        title.setAlignment(Qt.AlignCenter)

        title.setFont(
            QFont("Segoe UI", 22, QFont.Bold)
        )

        main_layout.addWidget(title)

        # =====================================================
        # SCROLL AREA
        # =====================================================
        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        container = QWidget()

        container_layout = QVBoxLayout()

        self.review_rows = []

        # =====================================================
        # LOOP SEMUA BARIS
        # =====================================================
        for i, data in enumerate(self.parsed_results):

            row_frame = QFrame()

            row_frame.setStyleSheet("""
                QFrame{
                    background:#111827;
                    border:1px solid #334155;
                    border-radius:18px;
                    padding:12px;
                }
            """)

            row_layout = QHBoxLayout()

            row_layout.setContentsMargins(12,12,12,12)

            row_layout.setSpacing(15)

            # =================================================
            # IMAGE CROP
            # =================================================
            crop = cv2.imread(data["crop_path"])

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            h, w, ch = crop_rgb.shape

            bytes_per_line = ch * w

            qt_image = QImage(
                crop_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qt_image)

            pixmap = pixmap.scaled(
                450,
                80,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            image_label = QLabel()

            image_label.setPixmap(pixmap)

            image_label.setMinimumWidth(470)

            # =================================================
            # BARIS LABEL
            # =================================================
            title = QLabel(
                f"BARIS {i+1}"
            )

            title.setAlignment(Qt.AlignCenter)

            title.setMinimumWidth(120)

            title.setStyleSheet("""
                background:#3F5221;
                border:1px solid #6B8E23;
                border-radius:14px;
                padding:12px;
                font-size:18px;
                font-weight:bold;
                color:#EAF4EE;
                letter-spacing:1px;
            """)

            # =================================================
            # HASIL OCR EDITABLE (SATU KOLOM)
            # =================================================
            full_text = (
                f"{data['barang']} "
                f"{data['jumlah']} "
                f"{data['total']}"
            ).strip()

            text_edit = QLineEdit(full_text)

            text_edit.setMinimumHeight(55)

            text_edit.setStyleSheet("""
                QLineEdit{
                    background:#1e293b;
                    border:1px solid #334155;
                    border-radius:14px;
                    padding:12px;
                    color:white;
                    font-size:18px;
                    font-weight:bold;
                }
            """)

            # =================================================
            # CONFIDENCE
            # =================================================
            conf_label = QLabel(
                f"{data['confidence']:.2f}%"
            )

            conf_label.setAlignment(Qt.AlignCenter)

            conf_label.setMinimumWidth(120)

            conf_label.setStyleSheet("""
                color:white;
                font-size:15px;
                font-weight:bold;
            """)



            # =================================================
            # DELETE BUTTON
            # =================================================
            delete_btn = QPushButton("🗑 Hapus")

            delete_btn.setMinimumHeight(55)

            delete_btn.setStyleSheet("""
                QPushButton{
                    background:#C62828;
                    border-radius:14px;
                    font-size:16px;
                    font-weight:bold;
                    color:white;
                    padding:12px;
                }

                QPushButton:hover{
                    background:#E53935;
                }
            """)

            # =================================================
            # SIMPAN DATA
            # =================================================
            row_data = {
                "text_edit": text_edit,
                "confidence": data["confidence"],
                "crop_path": data["crop_path"],
                "original_text": data["original_text"],
                "frame": row_frame,
                "is_corrected": False
            }

            self.review_rows.append(row_data)

            # =================================================
            # DELETE FUNCTION
            # =================================================
            def delete_row(_, frame=row_frame):
                frame.hide()

            delete_btn.clicked.connect(delete_row)

            # =================================================
            # GABUNG
            # =================================================
            row_layout.addWidget(image_label, 2)

            row_layout.addWidget(title)

            row_layout.addWidget(text_edit, 3)

            row_layout.addWidget(conf_label)

            row_layout.addWidget(delete_btn)

            row_frame.setLayout(row_layout)

            container_layout.addWidget(row_frame)

        container.setLayout(container_layout)

        scroll.setWidget(container)

        main_layout.addWidget(scroll)

        # =====================================================
        # SAVE BUTTON
        # =====================================================
        save_btn = QPushButton(
            "💾 Simpan Hasil Validasi"
        )

        save_btn.setMinimumHeight(55)

        save_btn.clicked.connect(
            self.save_review_result
        )

        main_layout.addWidget(save_btn)

        self.review_window.setLayout(
            main_layout
        )

        self.review_window.showMaximized()
        
    # =====================================================
    # SAVE REVIEW RESULT
    # =====================================================
    def save_review_result(self):

        self.parsed_results = []

        self.table.setRowCount(0)

        valid_row = 0

        for row_data in self.review_rows:

            frame = row_data["frame"]

            # skip jika dihapus
            if not frame.isVisible():
                continue

            # =========================================
            # AMBIL TEXT HASIL EDIT USER
            # =========================================
            edited_text = row_data["text_edit"].text().strip()

            confidence = row_data["confidence"]

            original_text = row_data["original_text"].strip()

            # =========================================
            # PARSING ULANG
            # =========================================
            parsed = parse_text(edited_text)

            barang = parsed["barang"]

            jumlah = parsed["jumlah"]

            total = parsed["total"]

            # =========================================
            # UPDATE TABLE
            # =========================================
            self.table.insertRow(valid_row)

            self.table.setItem(
                valid_row,
                0,
                QTableWidgetItem(barang)
            )

            self.table.setItem(
                valid_row,
                1,
                QTableWidgetItem(str(jumlah))
            )

            self.table.setItem(
                valid_row,
                2,
                QTableWidgetItem(str(total))
            )

            confidence_item = QTableWidgetItem(
                f"{confidence:.2f}%"
            )

            self.table.setItem(
                valid_row,
                3,
                confidence_item
            )

            # =========================================
            # CEK APAKAH USER MELAKUKAN KOREKSI
            # =========================================
            if edited_text.lower() != original_text.lower():

                parsed_original = parse_text(original_text)

                parsed_edited = parse_text(edited_text)

                salah = parsed_original["barang"].lower()
                benar = parsed_edited["barang"].lower()

                if salah != benar:

                    cursor.execute("""
                        INSERT INTO koreksi_barang
                        (salah, benar)
                        VALUES (%s,%s)
                    """, (salah, benar))

                    conn.commit()

            # =========================================
            # SIMPAN KE parsed_results
            # =========================================
            self.parsed_results.append({
                "barang": barang,
                "jumlah": jumlah,
                "total": total,
                "confidence": confidence,
                "crop_path": row_data["crop_path"]
            })

            valid_row += 1

        QMessageBox.information(
            self,
            "Success",
            "Validasi berhasil disimpan"
        )

        self.review_window.close()

    # =====================================================
    # LOAD IMAGE
    # =====================================================
    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Gambar",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return


        for i in reversed(range(self.result_layout.count())):

            widget = self.result_layout.itemAt(i).widget()

            if widget:
                widget.deleteLater()

        self.parsed_results = []
        self.current_crops = [] 

        self.status_label.setText(
            "Status : Memproses OCR..."
        )

        crops = crop_lines(file_path)
        self.current_crops = crops

        self.table.setRowCount(
            len(crops)
        )

        total_crop = len(crops)

        loading = QProgressDialog(
            "",
            None,
            0,
            total_crop,
            self
        )

        loading.setWindowTitle("Processing OCR")

        loading.setWindowModality(Qt.WindowModal)

        loading.setCancelButton(None)

        loading.setMinimumDuration(0)

        loading.resize(500, 180)

        loading.setStyleSheet("""
            QProgressDialog{
                background:#0f172a;
                color:white;
                font-size:18px;
                border-radius:18px;
            }

            QLabel{
                color:white;
                font-size:18px;
                font-weight:bold;
                padding:20px;
            }

            QProgressBar{
                height:28px;
                border-radius:14px;
                background:#22352C;
                text-align:center;
                color:white;
                font-size:15px;
                border:1px solid #556B2F;
            }

            QProgressBar::chunk{
                background:#296746;
                border-radius:14px;
            }
        """)

        loading.setLabelText(
            "🔍 Sedang memproses tulisan tangan..."
        )

        loading.show()

        loading.setWindowTitle("Processing OCR")

        loading.setWindowModality(Qt.WindowModal)

        loading.setCancelButton(None)

        loading.setMinimumDuration(0)

        loading.show()

        for i, crop in enumerate(crops):

            crop_filename = f"crop_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            crop_path = os.path.join(
                TEMP_CROP_DIR,
                crop_filename
            )

            cv2.imwrite(crop_path, crop)

            # ==========================
            # FILTER CROP KOSONG
            # ==========================
            gray_crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            ink_pixels = np.sum(gray_crop < 180)

            # =========================================
            # AUTO CORRECT
            # =========================================
            # 1. cek jumlah pixel tinta
            if ink_pixels < 500:
                continue

            # 2. OCR
            pred_text, confidence = predict_crop(crop)

            parsed = parse_text(pred_text)

            # 3. hasil OCR tidak mengandung jumlah & total
            if parsed["jumlah"] == 0 and parsed["total"] == 0:
                continue

            parsed = parse_text(pred_text)

            if confidence < CONFIDENCE_THRESHOLD:

                if parsed["valid_barang"]:

                    parsed["status"] = "AUTO CORRECT"

                else:

                    parsed["status"] = "PERLU REVIEW"

            else:

                parsed["status"] = "VALID"

            if confidence >= CONFIDENCE_THRESHOLD:

                parsed["need_review"] = False

            else:

                parsed["need_review"] = not parsed["valid_barang"]

            parsed["confidence"] = confidence
            parsed["crop_path"] = crop_path
            parsed["original_text"] = pred_text

            self.parsed_results.append(parsed)

            # =========================================
            # PREVIEW CROP
            # =========================================
            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            h, w, ch = crop_rgb.shape

            bytes_per_line = ch * w

            qt_image = QImage(
                crop_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qt_image)

            pixmap = pixmap.scaled(
                450,
                80,
            )

            # =========================================
            # CARD OCR
            # =========================================
            card = QFrame()

            card.setStyleSheet("""
                QFrame{
                    background:#182A22;
                    border:1px solid #355847;
                    border-radius:18px;
                    padding:12px;
                }
            """)

            card_layout = QHBoxLayout()
            card_layout.setContentsMargins(10,10,10,10)
            card_layout.setSpacing(15)

            # =========================================
            # IMAGE
            # =========================================
            image_label = QLabel()

            image_label.setPixmap(pixmap)

            image_label.setMinimumWidth(470)

            # =========================================
            # TEXT AREA
            # =========================================
            text_layout = QHBoxLayout()

            text_layout.setAlignment(Qt.AlignVCenter)

            title = QLabel(
                f"BARIS {i+1}"
            )

            title.setStyleSheet("""
                font-size:18px;
                font-weight:bold;
            """)

            preview_text = f"{parsed['barang']} {parsed['jumlah']} {parsed['total']}"
            pred_label = QLabel(preview_text)

            pred_label.setWordWrap(False)

            pred_label.setMinimumWidth(300)

            pred_label.setStyleSheet("""
                font-size:20px;
                color:white;
                font-weight:bold;
            """)

            conf_label = QLabel(
                f"Confidence : {confidence:.2f}%"
            )

            color = "white"

            text_layout.addWidget(title)

            text_layout.addSpacing(20)

            text_layout.addWidget(pred_label)

            text_layout.addSpacing(20)

            text_layout.addWidget(conf_label)

            card_layout.addWidget(image_label, 2)

            card_layout.addLayout(text_layout, 3)

            card.setLayout(card_layout)

            self.result_layout.addWidget(card)

            self.table.setItem(
                i,
                0,
                QTableWidgetItem(
                    parsed["barang"]
                )
            )

            self.table.setItem(
                i,
                1,
                QTableWidgetItem(
                    str(parsed["jumlah"])
                )
            )

            self.table.setItem(
                i,
                2,
                QTableWidgetItem(
                    str(parsed["total"])
                )
            )

            self.table.setItem(
                i,
                3,
                QTableWidgetItem(
                    f"{confidence:.2f}%"
                )
            )

            confidence_item = QTableWidgetItem(
                f"{confidence:.2f}%"
            )

            self.table.setItem(
                i,
                3,
                confidence_item
            )
            loading.setValue(i + 1)

            QApplication.processEvents()

        loading.close()

        self.total_label.setText(
            f"Total Data : {len(self.parsed_results)}"
        )

        self.status_label.setText(
            "Status : OCR selesai"
        )

    # =====================================================
    # SAVE DATABASE
    # =====================================================
    def save_database(self):

        if len(self.parsed_results) == 0:

            QMessageBox.warning(
                self,
                "Warning",
                "Tidak ada data"
            )

            return

        for data in self.parsed_results:

            cursor.execute('''
            INSERT INTO transaksi (
                tanggal,
                barang,
                jumlah,
                total
            )
            VALUES (%s, %s, %s, %s)
            ''', (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                data["barang"],
                data["jumlah"],
                data["total"]
            ))

        conn.commit()

        QMessageBox.information(
            self,
            "Success",
            "Data berhasil disimpan"
        )
    # =====================================================
    # EXPORT TXT
    # =====================================================
    def export_txt(self):

        if len(self.parsed_results) == 0:

            QMessageBox.warning(
                self,
                "Warning",
                "Tidak ada data"
            )

            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "hasil_ocr.txt",
            "Text Files (*.txt)"
        )

        if not file_path:
            return

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("HASIL OCR\n")
            f.write("="*50 + "\n\n")

            for i, data in enumerate(
                self.parsed_results
            ):

                f.write(
                    f"Data {i+1}\n"
                )

                f.write(
                    f"Barang : {data['barang']}\n"
                )

                f.write(
                    f"Jumlah : {data['jumlah']}\n"
                )

                f.write(
                    f"Total : {data['total']}\n"
                )

                f.write(
                    f"Confidence : {data['confidence']:.2f}%\n"
                )

                f.write(
                    "-"*40 + "\n"
                )

        QMessageBox.information(
            self,
            "Success",
            "File berhasil diexport"
        )

# =========================================================
# RUN APP
# =========================================================
app = QApplication(sys.argv)

window = MainWindow()

window.show()

sys.exit(app.exec_())