"""
Sistema de Reconhecimento Facial - Sci-Fi HUD Interface
Versão com interface futurista estilo Iron Man/Jarvis
"""

import cv2
import numpy as np
from datetime import datetime
import os
import pickle
import sqlite3
import time
import csv
import traceback
from PIL import Image, ImageDraw, ImageFont
import platform
import face_recognition

class FaceRecognitionSystem:
    """Sistema de reconhecimento facial com interface Sci-Fi HUD."""
    
    # Cores Sci-Fi (BGR para OpenCV)
    COLOR_CYAN = (255, 255, 0)  # Electric Blue/Cyan
    COLOR_ORANGE = (0, 165, 255)  # Alert Orange
    COLOR_GREEN = (0, 255, 0)  # Success Green
    COLOR_RED = (0, 0, 255)  # Error Red
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    
    def __init__(self, headless=False):
        """Inicializa o sistema.
        headless=True: sem câmera OpenCV (uso com servidor web / API).
        """
        self.headless = headless
        self.cap = None
        self.face_cascade = None
        self.known_face_features = []
        self.known_face_names = []
        
        # Arquivos
        self.face_encodings_file = "face_encodings_final.pkl"
        self.db_file = "recognitions_final.db"
        self.fotos_reais_dir = "fotos_reais"
        
        # Controle
        self.current_face = None
        self.last_recognition_time = 0
        self.recognition_cooldown = 2
        self.cadastro_pending = False
        self.input_nome = ""
        self.input_mode = False
        self.delete_mode = False
        self.cadastro_frame = None
        self.cadastro_face_rect = None
        self.last_recognized_name = None
        self.last_recognized_confidence = 0.0
        self.recognition_display_time = 0
        self.auto_recognition = False
        self.multiple_samples = {}
        self.recent_recognitions = []
        self.face_box_alpha = 0.0
        self.stable_face = None
        self.face_stability_count = 0
        
        # HUD Animation
        self.scan_line_pos = 0
        self.scan_line_direction = 1
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Performance optimization
        self.frame_count = 0
        self.detection_skip_frames = 2  # Processa detecção a cada 2 frames
        self.last_known_face = None  # Última face conhecida para frames pulados
        
        # Fontes PIL
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self._init_fonts()
        
        # Estatísticas
        self.stats = {
            'total_cadastros': 0,
            'total_reconhecimentos': 0,
            'fotos_salvas': 0
        }
        
        # Inicializa sistema
        self.init_system()
    
    def _init_fonts(self):
        """Inicializa fontes monospace do sistema."""
        try:
            # Tenta usar fontes monospace do sistema
            if platform.system() == "Windows":
                font_paths = [
                    "C:/Windows/Fonts/consola.ttf",
                    "C:/Windows/Fonts/cour.ttf",
                    "C:/Windows/Fonts/courbd.ttf"
                ]
            elif platform.system() == "Darwin":  # macOS
                font_paths = [
                    "/Library/Fonts/Courier New.ttf",
                    "/System/Library/Fonts/Menlo.ttc"
                ]
            else:  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
                ]
            
            # Tenta carregar fonte
            font_loaded = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.font_large = ImageFont.truetype(font_path, 32)
                        self.font_medium = ImageFont.truetype(font_path, 20)
                        self.font_small = ImageFont.truetype(font_path, 16)
                        font_loaded = True
                        break
                    except:
                        continue
            
            # Fallback para fonte padrão
            if not font_loaded:
                self.font_large = ImageFont.load_default()
                self.font_medium = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
        except:
            # Fallback absoluto
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def draw_text_pil(self, frame, text, position, font, color, shadow=True):
        """
        Desenha texto usando PIL para anti-aliasing e fontes customizadas.
        OTIMIZADO: Usa cache e apenas para textos importantes.
        
        Args:
            frame: Frame OpenCV (BGR)
            text: Texto a desenhar
            position: (x, y) posição
            font: Fonte PIL
            color: Cor BGR
            shadow: Se deve desenhar sombra
        """
        # Para performance, usa cv2.putText para textos simples
        # PIL apenas para textos grandes/importantes
        try:
            # Se fonte é default (pequena), usa cv2.putText (mais rápido)
            if font == self.font_small or font == ImageFont.load_default():
                if shadow:
                    cv2.putText(frame, text, (position[0] + 2, position[1] + 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, text, position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                return
            
            # Para textos grandes, usa PIL
            color_rgb = (color[2], color[1], color[0])
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            if shadow:
                shadow_pos = (position[0] + 2, position[1] + 2)
                draw.text(shadow_pos, text, font=font, fill=(0, 0, 0, 200))
            
            draw.text(position, text, font=font, fill=color_rgb)
            frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Fallback sempre para cv2.putText
            if shadow:
                cv2.putText(frame, text, (position[0] + 2, position[1] + 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_corner_brackets(self, frame, x, y, w, h, color, thickness=2, length=20):
        """
        Desenha cantos estilo bracket minimalista (apenas os cantos, não retângulo completo).
        Estilo Apple Face ID - pequeno e elegante.
        
        Args:
            frame: Frame OpenCV
            x, y, w, h: Coordenadas e dimensões do retângulo
            color: Cor BGR
            thickness: Espessura da linha (padrão: 2px)
            length: Comprimento dos braços dos cantos (padrão: 20px)
        """
        # Canto superior esquerdo
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
        
        # Canto superior direito
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        
        # Canto inferior esquerdo
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        
        # Canto inferior direito
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)
    
    def draw_facial_landmarks(self, frame, face_rect, scale_factor=2):
        """
        Desenha landmarks faciais estilo Apple Face ID (3D mesh/scanning effect).
        
        Args:
            frame: Frame OpenCV (BGR)
            face_rect: (x, y, w, h) coordenadas da face no frame completo
            scale_factor: Fator de escala usado na detecção (para escalar landmarks de volta)
        """
        try:
            x, y, w, h = face_rect
            
            # Converte frame completo para RGB (face_recognition usa RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # face_recognition espera formato (top, right, bottom, left)
            # top = y, right = x + w, bottom = y + h, left = x
            face_location = (y, x + w, y + h, x)
            
            # Detecta landmarks no frame completo
            landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
            
            if not landmarks or len(landmarks) == 0:
                return
            
            landmarks_dict = landmarks[0]
            
            # Cor sutil para landmarks (branco semi-transparente)
            landmark_color = (255, 255, 255)  # BGR white
            
            # Desenha landmarks
            for feature_name, points in landmarks_dict.items():
                # Desenha pontos pequenos e linhas conectando-os
                for i, point in enumerate(points):
                    px, py = point
                    # Desenha ponto pequeno (1px)
                    cv2.circle(frame, (px, py), 1, landmark_color, 1)
                    
                    # Conecta pontos adjacentes com linha fina (1px)
                    if i < len(points) - 1:
                        next_point = points[i + 1]
                        cv2.line(frame, (px, py), next_point, landmark_color, 1)
                    
                    # Conecta último ponto ao primeiro para features fechadas
                    if i == len(points) - 1 and feature_name in ['chin', 'top_lip', 'bottom_lip']:
                        first_point = points[0]
                        cv2.line(frame, (px, py), first_point, landmark_color, 1)
        
        except Exception as e:
            # Silenciosamente ignora erros de landmark detection
            pass
    
    def draw_scanning_line(self, frame, x, y, w, h):
        """
        Desenha linha de scanning animada sobre a face.
        
        Args:
            frame: Frame OpenCV
            x, y, w, h: Coordenadas da face
        """
        # Atualiza posição da linha
        scan_height = y + int(self.scan_line_pos * h)
        scan_height = max(y, min(y + h, scan_height))
        
        # Desenha linha com gradiente
        line_thickness = 3
        for i in range(line_thickness):
            alpha = 1.0 - (i / line_thickness) * 0.5
            color = tuple(int(c * alpha) for c in self.COLOR_CYAN)
            cv2.line(frame, (x, scan_height + i), (x + w, scan_height + i), color, 1)
        
        # Atualiza posição para próxima frame
        self.scan_line_pos += 0.05 * self.scan_line_direction
        if self.scan_line_pos >= 1.0:
            self.scan_line_direction = -1
        elif self.scan_line_pos <= 0.0:
            self.scan_line_direction = 1
    
    def draw_header(self, frame):
        """Desenha header superior com status, data/hora e FPS."""
        h, w = frame.shape[:2]
        header_height = 50
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Borda inferior
        cv2.line(frame, (0, header_height), (w, header_height), self.COLOR_CYAN, 2)
        
        # Status (usa cv2.putText para performance)
        status_text = "SYSTEM ONLINE"
        cv2.putText(frame, status_text, (22, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, status_text, (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GREEN, 2)
        
        # Data/Hora
        now = datetime.now()
        datetime_text = now.strftime("%Y-%m-%d %H:%M:%S")
        text_size = cv2.getTextSize(datetime_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = w - text_size[0] - 150
        cv2.putText(frame, datetime_text, (text_x + 2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, datetime_text, (text_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
        
        # FPS
        fps_text = f"FPS: {self.current_fps}"
        fps_x = w - 120
        cv2.putText(frame, fps_text, (fps_x + 2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, fps_text, (fps_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
    
    def draw_sidebar(self, frame):
        """Desenha sidebar direito com controles e logs (otimizado)."""
        h, w = frame.shape[:2]
        sidebar_width = 300
        sidebar_x = w - sidebar_width
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (sidebar_x, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Borda esquerda
        cv2.line(frame, (sidebar_x, 0), (sidebar_x, h), self.COLOR_CYAN, 2)
        
        # Título (usa cv2.putText)
        title_y = 60
        cv2.putText(frame, "CONTROLS", (sidebar_x + 12, title_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, "CONTROLS", (sidebar_x + 10, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
        
        # Lista de controles (usa cv2.putText para performance)
        controls = [
            ("[C]", "Cadastrar"),
            ("[SPACE]", "Reconhecer"),
            ("[D]", "Deletar"),
            ("[M]", "Adicionar amostra"),
            ("[A]", "Auto ON/OFF"),
            ("[S]", "Estatisticas"),
            ("[E]", "Exportar CSV"),
            ("[H]", "Historico"),
            ("[Q]", "Sair")
        ]
        
        start_y = title_y + 40
        for i, (key, desc) in enumerate(controls):
            y_pos = start_y + (i * 25)
            # Tecla
            cv2.putText(frame, key, (sidebar_x + 12, y_pos + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, key, (sidebar_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ORANGE, 1)
            # Descrição
            cv2.putText(frame, desc, (sidebar_x + 62, y_pos + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, desc, (sidebar_x + 60, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        # Status do sistema
        status_y = start_y + len(controls) * 25 + 30
        cv2.putText(frame, "SYSTEM STATUS", (sidebar_x + 12, status_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, "SYSTEM STATUS", (sidebar_x + 10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
        
        # Estatísticas
        stats_y = status_y + 35
        stats = [
            f"Pessoas: {len(self.known_face_names)}",
            f"Fotos: {self.stats['fotos_salvas']}",
            f"Reconhecimentos: {self.stats['total_reconhecimentos']}"
        ]
        
        for i, stat in enumerate(stats):
            y_pos = stats_y + (i * 20)
            cv2.putText(frame, stat, (sidebar_x + 12, y_pos + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, stat, (sidebar_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        # Modo atual
        mode_y = stats_y + len(stats) * 20 + 20
        if self.delete_mode:
            mode_text = "MODE: DELETE"
            mode_color = self.COLOR_RED
        elif self.input_mode:
            mode_text = "MODE: INPUT"
            mode_color = self.COLOR_ORANGE
        elif self.auto_recognition:
            mode_text = "MODE: AUTO"
            mode_color = self.COLOR_GREEN
        else:
            mode_text = "MODE: MANUAL"
            mode_color = self.COLOR_CYAN
        
        cv2.putText(frame, mode_text, (sidebar_x + 12, mode_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, mode_text, (sidebar_x + 10, mode_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Input atual
        if self.input_mode or self.delete_mode:
            input_y = mode_y + 35
            input_text = f"INPUT: {self.input_nome}_"
            cv2.putText(frame, input_text, (sidebar_x + 12, input_y + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, input_text, (sidebar_x + 10, input_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ORANGE, 1)
        
        # Logs recentes (últimos 3)
        if self.recent_recognitions:
            logs_y = mode_y + 80
            cv2.putText(frame, "RECENT LOGS", (sidebar_x + 12, logs_y + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, "RECENT LOGS", (sidebar_x + 10, logs_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
            
            for i, (name, conf, ts) in enumerate(self.recent_recognitions[:3]):
                log_y = logs_y + 30 + (i * 20)
                log_text = f"{name[:10]:10s} {conf*100:.0f}%"
                cv2.putText(frame, log_text, (sidebar_x + 12, log_y + 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(frame, log_text, (sidebar_x + 10, log_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1)
    
    def draw_face_hud(self, frame, x, y, w, h, is_recognized=False):
        """
        Desenha HUD sobre a face detectada (otimizado).
        Estilo Apple Face ID - minimalista e elegante.
        
        Args:
            frame: Frame OpenCV
            x, y, w, h: Coordenadas da face
            is_recognized: Se a face foi reconhecida
        """
        # Escolhe cor baseado no status (sutil cyan ou branco)
        if is_recognized:
            primary_color = self.COLOR_CYAN  # (255, 255, 0) em BGR
            secondary_color = self.COLOR_GREEN
        else:
            primary_color = (255, 255, 255)  # Branco sutil para não reconhecido
            secondary_color = (255, 255, 255)
        
        # Desenha corner brackets minimalistas (20px, 2px)
        self.draw_corner_brackets(frame, x, y, w, h, primary_color, thickness=2, length=20)
        
        # Painel de informação acima da face
        panel_height = 60
        panel_y = y - panel_height - 10
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, panel_y), (x + w + 5, y - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Borda do painel
        cv2.rectangle(frame, (x - 5, panel_y), (x + w + 5, y - 5), primary_color, 2)
        
        # Texto (usa cv2.putText para performance, PIL apenas para nome grande)
        if is_recognized:
            name_text = self.last_recognized_name.upper()
            conf_text = f"{self.last_recognized_confidence*100:.0f}%"
            
            # Nome (usa PIL apenas para texto grande)
            self.draw_text_pil(frame, name_text, (x + 5, panel_y + 15), 
                              self.font_large, secondary_color)
            
            # Confiança (usa cv2.putText)
            cv2.putText(frame, conf_text, (x + 7, panel_y + 47), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, conf_text, (x + 5, panel_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)
        else:
            status_text = "FACE DETECTED"
            cv2.putText(frame, status_text, (x + 7, panel_y + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, status_text, (x + 5, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, primary_color, 2)
    
    def draw_hud(self, frame):
        """
        Desenha toda a interface HUD.
        
        Args:
            frame: Frame OpenCV
        """
        # Header
        self.draw_header(frame)
        
        # Sidebar
        self.draw_sidebar(frame)
        
        # Face HUD (se houver face)
        if self.current_face:
            x, y, w, h = self.current_face
            is_recognized = (self.last_recognized_name and 
                           time.time() < self.recognition_display_time)
            self.draw_face_hud(frame, x, y, w, h, is_recognized)
    
    def process_frame(self, frame):
        """
        Processa frame: detecta faces e realiza reconhecimento.
        Otimizado para processar detecção a cada N frames.
        
        Args:
            frame: Frame OpenCV em resolução completa
        """
        self.frame_count += 1
        
        # Processa detecção apenas a cada N frames para performance
        # Usa última face conhecida nos frames pulados
        if self.frame_count % self.detection_skip_frames == 0:
            # Detecta faces (usa escala 1/4 internamente)
            faces = self.detect_faces(frame)
            
            # Processa apenas a maior face válida
            if len(faces) > 0:
                # Seleciona maior face
                faces_with_area = [(f, f[2] * f[3]) for f in faces]
                faces_with_area.sort(key=lambda x: x[1], reverse=True)
                best_face = faces_with_area[0][0]
                x, y, w, h = best_face
                
                # Suaviza transições
                if self.stable_face is not None:
                    sx, sy, sw, sh = self.stable_face
                    diff = abs(x - sx) + abs(y - sy) + abs(w - sw) + abs(h - sh)
                    if diff < 30:
                        self.face_stability_count += 1
                        if self.face_stability_count > 3:
                            x, y, w, h = self.stable_face
                    else:
                        self.face_stability_count = 0
                        self.stable_face = (x, y, w, h)
                else:
                    self.stable_face = (x, y, w, h)
                    self.face_stability_count = 0
                
                # Salva última face conhecida
                self.last_known_face = (x, y, w, h)
                
                # Reconhecimento automático (apenas quando detecta nova face)
                if (self.auto_recognition and 
                    time.time() - self.last_recognition_time > self.recognition_cooldown):
                    name, confidence = self.recognize_face_simple(frame, (x, y, w, h))
                    if name:
                        self.last_recognized_name = name
                        self.last_recognized_confidence = confidence
                        self.recognition_display_time = time.time() + 3
                        self.last_recognition_time = time.time()
                        self.log_recognition(name, confidence)
                        print(f"🤖 AUTO: {name} ({confidence:.3f})")
                
                # Atualiza face atual
                self.current_face = (x, y, w, h)
            else:
                # Reset quando não há face
                self.stable_face = None
                self.face_stability_count = 0
                self.face_box_alpha = 0.0
                self.last_known_face = None
                self.current_face = None
        else:
            # Usa última face conhecida para manter desenho suave
            if self.last_known_face is not None:
                self.current_face = self.last_known_face
            else:
                self.current_face = None
        
        # Calcula FPS
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def handle_keys(self, key):
        """
        Processa teclas pressionadas.
        
        Args:
            key: Código da tecla
        """
        if key == ord('q'):
            return False  # Sair
        
        elif self.delete_mode:
            if key == 13:  # ENTER
                nome = self.input_nome.strip()
                if nome:
                    self.deletar_pessoa(nome)
                self.delete_mode = False
                self.input_nome = ""
            elif key == 27:  # ESC
                print("❌ Exclusão cancelada")
                self.delete_mode = False
                self.input_nome = ""
            elif key == 8:  # BACKSPACE
                self.input_nome = self.input_nome[:-1] if self.input_nome else ""
            elif 32 <= key <= 126:
                self.input_nome += chr(key)
        
        elif self.input_mode:
            if key == 13:  # ENTER
                if self.cadastro_frame is not None and self.cadastro_face_rect is not None:
                    nome = self.input_nome.strip()
                    adicionar = nome in self.known_face_names
                    self.processar_cadastro(self.cadastro_frame, self.cadastro_face_rect, 
                                          adicionar_amostra=adicionar)
                    self.cadastro_pending = False
                self.input_mode = False
                self.cadastro_frame = None
                self.cadastro_face_rect = None
            elif key == 27:  # ESC
                print("❌ Operação cancelada")
                self.input_mode = False
                self.input_nome = ""
                self.cadastro_pending = False
                self.cadastro_frame = None
                self.cadastro_face_rect = None
            elif key == 8:  # BACKSPACE
                self.input_nome = self.input_nome[:-1] if self.input_nome else ""
            elif 32 <= key <= 126:
                self.input_nome += chr(key)
        
        elif key == ord('c'):
            if self.current_face:
                self.cadastrar_pessoa(self.current_frame, self.current_face)
            else:
                print("❌ Nenhuma face detectada")
        
        elif key == ord('m'):
            if self.current_face:
                print("\n📸 ADICIONAR AMOSTRA - Digite o nome")
                self.input_mode = True
                self.input_nome = ""
                self.cadastro_frame = self.current_frame.copy()
                self.cadastro_face_rect = self.current_face
                self.cadastro_pending = True
            else:
                print("❌ Nenhuma face detectada")
        
        elif key == ord(' '):
            if (self.current_face and 
                time.time() - self.last_recognition_time > self.recognition_cooldown):
                print("\n🔍 Reconhecendo...")
                name, confidence = self.recognize_face_simple(self.current_frame, 
                                                             self.current_face)
                if name:
                    print(f"✅ {name} ({confidence:.3f})")
                    self.last_recognized_name = name
                    self.last_recognized_confidence = confidence
                    self.recognition_display_time = time.time() + 3
                    self.log_recognition(name, confidence)
                    self.last_recognition_time = time.time()
                else:
                    print("❌ Não reconhecido")
            elif not self.current_face:
                print("❌ Nenhuma face detectada")
            else:
                print("⏳ Aguarde...")
        
        elif key == ord('d'):
            if not self.delete_mode:
                print("\n🗑️  DELETE MODE - Digite o nome")
                self.delete_mode = True
                self.input_nome = ""
        
        elif key == ord('a'):
            self.auto_recognition = not self.auto_recognition
            status = "ATIVADO" if self.auto_recognition else "DESATIVADO"
            print(f"🤖 Auto: {status}")
        
        elif key == ord('s'):
            self.show_statistics()
        
        elif key == ord('e'):
            print("\n📤 Exportando...")
            csv_file = self.export_to_csv()
            if csv_file:
                print(f"✅ {os.path.abspath(csv_file)}")
        
        elif key == ord('h'):
            print("\n" + "="*60)
            print("📜 HISTÓRICO")
            print("="*60)
            if self.recent_recognitions:
                for i, (name, conf, ts) in enumerate(self.recent_recognitions[:10], 1):
                    print(f"{i}. {name} - {conf*100:.1f}% - {ts.strftime('%d/%m/%Y %H:%M:%S')}")
            else:
                print("Nenhum registro.")
            print("="*60)
        
        return True  # Continuar
    
    # ========== MÉTODOS ORIGINAIS (mantidos) ==========
    
    def init_system(self):
        """Inicializa todos os componentes."""
        print("🚀 Iniciando Sistema Sci-Fi HUD...")
        
        os.makedirs(self.fotos_reais_dir, exist_ok=True)
        
        # Câmera (apenas modo desktop)
        if not self.headless:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Erro ao abrir câmera")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = None
            print("🌐 Modo headless (reconhecimento via rede / API)")
        
        # Detector de faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("❌ Erro ao carregar detector")
        
        # Banco de dados
        self.init_database()
        
        # Carrega dados
        self.load_face_encodings()
        
        print("✅ Sistema inicializado!")
        self.show_info()
    
    def init_database(self):
        """Inicializa banco de dados."""
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL
            )
        ''')
        self.conn.commit()
        print(f"💾 DB: {self.db_file}")
    
    def recodificar_pessoa_de_foto(self, nome):
        """Recodifica pessoa usando foto salva com face_recognition (alta qualidade)."""
        try:
            person_dir = os.path.join(self.fotos_reais_dir, nome)
            if not os.path.exists(person_dir):
                return None
            
            fotos = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not fotos:
                return None
            
            foto_path = os.path.join(person_dir, fotos[0])
            img = cv2.imread(foto_path)
            if img is None:
                return None
            
            faces = self.detect_faces(img)
            if len(faces) == 0:
                return None
            
            face_rect = faces[0]
            # Usa face_recognition com num_jitters=3 para melhor qualidade
            features = self.extract_face_encoding_face_recognition(img, face_rect, num_jitters=3)
            return features
        except Exception as e:
            print(f"⚠️  Erro ao recodificar {nome}: {e}")
            return None
    
    def load_face_encodings(self):
        """Carrega codificações existentes."""
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data.get('features', [])
                    self.known_face_names = data.get('names', [])
                    self.multiple_samples = data.get('multiple_samples', {})
                
                # Recodifica faces antigas
                recodificadas = 0
                for i, (name, features) in enumerate(zip(self.known_face_names[:], 
                                                        self.known_face_features[:])):
                    if features is not None and len(features) == 64:
                        print(f"🔄 Recodificando {name}...")
                        new_features = self.recodificar_pessoa_de_foto(name)
                        if new_features is not None:
                            self.known_face_features[i] = new_features
                            recodificadas += 1
                
                if recodificadas > 0:
                    self.save_face_encodings()
                    print(f"💾 {recodificadas} recodificada(s)")
                
                print(f"✅ {len(self.known_face_names)} pessoas carregadas")
            except Exception as e:
                print(f"⚠️  Erro: {e}")
                self.known_face_features = []
                self.known_face_names = []
        else:
            print("⚠️  Nenhuma pessoa cadastrada")
    
    def save_face_encodings(self):
        """Salva codificações."""
        if not self.known_face_features:
            return
        
        data = {
            'features': self.known_face_features,
            'names': self.known_face_names,
            'multiple_samples': self.multiple_samples
        }
        
        with open(self.face_encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Codificações salvas")
    
    def validate_face(self, face_rect, frame_shape):
        """Valida se a face detectada é realmente uma face."""
        x, y, w, h = face_rect
        height, width = frame_shape[:2]
        
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        
        if w < 80 or h < 80:
            return False
        
        if y > height * 0.6:
            return False
        
        face_area = w * h
        frame_area = width * height
        if face_area < frame_area * 0.01:
            return False
        
        return True
    
    def filter_overlapping_faces(self, faces):
        """Remove faces sobrepostas."""
        if len(faces) <= 1:
            return faces
        
        faces_with_area = [(f, f[2] * f[3]) for f in faces]
        faces_with_area.sort(key=lambda x: x[1], reverse=True)
        
        filtered = []
        for face, area in faces_with_area:
            x, y, w, h = face
            overlap = False
            
            for fx, fy, fw, fh in filtered:
                overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
                overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > (w * h * 0.3):
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(face)
        
        return np.array(filtered) if filtered else np.array([])
    
    def detect_faces(self, frame):
        """
        Detecta faces no frame usando escala 1/2 para balance entre performance e close-range accuracy.
        
        Args:
            frame: Frame OpenCV em resolução completa
            
        Returns:
            Array de faces detectadas com coordenadas escaladas para o frame original
        """
        # Redimensiona para 1/2 da resolução para detecção rápida mas precisa
        # 0.5x mantém detalhes suficientes para faces próximas
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta faces na imagem pequena (mais rápido que full-res)
        # Ajusta minSize proporcionalmente (80/2 = 40)
        faces_small = self.face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(40, 40),  # Escalado proporcionalmente (era 20 para 0.25x)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces_small) == 0:
            return np.array([])
        
        # Escala as coordenadas de volta para o frame original (x2)
        faces = []
        for (x, y, w, h) in faces_small:
            # Multiplica por 2 para voltar à escala original (era x4 para 0.25x)
            x_scaled = int(x * 2)
            y_scaled = int(y * 2)
            w_scaled = int(w * 2)
            h_scaled = int(h * 2)
            faces.append((x_scaled, y_scaled, w_scaled, h_scaled))
        
        faces = np.array(faces)
        
        # Valida faces no frame original
        valid_faces = []
        for face in faces:
            if self.validate_face(face, frame.shape):
                valid_faces.append(face)
        
        if len(valid_faces) == 0:
            return np.array([])
        
        valid_faces = self.filter_overlapping_faces(valid_faces)
        return np.array(valid_faces)
    
    def extract_face_encoding_face_recognition(self, frame, face_rect, num_jitters=2):
        """
        Extrai encoding facial usando face_recognition library (alta precisão).
        
        Args:
            frame: Frame OpenCV (BGR)
            face_rect: (x, y, w, h) coordenadas da face
            num_jitters: Número de jitters para melhorar qualidade (padrão: 2)
            
        Returns:
            Encoding numpy array ou None
        """
        try:
            x, y, w, h = face_rect
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # Converte para RGB (face_recognition usa RGB)
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Detecta encoding com num_jitters para melhor qualidade
            # face_recognition espera formato (top, right, bottom, left)
            face_location = (0, w, h, 0)
            encodings = face_recognition.face_encodings(rgb_face, [face_location], num_jitters=num_jitters)
            
            if encodings and len(encodings) > 0:
                return encodings[0]
            
            return None
        except Exception as e:
            print(f"❌ Erro ao extrair encoding: {e}")
            return None
    
    def extract_face_features_simple(self, frame, face_rect):
        """
        Extrai características simples da face (método legado para compatibilidade).
        Agora usa face_recognition por padrão.
        """
        # Tenta usar face_recognition primeiro
        encoding = self.extract_face_encoding_face_recognition(frame, face_rect, num_jitters=1)
        if encoding is not None:
            return encoding
        
        # Fallback para método antigo se necessário
        try:
            x, y, w, h = face_rect
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            face_resized = cv2.resize(face_roi, (128, 128))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            normalized = cv2.equalizeHist(gray_face)
            
            hist_global = cv2.calcHist([normalized], [0], None, [128], [0, 256])
            
            h_center = normalized.shape[0] // 2
            w_center = normalized.shape[1] // 2
            
            top_left = normalized[0:h_center, 0:w_center]
            top_right = normalized[0:h_center, w_center:]
            bottom_left = normalized[h_center:, 0:w_center]
            bottom_right = normalized[h_center:, w_center:]
            
            hist_tl = cv2.calcHist([top_left], [0], None, [64], [0, 256])
            hist_tr = cv2.calcHist([top_right], [0], None, [64], [0, 256])
            hist_bl = cv2.calcHist([bottom_left], [0], None, [64], [0, 256])
            hist_br = cv2.calcHist([bottom_right], [0], None, [64], [0, 256])
            
            features = np.concatenate([
                hist_global.flatten(),
                hist_tl.flatten(),
                hist_tr.flatten(),
                hist_bl.flatten(),
                hist_br.flatten()
            ])
            
            return features
        except Exception as e:
            print(f"❌ Erro ao extrair características: {e}")
            return None
    
    def save_photo(self, nome, frame, face_rect):
        """Salva foto real."""
        try:
            person_dir = os.path.join(self.fotos_reais_dir, nome)
            os.makedirs(person_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{nome}_{timestamp}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            cv2.imwrite(filepath, frame)
            self.stats['fotos_salvas'] += 1
            print(f"📸 Foto salva: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Erro ao salvar foto: {e}")
            return None
    
    def processar_cadastro(self, frame_copy, face_rect_copy, adicionar_amostra=False):
        """Processa o cadastro após nome ser digitado."""
        nome = self.input_nome.strip()
        self.input_nome = ""
        self.input_mode = False
        
        if not nome:
            print("❌ Nome não pode estar vazio")
            return
        
        if nome not in self.known_face_names and adicionar_amostra:
            print(f"❌ '{nome}' não cadastrada")
            return
        
        if nome in self.known_face_names and not adicionar_amostra:
            print(f"ℹ️  '{nome}' já cadastrada. Use 'm' para adicionar amostras.")
            return
        
        print("\n" + "="*50)
        if adicionar_amostra:
            print(f"📸 ADICIONANDO AMOSTRA: {nome}")
        else:
            print(f"🔍 CADASTRANDO: {nome}")
        print("="*50)
        
        # Usa face_recognition com num_jitters=2-3 para alta qualidade durante cadastro
        features = self.extract_face_encoding_face_recognition(frame_copy, face_rect_copy, num_jitters=3)
        
        if features is not None:
            if adicionar_amostra:
                if nome not in self.multiple_samples:
                    self.multiple_samples[nome] = []
                self.multiple_samples[nome].append(features)
                print(f"✅ Nova amostra para '{nome}'!")
            else:
                self.known_face_features.append(features)
                self.known_face_names.append(nome)
                self.multiple_samples[nome] = []
                self.stats['total_cadastros'] += 1
            
            self.save_photo(nome, frame_copy, face_rect_copy)
            self.save_face_encodings()
            
            print(f"✅ Concluído! {len(features)} dimensões")
            print(f"👥 Total: {len(self.known_face_names)}")
            print("="*50)
        else:
            print("❌ Erro ao extrair características")
    
    def cadastrar_web(self, frame_copy, face_rect_copy, nome, adicionar_amostra=False):
        """Cadastro chamado pela API web (sem fluxo de teclado). Retorna (ok, mensagem)."""
        nome = (nome or "").strip()
        if not nome:
            return False, "Nome não pode estar vazio"
        if nome not in self.known_face_names and adicionar_amostra:
            return False, f"Pessoa '{nome}' não está cadastrada"
        if nome in self.known_face_names and not adicionar_amostra:
            return False, f"'{nome}' já cadastrada; use amostra extra"
        features = self.extract_face_encoding_face_recognition(
            frame_copy, face_rect_copy, num_jitters=3
        )
        if features is None:
            return False, "Não foi possível extrair características da face"
        if adicionar_amostra:
            if nome not in self.multiple_samples:
                self.multiple_samples[nome] = []
            self.multiple_samples[nome].append(features)
        else:
            self.known_face_features.append(features)
            self.known_face_names.append(nome)
            self.multiple_samples[nome] = []
            self.stats["total_cadastros"] += 1
        self.save_photo(nome, frame_copy, face_rect_copy)
        self.save_face_encodings()
        return True, "Cadastro concluído"
    
    def cadastrar_pessoa(self, frame, face_rect):
        """Inicia modo de entrada de nome."""
        if self.input_mode:
            print("⏳ Já está digitando...")
            return
        
        print("\n📝 DIGITE O NOME (ENTER=confirmar, ESC=cancelar)")
        self.input_mode = True
        self.input_nome = ""
        self.cadastro_frame = frame.copy()
        self.cadastro_face_rect = face_rect
        self.cadastro_pending = True
    
    def recognize_face_simple(self, frame, face_rect):
        """
        Reconhece face usando face_recognition library com tolerância estrita.
        Alta precisão - apenas reconhece com alta confiança.
        """
        if not self.known_face_features:
            return None, 0.0
        
        try:
            # Extrai encoding atual
            current_encoding = self.extract_face_encoding_face_recognition(frame, face_rect, num_jitters=1)
            
            if current_encoding is None:
                return None, 0.0
            
            best_match = None
            best_distance = float('inf')
            
            # Tolerância estrita: 0.45-0.50 (padrão é 0.6, menor = mais estrito)
            strict_tolerance = 0.45
            
            for i, known_name in enumerate(self.known_face_names):
                try:
                    all_samples = [self.known_face_features[i]]
                    if known_name in self.multiple_samples:
                        all_samples.extend(self.multiple_samples[known_name])
                    
                    min_distance_for_person = float('inf')
                    
                    for known_features in all_samples:
                        # Verifica se é encoding face_recognition (128 dimensões)
                        if isinstance(known_features, np.ndarray) and len(known_features) == 128:
                            # Usa face_recognition.compare_faces com tolerância estrita
                            matches = face_recognition.compare_faces(
                                [known_features], 
                                current_encoding, 
                                tolerance=strict_tolerance
                            )
                            
                            if matches[0]:
                                # Calcula distância para confiança
                                distance = face_recognition.face_distance([known_features], current_encoding)[0]
                                if distance < min_distance_for_person:
                                    min_distance_for_person = distance
                        else:
                            # Fallback para método antigo (compatibilidade)
                            if len(known_features) == len(current_encoding):
                                known_features = known_features.astype(np.float32)
                                current_features = current_encoding.astype(np.float32)
                                
                                known_norm = np.linalg.norm(known_features)
                                current_norm = np.linalg.norm(current_features)
                                
                                if known_norm > 0 and current_norm > 0:
                                    known_features = known_features / known_norm
                                    current_features = current_features / current_norm
                                    
                                    similarity = np.dot(current_features, known_features)
                                    distance = 1.0 - similarity  # Converte similaridade para distância
                                    
                                    if distance < min_distance_for_person:
                                        min_distance_for_person = distance
                    
                    if min_distance_for_person < best_distance:
                        best_distance = min_distance_for_person
                        best_match = known_name
                        
                except Exception as e:
                    continue
            
            # Converte distância para confiança (menor distância = maior confiança)
            # face_recognition: distância 0.0 = match perfeito, 0.6 = limite padrão
            if best_match and best_distance < strict_tolerance:
                # Confiança: 1.0 - (distância / tolerância)
                confidence = max(0.0, min(1.0, 1.0 - (best_distance / strict_tolerance)))
                return best_match, confidence
            
            return None, 0.0
        except Exception as e:
            print(f"❌ Erro no reconhecimento: {e}")
            return None, 0.0
    
    def log_recognition(self, name, confidence):
        """Registra reconhecimento."""
        try:
            timestamp = datetime.now()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO recognitions (name, timestamp, confidence)
                VALUES (?, ?, ?)
            ''', (name, timestamp.isoformat(), confidence))
            self.conn.commit()
            
            self.recent_recognitions.insert(0, (name, confidence, timestamp))
            if len(self.recent_recognitions) > 5:
                self.recent_recognitions.pop()
            
            self.stats['total_reconhecimentos'] += 1
            print(f"📝 Registrado: {name}")
        except Exception as e:
            print(f"❌ Erro ao registrar: {e}")
    
    def get_statistics_by_person(self):
        """Retorna estatísticas por pessoa."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT name, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM recognitions
                GROUP BY name
                ORDER BY count DESC
            ''')
            return cursor.fetchall()
        except Exception as e:
            print(f"❌ Erro: {e}")
            return []
    
    def export_to_csv(self):
        """Exporta reconhecimentos para CSV."""
        try:
            os.makedirs("exports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"exports/reconhecimentos_{timestamp}.csv"
            
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT name, timestamp, confidence
                FROM recognitions
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                print("⚠️  Nenhum reconhecimento")
                return None
            
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['Nome', 'Data', 'Hora', 'Confiança'])
                
                for row in rows:
                    try:
                        name, timestamp_str, confidence = row
                        
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='ignore')
                        if isinstance(timestamp_str, bytes):
                            timestamp_str = timestamp_str.decode('utf-8', errors='ignore')
                        
                        if name is None:
                            name = 'Desconhecido'
                        else:
                            name = str(name)
                        
                        if timestamp_str:
                            try:
                                if isinstance(timestamp_str, str):
                                    dt = datetime.fromisoformat(timestamp_str)
                                else:
                                    dt = datetime.now()
                            except:
                                try:
                                    if isinstance(timestamp_str, str):
                                        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                    else:
                                        dt = datetime.now()
                                except:
                                    dt = datetime.now()
                        else:
                            dt = datetime.now()
                        
                        if confidence is None:
                            confidence = 0.0
                        elif isinstance(confidence, bytes):
                            try:
                                confidence = float(confidence.decode('utf-8', errors='ignore'))
                            except:
                                confidence = 0.0
                        elif isinstance(confidence, str):
                            try:
                                confidence = float(confidence)
                            except:
                                confidence = 0.0
                        
                        writer.writerow([
                            name,
                            dt.strftime('%Y-%m-%d'),
                            dt.strftime('%H:%M:%S'),
                            f'{confidence:.3f}'
                        ])
                    except Exception as e:
                        print(f"⚠️  Erro: {e}")
                        continue
            
            print(f"✅ Exportado: {csv_file}")
            print(f"📊 {len(rows)} registros")
            return csv_file
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")
            traceback.print_exc()
            return None
    
    def show_info(self):
        """Mostra informações do sistema."""
        print("\n" + "="*60)
        print("🎯 SISTEMA SCI-FI HUD")
        print("="*60)
        print(f"👥 Pessoas: {len(self.known_face_names)}")
        print(f"📸 Fotos: {self.stats['fotos_salvas']}")
        print(f"🔍 Reconhecimentos: {self.stats['total_reconhecimentos']}")
        print("="*60)
        print("🎮 CONTROLES:")
        print("  'c' = CADASTRAR")
        print("  'm' = ADICIONAR AMOSTRA")
        print("  ESPAÇO = RECONHECER")
        print("  'a' = AUTO ON/OFF")
        print("  'd' = DELETAR")
        print("  's' = ESTATÍSTICAS")
        print("  'e' = EXPORTAR CSV")
        print("  'h' = HISTÓRICO")
        print("  'q' = SAIR")
        print("="*60)
    
    def deletar_pessoa(self, nome):
        """Deleta pessoa cadastrada."""
        if nome not in self.known_face_names:
            print(f"❌ '{nome}' não encontrada")
            return False
        
        try:
            index = self.known_face_names.index(nome)
            self.known_face_names.pop(index)
            self.known_face_features.pop(index)
            
            if nome in self.multiple_samples:
                del self.multiple_samples[nome]
            
            self.save_face_encodings()
            print(f"✅ '{nome}' deletada")
            return True
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def show_statistics(self):
        """Mostra estatísticas."""
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS")
        print("="*60)
        print(f"👥 Pessoas: {len(self.known_face_names)}")
        print(f"📸 Fotos: {self.stats['fotos_salvas']}")
        print(f"🔍 Total: {self.stats['total_reconhecimentos']}")
        
        stats_by_person = self.get_statistics_by_person()
        if stats_by_person:
            print("\n📈 POR PESSOA:")
            print("-" * 60)
            for name, count, avg_conf in stats_by_person:
                print(f"  {name}: {count} vezes ({avg_conf:.2f})")
        
        if self.known_face_names:
            print("\n👤 CADASTRADAS:")
            for i, name in enumerate(self.known_face_names, 1):
                amostras = len(self.multiple_samples.get(name, [])) + 1
                print(f"  {i}. {name} ({amostras} amostra(s))")
        
        print("="*60)
    
    def run(self):
        """Loop principal."""
        if self.headless or self.cap is None:
            print("❌ Modo headless: use o servidor web (web_app) em vez de run()")
            return
        print("🎥 Iniciando câmera...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Erro ao capturar frame")
                break
            
            # Salva frame atual para uso em callbacks
            self.current_frame = frame.copy()
            
            # Processa frame
            self.process_frame(frame)
            
            # Desenha HUD
            self.draw_hud(frame)
            
            # Mostra frame
            cv2.imshow('Face Recognition - Sci-Fi HUD', frame)
            
            # Processa teclas
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keys(key):
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Limpa recursos."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if hasattr(self, 'conn'):
            self.conn.close()
        
        self.save_face_encodings()
        print("✅ Sistema finalizado!")

def main():
    """Função principal."""
    try:
        sistema = FaceRecognitionSystem()
        sistema.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
