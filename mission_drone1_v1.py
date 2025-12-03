# -*- coding: utf-8 -*-
"""
DRON 1: VERSIÓN OPTIMIZADA - MENOS PAUSAS + YOLO MEJORADO
- Reducción inteligente de time.sleep() para mayor fluidez
- Detección de fuego YOLO más robusta y confiable
- Mejor logging y feedback visual
"""

import time
import threading
import subprocess
import socket
import cv2
import numpy as np
import robomaster
from robomaster import robot
from ultralytics import YOLO

# ==============================
# CONFIGURACIÓN WiFi (Windows)
# ==============================

TELLO1_WIFI_PROFILE = "TELLO-FE1A04"
WIFI_INTERFACE_NAME = "Wi-Fi"


def connect_to_wifi(profile_name: str,
                    interface_name: str = WIFI_INTERFACE_NAME,
                    timeout: int = 25):
    """Conecta a una red WiFi usando un perfil de Windows (netsh)."""
    print(f"\n[WiFi] Conectando al perfil/SSID '{profile_name}' en interfaz '{interface_name}'...")
    cmd = [
        "netsh", "wlan", "connect",
        f"name={profile_name}",
        f"ssid={profile_name}",
        f"interface={interface_name}"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[WiFi] ERROR al intentar conectar:")
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("No se pudo lanzar el comando de conexión WiFi.")

    start = time.time()
    while time.time() - start < timeout:
        status = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True
        )
        if status.returncode == 0 and profile_name in status.stdout:
            print(f"[WiFi] Conectado correctamente a '{profile_name}'.")
            return
        time.sleep(1.0)

    print("[WiFi] Tiempo de espera agotado. Verifica la conexión manualmente.")


# ==============================
# CONFIG IP LOCAL DEL SDK
# ==============================

def configure_local_ip():
    """Detecta la IP local de la laptop en la red del dron."""
    try:
        DRONE_IP = "192.168.10.1"
        DRONE_PORT = 8889

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((DRONE_IP, DRONE_PORT))
        local_ip = s.getsockname()[0]
        s.close()

        robomaster.config.LOCAL_IP_STR = local_ip
        print(f"[IP] LOCAL_IP_STR configurada a {local_ip}")
    except Exception as e:
        print("[IP] No se pudo detectar IP local, usando 0.0.0.0:", e)
        robomaster.config.LOCAL_IP_STR = "0.0.0.0"


# ==============================
# HELPERS OPTIMIZADOS
# ==============================

def safe_takeoff(tl_flight, timeout=12):
    """
    Despegue seguro con mejor timing.
    """
    print("[SAFE] Despegando...")
    try:
        action = tl_flight.takeoff()
        
        # Dar tiempo suficiente para que el dron alcance altura de despegue
        print("[SAFE] Esperando estabilización del despegue...")
        time.sleep(5.0)  # Regresamos a 5s para mayor seguridad
        
        print("[SAFE] ✅ Takeoff completado.")
        return True
        
    except Exception as e:
        print(f"[SAFE][ERROR] Error en takeoff: {e}")
        return False

def safe_forward(tl_flight, dist_cm, timeout=25):
    print(f"[SAFE] Avanzando {dist_cm} cm...")
    
    # ⚡ OPTIMIZACIÓN: Solo 1 comando RC es suficiente
    tl_flight.rc(a=0, b=0, c=0, d=0)
    time.sleep(0.15)  # ⚡ Reducido de 0.8s - mucho más rápido
    
    try:
        action = tl_flight.forward(distance=dist_cm)
        
        # Tiempo estimado más preciso
        estimated_time = (dist_cm / 12.0) + 1.5  # ⚡ Velocidad ajustada + menos margen
        
        time.sleep(estimated_time)
        
        if action.is_completed:
            print("[SAFE] ✅ Forward completado.")
        else:
            print("[SAFE] ⚠️ Forward sin confirmación SDK.")
        
        return True
        
    except Exception as e:
        print(f"[SAFE][ERROR] Excepción durante forward: {e}")
        return False

def safe_land(tl_flight, timeout=15):
    print("[SAFE] Aterrizando...")
    try:
        action = tl_flight.land()
        time.sleep(4.0)  # ⚡ Reducido de 5.0s
        print("[SAFE] ✅ Land completado.")
        return True
    except Exception as e:
        print(f"[SAFE][ERROR] Error en land: {e}")
        return False

def safe_down(tl_flight, dist_cm, timeout=12):
    """
    Baja el dron de forma segura con mejor manejo de errores.
    """
    print(f"[SAFE] Bajando {dist_cm} cm...")
    
    # Detener cualquier movimiento previo
    tl_flight.rc(a=0, b=0, c=0, d=0)
    time.sleep(0.2)
    
    try:
        action = tl_flight.down(distance=dist_cm)
        
        # Tiempo estimado basado en velocidad típica del dron
        estimated_time = (dist_cm / 10.0) + 2.0  # ~10 cm/s + margen de seguridad
        
        print(f"[SAFE] Esperando {estimated_time:.1f}s para descenso...")
        time.sleep(estimated_time)
        
        # Verificar si se completó
        if hasattr(action, 'is_completed') and action.is_completed:
            print("[SAFE] ✅ Down completado exitosamente.")
            return True
        else:
            print("[SAFE] ⚠️ Down sin confirmación clara del SDK.")
            return True  # Continuar de todas formas
            
    except Exception as e:
        print(f"[SAFE][ERROR] Error durante descenso: {e}")
        return False
    
def initialize_drone_position(tl_flight):
    """
    Secuencia de inicialización después del despegue.
    Incluye verificación y descenso controlado.
    """
    print("\n[INIT] Ajustando altura inicial...")
    
    # Pausa para estabilización después del takeoff
    print("[INIT] Estabilizando después del despegue...")
    time.sleep(2.0)
    
    # Primer intento de descenso
    descenso_exitoso = safe_down(tl_flight, 20)
    
    if not descenso_exitoso:
        print("[INIT] ⚠️ Primer intento de descenso falló, reintentando...")
        time.sleep(1.0)
        
        # Segundo intento con comando RC directo
        print("[INIT] Intentando descenso con comandos RC...")
        try:
            # Descenso suave con RC (5 segundos a velocidad baja)
            for _ in range(10):
                tl_flight.rc(a=0, b=0, c=-15, d=0)  # c negativo = bajar
                time.sleep(0.5)
            
            # Detener
            tl_flight.rc(a=0, b=0, c=0, d=0)
            time.sleep(0.5)
            print("[INIT] ✅ Descenso manual completado.")
            
        except Exception as e:
            print(f"[INIT][ERROR] Error en descenso manual: {e}")
    
    else:
        print("[INIT] ✅ Descenso inicial completado correctamente.")
    
    # Estabilización final
    print("[INIT] Estabilizando en nueva altura...")
    tl_flight.rc(a=0, b=0, c=0, d=0)
    time.sleep(1.0)
    
    return True

def safe_rotate(tl_flight, angle_deg, timeout=12):
    print(f"[SAFE] Rotando {angle_deg} grados...")
    
    tl_flight.rc(a=0, b=0, c=0, d=0)
    time.sleep(0.15)  # ⚡ Reducido de 0.3s
    
    try:
        action = tl_flight.rotate(angle_deg)
        estimated_time = (abs(angle_deg) / 40.0) + 1.5  # ⚡ Velocidad ajustada
        time.sleep(estimated_time)
        print("[SAFE] ✅ Rotate completado.")
        return True
    except Exception as e:
        print(f"[SAFE][ERROR] Error en rotate: {e}")
        return False


# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

YOLO_MODEL_PATH = r"C:\Users\Kim\Tec\Documents\Carrera_IRS\7th Semester\FinalChallenge\fire_detector_model_v2\best.pt"

FIRE_CLASS_ID = 0
FIRE_CONF_THRES = 0.25  # ⚡ Reducido de 0.5 para detectar más fuegos

CELL_DIST_CM   = 40
STEPS_PER_SIDE = [10, 11, 10, 11]

ALIGN_MAX_TIME = 1.0  # ⚡ Reducido de 1.2s
ALIGN_DT       = 0.08  # ⚡ Aumentado de 0.06s para menos procesamiento

# 🔥 YOLO MEJORADO: Más frames + mejor verificación
FIRE_FRAMES_PER_CELL = 5  # ⚡ Aumentado de 3 para mayor confiabilidad
FIRE_DETECTION_THRESHOLD = 1  # Número mínimo de frames con fuego para confirmar

K_ROLL            = 2
ROLL_MAX          = 35
ROLL_DEADZONE_PIX = 18

FRAME_WIDTH       = 480
FRAME_HEIGHT      = 360
ROI_TOP_PERCENT   = 0.6

CAMERA_OFFSET_X   = 20
CAMERA_OFFSET_Y   = -30

VIRTUAL_CENTER_X = FRAME_WIDTH // 2 + CAMERA_OFFSET_X
VIRTUAL_CENTER_Y = FRAME_HEIGHT // 2 + CAMERA_OFFSET_Y

LOWER_WHITE = np.array([0, 0, 200], dtype=np.uint8)
UPPER_WHITE = np.array([180, 40, 255], dtype=np.uint8)
MIN_LINE_AREA = 400

SHOW_WINDOWS = True
APPLY_MIRROR = True

FIRE_IMGSZ = 320  

TURN_YAW_CMD = -90
TURN_TIME    = 3

# ==============================
# VARIABLES GLOBALES DE VIDEO
# ==============================

latest_frame = None
frame_lock   = threading.Lock()
video_running = False


def transform_frame(frame):
    """Corrige espejo si es necesario."""
    if APPLY_MIRROR:
        frame = cv2.flip(frame, -1)
    return frame


def video_loop(tl_camera):
    """Hilo dedicado para captura de video."""
    global latest_frame, video_running

    print("[VIDEO] Hilo de video iniciado, configurando camara...")
    try:
        tl_camera.start_video_stream(display=False)
        tl_camera.set_fps("low")
        tl_camera.set_resolution("low")
        tl_camera.set_bitrate(6)
        time.sleep(1.0)
    except Exception as e:
        print("[VIDEO] Error al iniciar video_stream:", e)
        video_running = False
        return

    error_count = 0

    while video_running:
        try:
            frame = tl_camera.read_cv2_image(timeout=2, strategy="newest")
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print("[VIDEO] Error leyendo frame:", e)
            time.sleep(0.03)
            continue

        if frame is None:
            time.sleep(0.01)
            continue

        frame = transform_frame(frame)

        with frame_lock:
            latest_frame = frame

        time.sleep(0.008)

    try:
        tl_camera.stop_video_stream()
    except Exception:
        pass

    print("[VIDEO] Hilo de video terminado.")


def get_latest_frame_copy():
    """Devuelve una copia del último frame disponible (o None)."""
    with frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


# ==============================
# VISIÓN: LÍNEA BLANCA
# ==============================

def compute_line_error(frame):
    """Detección de línea blanca."""
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    debug_img = frame_resized.copy()

    top_ignore = int(FRAME_HEIGHT * 0.20)
    mask[0:top_ignore, :] = 0
    cv2.rectangle(debug_img, (0, 0), (FRAME_WIDTH, top_ignore), (0, 0, 255), 1)
    cv2.putText(debug_img, "Ignorado 20% superior", (10, top_ignore - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    x_left   = int(FRAME_WIDTH * 0.15)
    x_right  = int(FRAME_WIDTH * 0.85)
    y_top    = top_ignore
    y_bottom = int(FRAME_HEIGHT / 1.25)

    if y_top >= y_bottom:
        y_top    = int(FRAME_HEIGHT * 0.2)
        y_bottom = int(FRAME_HEIGHT * 0.6)

    roi_mask = np.zeros_like(mask)
    roi_mask[y_top:y_bottom, x_left:x_right] = mask[y_top:y_bottom, x_left:x_right]
    mask = roi_mask

    cv2.rectangle(debug_img, (x_left, y_top), (x_right, y_bottom), (100, 100, 255), 2)
    cv2.putText(debug_img, "ROI central hacia arriba", (x_left, y_top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    real_center_x = FRAME_WIDTH // 2
    real_center_y = FRAME_HEIGHT // 2
    cv2.circle(debug_img, (real_center_x, real_center_y), 5, (128, 128, 128), -1)
    cv2.circle(debug_img, (VIRTUAL_CENTER_X, VIRTUAL_CENTER_Y), 7, (0, 255, 255), -1)

    if not contours:
        cv2.putText(debug_img, "Linea NO detectada", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug_img

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_LINE_AREA:
        cv2.putText(debug_img, "Linea muy pequena", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return None, debug_img

    M = cv2.moments(largest)
    if M["m00"] == 0:
        cv2.putText(debug_img, "Momento nulo", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return None, debug_img

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
    cv2.circle(debug_img, (cx, cy), 8, (255, 0, 0), -1)
    cv2.line(debug_img,
             (VIRTUAL_CENTER_X, VIRTUAL_CENTER_Y),
             (cx, cy),
             (255, 255, 0), 2)

    error_x = cx - VIRTUAL_CENTER_X

    cv2.putText(debug_img, f"error_x: {error_x}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(debug_img, f"area: {int(area)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return error_x, debug_img


def align_with_line(tl_flight):
    """Alinea el dron con la línea blanca ajustando SOLO ROLL."""
    print("  -> Alineando con linea (hover, roll)...")
    start = time.time()
    stable_frames = 0
    no_line_frames = 0

    while time.time() - start < ALIGN_MAX_TIME:
        frame = get_latest_frame_copy()
        if frame is None:
            time.sleep(0.03)
            continue

        error, debug_img = compute_line_error(frame)

        roll_cmd = 0
        if error is None:
            stable_frames = 0
            no_line_frames += 1
            roll_cmd = 0

            if no_line_frames > 10:
                print("  -> No se ve línea, continuando.")
                break
        else:
            no_line_frames = 0
            if abs(error) < ROLL_DEADZONE_PIX:
                roll_cmd = 0
                stable_frames += 1
                if stable_frames >= 4:  # ⚡ Reducido de 5 frames
                    print("  -> Linea centrada.")
                    break
            else:
                stable_frames = 0
                norm_err = error / (FRAME_WIDTH / 2.0)
                roll_cmd = K_ROLL * norm_err * 100.0
                roll_cmd = max(-ROLL_MAX, min(ROLL_MAX, roll_cmd))

        tl_flight.rc(a=int(-roll_cmd), b=0, c=0, d=0)

        if SHOW_WINDOWS:
            cv2.imshow("Alineacion linea (roll)", debug_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(ALIGN_DT)

    tl_flight.rc(a=0, b=0, c=0, d=0)
    time.sleep(0.05)


# ==============================
# 🔥 YOLO MEJORADO: DETECCIÓN DE FUEGO
# ==============================
def detect_fire_in_frame(frame, model):
    """
    🔥 Detección de fuego con bounding boxes y análisis detallado.
    Retorna: (fuego_detectado, confianza_máxima, frame_anotado)
    """
    # Frame para mostrar (copia del original)
    display_frame = frame.copy()
    
    # Inferencia YOLO
    results = model(frame, verbose=False, imgsz=320)
    
    if len(results) == 0:
        cv2.putText(display_frame, "No YOLO results", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, 0.0, display_frame
    
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        cv2.putText(display_frame, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return False, 0.0, display_frame
    
    boxes = r0.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    
    max_confidence = 0.0
    fire_detected = False
    fire_boxes = []
    
    # Procesar todas las detecciones
    for i, (c, p, box) in enumerate(zip(cls, conf, xyxy)):
        class_id = int(c)
        confidence = float(p)
        
        if class_id != FIRE_CLASS_ID:
            continue
        
        # Coordenadas del bounding box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        max_confidence = max(max_confidence, confidence)
        
        # 👇 NUEVA LÓGICA: Detectar con umbral más bajo
        if confidence >= FIRE_CONF_THRES:
            fire_detected = True
            fire_boxes.append({
                'box': (x1, y1, x2, y2),
                'conf': confidence,
                'confirmed': True
            })
            color = (0, 0, 255)  # ROJO - fuego confirmado
            label = f"FIRE {confidence:.1%}"
            thickness = 3
        else:
            # Mostrar detecciones débiles también
            fire_boxes.append({
                'box': (x1, y1, x2, y2),
                'conf': confidence,
                'confirmed': False
            })
            color = (0, 165, 255)  # NARANJA - posible fuego
            label = f"fire? {confidence:.1%}"
            thickness = 2
        
        # Dibujar bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Etiqueta con fondo
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(display_frame, 
                     (x1, y1_label - label_size[1] - 10),
                     (x1 + label_size[0], y1_label),
                     color, -1)
        cv2.putText(display_frame, label, (x1, y1_label - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return fire_detected, max_confidence, display_frame


def detect_fire_at_cell(model, cell_number, total_cells):
    """
    🔥 Detección en celda con visualización mejorada y contador.
    Ahora SOLO necesita 1 detección positiva para confirmar.
    """
    print("  🔍 Iniciando análisis de fuego...")
    
    fire_detections = 0
    confidence_values = []
    frames_analyzed = 0
    all_max_confidences = []
    
    # Ventana para mostrar detecciones
    window_name = "🔥 Fire Detection - Live Feed"
    
    for i in range(FIRE_FRAMES_PER_CELL):
        frame = get_latest_frame_copy()
        if frame is None:
            print(f"  ⚠️ Frame {i+1}/{FIRE_FRAMES_PER_CELL}: No disponible")
            time.sleep(0.05)
            continue
        
        # Detectar fuego con visualización
        detected, confidence, display_frame = detect_fire_in_frame(frame, model)
        frames_analyzed += 1
        all_max_confidences.append(confidence)
        
        if detected:
            fire_detections += 1
            confidence_values.append(confidence)
            print(f"  🔥 Frame {i+1}/{FIRE_FRAMES_PER_CELL}: FUEGO (conf: {confidence:.1%})")
        else:
            print(f"  ✓ Frame {i+1}/{FIRE_FRAMES_PER_CELL}: Sin fuego (max: {confidence:.1%})")
        
        # 👇 AGREGAR OVERLAY CON INFORMACIÓN
        h, w = display_frame.shape[:2]
        
        # Panel superior con información
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
        
        # Celda actual
        cv2.putText(display_frame, f"Celda: {cell_number}/{total_cells}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Frame actual
        cv2.putText(display_frame, f"Frame: {i+1}/{FIRE_FRAMES_PER_CELL}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Detecciones acumuladas
        if fire_detections > 0:
            status_color = (0, 255, 0)
            status_text = f"Fuegos detectados: {fire_detections}"
        else:
            status_color = (100, 100, 100)
            status_text = "Sin detecciones"
        cv2.putText(display_frame, status_text, 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Mostrar frame
        if SHOW_WINDOWS:
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(100)  # 100ms por frame
        
        time.sleep(0.03)
    
    # 👇 NUEVA LÓGICA: Solo necesita 1 detección para confirmar
    fire_confirmed = fire_detections >= 1  # Era >= FIRE_DETECTION_THRESHOLD
    
    print(f"\n  {'='*55}")
    if fire_confirmed:
        avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        max_conf = max(confidence_values) if confidence_values else 0
        print(f"  🔥 *** FUEGO CONFIRMADO ***")
        print(f"     Detecciones: {fire_detections}/{frames_analyzed}")
        print(f"     Confianza promedio: {avg_conf:.1%}")
        print(f"     Confianza máxima: {max_conf:.1%}")
    else:
        print(f"  ✅ Sin fuego confirmado")
        print(f"     Detecciones: {fire_detections}/{frames_analyzed}")
        if all_max_confidences:
            print(f"     Máxima confianza vista: {max(all_max_confidences):.1%}")
        print(f"     Umbral: {FIRE_CONF_THRES:.1%}")
    print(f"  {'='*55}\n")
    
    return fire_confirmed
# ==============================
# NAVEGACIÓN CUADRADA
# ==============================

def update_coordinates(x, y, direction):
    """Actualiza coordenadas según dirección."""
    if direction == 0:
        x += 1
    elif direction == 1:
        y += 1
    elif direction == 2:
        x -= 1
    elif direction == 3:
        y -= 1
    return x, y


def rotate_left_90(tl_flight):
    """Gira ~90° a la izquierda."""
    print("  -> Girando 90° a la izquierda...")
    safe_rotate(tl_flight, -90)
    time.sleep(0.3)  # ⚡ Reducido de 0.4s


def save_fire_coordinates(fire_coords):
    """Guarda las coordenadas de fuego en un archivo."""
    filename = "fire_coordinates.txt"
    
    try:
        with open(filename, 'w') as f:
            if not fire_coords:
                f.write("# No se detectaron fuegos\n")
                print(f"[INFO] Archivo '{filename}' creado (sin fuegos detectados).")
            else:
                f.write(f"# Fuegos detectados: {len(fire_coords)}\n")
                for (x, y) in fire_coords:
                    f.write(f"{x},{y}\n")
                print(f"[INFO] ✅ {len(fire_coords)} coordenada(s) guardada(s) en '{filename}'")
    except Exception as e:
        print(f"[ERROR] No se pudieron guardar las coordenadas: {e}")


# ==============================
# MAIN DRON 1 OPTIMIZADO
# ==============================

def drone1_main():
    """Main del dron 1: recorrido rectangular + línea + fuego."""
    global video_running

    configure_local_ip()

    print("[INFO] Inicializando dron 1...")
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_flight = tl_drone.flight
    tl_camera = tl_drone.camera

    fire_coords = []

    try:
        try:
            bat = tl_drone.get_battery()
            print(f"[INFO] Bateria: {bat}%")
        except Exception:
            pass

        print(f"[INFO] Cargando modelo YOLO: {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = model(dummy, verbose=False, imgsz=FIRE_IMGSZ)

        print("[INFO] Iniciando hilo de video...")
        video_running = True
        v_thread = threading.Thread(target=video_loop, args=(tl_camera,), daemon=True)
        v_thread.start()

        print("[INFO] Esperando primeros frames de camara...")
        t0 = time.time()
        while get_latest_frame_copy() is None and time.time() - t0 < 5.0:
            time.sleep(0.05)

        if get_latest_frame_copy() is None:
            print("[WARN] No se recibieron frames de la camara.")

        if not safe_takeoff(tl_flight):
            print("[ERROR] Falló el takeoff.")
            return fire_coords

        # Usar la nueva función de inicialización
        if not initialize_drone_position(tl_flight):
            print("[ERROR] Falló la inicialización de posición.")
            safe_land(tl_flight)
            return fire_coords

        x, y = 0, 0
        direction = 0

        print("[INFO] Iniciando recorrido del RECTÁNGULO...")
        print(f"[INFO] Pasos por lado: {STEPS_PER_SIDE}")

        should_abort = False
        
        # 👇 NUEVO: Contador global de celdas
        total_cells = sum(STEPS_PER_SIDE)
        cell_counter = 0

        for side in range(4):
            if should_abort:
                break
            
            steps_this_side = STEPS_PER_SIDE[side]
            
            print(f"\n{'='*60}")
            print(f"===== LADO {side + 1} / 4 ({steps_this_side} pasos) =====")
            print(f"{'='*60}")
            
            for step in range(steps_this_side):
                if should_abort:
                    break
                
                cell_counter += 1  # 👈 Incrementar contador
                    
                print(f"\n--- Celda {cell_counter} / {total_cells} (Lado {side + 1}, Paso {step + 1}) ---")

                align_with_line(tl_flight)

                if not safe_forward(tl_flight, CELL_DIST_CM):
                    print("[ERROR] Forward falló. Abortando.")
                    should_abort = True
                    break

                time.sleep(0.1)

                x, y = update_coordinates(x, y, direction)
                print(f"  -> Coordenadas: (x={x}, y={y})")

                # 👇 PASAR CONTADOR A LA FUNCIÓN
                has_fire = detect_fire_at_cell(model, cell_counter, total_cells)
                if has_fire:
                    print(f"  *** 🔥 FUEGO CONFIRMADO EN (x={x}, y={y}) ***")
                    fire_coords.append((x, y))

            if not should_abort and side < 3:
                direction = (direction + 1) % 4
                rotate_left_90(tl_flight)

        for side in range(4):
            if should_abort:
                break
            
            steps_this_side = STEPS_PER_SIDE[side]
            
            print(f"\n{'='*60}")
            print(f"===== LADO {side + 1} / 4 ({steps_this_side} pasos) =====")
            print(f"{'='*60}")
            
            for step in range(steps_this_side):
                if should_abort:
                    break
                    
                print(f"\n--- Celda {step + 1} / {steps_this_side} en lado {side + 1} ---")

                align_with_line(tl_flight)
                
                # ⚡ ELIMINADO el delay de 0.1s - innecesario después de align

                if not safe_forward(tl_flight, CELL_DIST_CM):
                    print("[ERROR] Forward falló. Abortando.")
                    should_abort = True
                    break

                # ⚡ Reducido de 0.2s - movimiento continuo más fluido
                time.sleep(0.1)

                x, y = update_coordinates(x, y, direction)
                print(f"  -> Coordenadas: (x={x}, y={y})")

                has_fire = detect_fire_at_cell(model)
                if has_fire:
                    print(f"  *** 🔥 FUEGO CONFIRMADO EN (x={x}, y={y}) ***")
                    fire_coords.append((x, y))

            if not should_abort and side < 3:
                direction = (direction + 1) % 4
                rotate_left_90(tl_flight)

        if not should_abort:
            print("\n[INFO] ✅ Recorrido completo terminado!")
        else:
            print("\n[WARN] ⚠️ Recorrido abortado.")

        print("\n[INFO] Aterrizando...")
        safe_land(tl_flight)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción manual (Ctrl+C).")
        try:
            safe_land(tl_flight)
        except Exception:
            pass

    except Exception as e:
        print("\n[ERROR] Excepcion en drone1_main:")
        print(e)
        import traceback
        traceback.print_exc()
        try:
            safe_land(tl_flight)
        except Exception:
            pass

    finally:
        video_running = False
        time.sleep(0.2)  # ⚡ Reducido de 0.3s

        try:
            tl_flight.rc(a=0, b=0, c=0, d=0)
        except Exception:
            pass

        try:
            tl_camera.stop_video_stream()
        except Exception:
            pass

        try:
            tl_drone.close()
        except Exception:
            pass

        if SHOW_WINDOWS:
            cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(" 🔥 RESUMEN DE FUEGOS DETECTADOS 🔥")
        print("="*60)
        if not fire_coords:
            print("❌ No se detectaron fuegos en ninguna celda.")
        else:
            print(f"✅ Total de fuegos detectados: {len(fire_coords)}")
            for i, (fx, fy) in enumerate(fire_coords, 1):
                print(f"   {i}. 🔥 Fuego en coordenada (x={fx}, y={fy})")
        print("="*60)
        print(f"📊 Total de celdas inspeccionadas: {sum(STEPS_PER_SIDE)}")
        print("="*60 + "\n")

        save_fire_coordinates(fire_coords)

    return fire_coords


# ==============================
# MAIN
# ==============================

def main():
    print("="*60)
    print("  🚁 SISTEMA DRON 1: OPTIMIZADO + YOLO MEJORADO")
    print("="*60)
    print("\n⚡ MEJORAS:")
    print(" • Reducción de pausas innecesarias")
    print(" • Detección YOLO con sistema de votación")
    print(" • Mayor confiabilidad en detección de fuego")
    print(" • Estadísticas detalladas por celda")
    print("="*60 + "\n")

    connect_to_wifi(TELLO1_WIFI_PROFILE)
    time.sleep(2.5)  # ⚡ Reducido de 3.0s

    fire_coords = drone1_main()
    print(f"\n[INFO] 🎯 Coordenadas finales: {fire_coords}")
    print("\n[INFO] ✅ Misión del DRON 1 terminada.")


if __name__ == "__main__":
    main()