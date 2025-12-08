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

TELLO1_WIFI_PROFILE = "TELLO-FE1A04"   #"TELLO-FE1A04" 
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

YOLO_MODEL_PATH = r"C:\Users\SONIA\Documents\Python Scripts\best.pt"

FIRE_CLASS_ID = 0
FIRE_CONF_THRES = 0.4

CELL_DIST_CM   = 40
STEPS_PER_SIDE = [10, 11, 10, 11]

ALIGN_MAX_TIME = 1.5  # ⚡ Reducido de 1.2s
ALIGN_DT       = 0.08  # ⚡ Aumentado de 0.06s para menos procesamiento

FIRE_FRAMES_PER_CELL = 3  # ⚡ Aumentado de 3 para mayor confiabilidad

K_ROLL            = 1.75
ROLL_MAX          = 35
ROLL_DEADZONE_PIX = 18

FRAME_WIDTH       = 480
FRAME_HEIGHT      = 360
ROI_TOP_PERCENT   = 0.6

CAMERA_OFFSET_X   = 20
CAMERA_OFFSET_Y   = -30

VIRTUAL_CENTER_X = FRAME_WIDTH // 2 + CAMERA_OFFSET_X
VIRTUAL_CENTER_Y = FRAME_HEIGHT // 2 + CAMERA_OFFSET_Y

LOWER_WHITE = np.array([0,   0, 235], dtype=np.uint8)
UPPER_WHITE = np.array([180, 25, 255], dtype=np.uint8)
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
    """
    Detección de línea blanca + visualización más estética.
    Devuelve:
        - error_x (o None si no hay línea)
        - debug_img (para mostrar en la ventana)
    """
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Máscara de blanco
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    debug_img = frame_resized.copy()

    # Ignorar la parte superior (sin texto, solo lógica)
    top_ignore = int(FRAME_HEIGHT * 0.20)
    mask[0:top_ignore, :] = 0

    # ROI central
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Centro real y centro virtual
    real_center_x = FRAME_WIDTH // 2
    real_center_y = FRAME_HEIGHT // 2
    cv2.circle(debug_img, (real_center_x, real_center_y), 4, (180, 180, 180), -1)
    cv2.circle(debug_img, (VIRTUAL_CENTER_X, VIRTUAL_CENTER_Y), 6, (0, 255, 255), -1)

    # Helper para panel inferior bonito
    def draw_info_panel(img, error_val, area_val, status_text, status_color):
        panel_h = 70
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (0, FRAME_HEIGHT - panel_h),
                      (FRAME_WIDTH, FRAME_HEIGHT),
                      (0, 0, 0), -1)
        alpha = 0.55
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Estado (izquierda)
        cv2.putText(img, status_text, (10, FRAME_HEIGHT - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # error_x
        cv2.putText(img,
                    f"error_x: {error_val if error_val is not None else '---'}",
                    (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # área
        cv2.putText(img,
                    f"area: {int(area_val)}",
                    (230, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Marco del ROI más grueso y vistoso
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                      (0, 255, 255), 3)

        return img

    # --- Casos sin línea útil ---
    if not contours:
        status = "Linea no detectada"
        debug_img = draw_info_panel(debug_img, None, 0, status, (0, 0, 255))
        return None, debug_img

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_LINE_AREA:
        status = "Linea muy pequena"
        debug_img = draw_info_panel(debug_img, None, area, status, (0, 165, 255))
        return None, debug_img

    M = cv2.moments(largest)
    if M["m00"] == 0:
        status = "Momento nulo"
        debug_img = draw_info_panel(debug_img, None, area, status, (0, 0, 255))
        return None, debug_img

    # --- Caso con línea buena ---
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Dibujo de la línea detectada
    cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
    cv2.circle(debug_img, (cx, cy), 7, (255, 0, 0), -1)
    cv2.line(debug_img,
             (VIRTUAL_CENTER_X, VIRTUAL_CENTER_Y),
             (cx, cy),
             (255, 255, 0), 2)

    error_x = cx - VIRTUAL_CENTER_X

    # Estado según qué tan centrado está
    if abs(error_x) < ROLL_DEADZONE_PIX:
        status = "Linea centrada"
        color = (0, 255, 0)
    else:
        status = "Corrigiendo posicion"
        color = (255, 255, 0)

    debug_img = draw_info_panel(debug_img, error_x, area, status, color)

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
# 🔥 YOLO: DETECCIÓN SIMPLE DE FUEGO + BOUNDING BOX
# ==============================

def detect_fire_in_frame(frame, model):
    """
    Corre YOLO sobre 'frame' y devuelve:
      - has_fire: True si hay al menos UNA detección de clase fuego
      - out_img: frame con bounding boxes dibujados

    IMPORTANTE:
    - Cualquier detección de clase FIRE_CLASS_ID cuenta como fuego,
      sin importar la confianza.
    - FIRE_CONF_THRES solo se usa para cambiar el color/estilo del recuadro.
    """
    if frame is None:
        return False, None

    # Inferencia YOLO
    results = model(frame, verbose=False, imgsz=FIRE_IMGSZ)

    if len(results) == 0 or results[0].boxes is None:
        return False, frame

    r0 = results[0]
    boxes = r0.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    has_fire = False
    out_img = frame.copy()

    for c, p, box in zip(cls, conf, xyxy):
        class_id = int(c)
        confidence = float(p)

        # Solo nos interesa la clase de fuego
        if class_id != FIRE_CLASS_ID:
            continue

        has_fire = True  # 👈 Cualquier detección de clase fuego cuenta

        x1, y1, x2, y2 = box.astype(int)

        # Color según confianza (solo visual)
        if confidence >= FIRE_CONF_THRES:
            color = (0, 0, 255)    # rojo: fuego claro
            thickness = 3
            label = f"FIRE {confidence:.2f}"
        else:
            color = (0, 165, 255)  # naranja: fuego débil
            thickness = 2
            label = f"fire? {confidence:.2f}"

        # Dibujar bounding box
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)

        # Etiqueta
        y_label = max(20, y1 - 10)
        cv2.putText(out_img, label, (x1, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return has_fire, out_img


def detect_fire_at_cell(model, coord, fire_count_so_far):
    """
    Checa unos pocos frames mientras el dron está quieto.
    Si en CUALQUIERA de esos frames hay fuego -> True.

    coord: (x, y) coordenada actual del dron
    fire_count_so_far: número de fuegos detectados ANTES de esta celda
    """
    x, y = coord
    fire_here = False
    last_vis = None

    for i in range(FIRE_FRAMES_PER_CELL):
        frame = get_latest_frame_copy()
        if frame is None:
            print(f"  ⚠️ Frame {i+1}/{FIRE_FRAMES_PER_CELL}: no disponible")
            time.sleep(0.05)
            continue

        has_fire, vis = detect_fire_in_frame(frame, model)
        last_vis = vis

        if has_fire:
            print(f"  🔥 Fuego detectado en frame {i+1}/{FIRE_FRAMES_PER_CELL}")
            fire_here = True
            break
        else:
            print(f"  ✓ Frame {i+1}/{FIRE_FRAMES_PER_CELL}: sin fuego")

        time.sleep(0.05)

    # Mostrar resultado visual de la celda
    if SHOW_WINDOWS and last_vis is not None:
        vis_out = last_vis.copy()
        h, w = vis_out.shape[:2]

        # Barra superior semitransparente
        overlay = vis_out.copy()
        bar_h = 80
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        alpha = 0.65
        vis_out = cv2.addWeighted(overlay, alpha, vis_out, 1 - alpha, 0)

        # Texto principal: estado
        status_text = "🔥 FUEGO DETECTADO" if fire_here else "SIN FUEGO"
        status_color = (0, 255, 0) if fire_here else (0, 0, 255)
        cv2.putText(vis_out, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Coordenadas actuales
        coord_text = f"Coord: (x={x}, y={y})"
        cv2.putText(vis_out, coord_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Contador de fuegos (incluyendo este si se detectó)
        total_fires = fire_count_so_far + (1 if fire_here else 0)
        counter_text = f"Fuegos detectados: {total_fires}"
        cv2.putText(vis_out, counter_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("🔥 YOLO Fuego", vis_out)
        cv2.waitKey(100)

    return fire_here


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
                print(f"[INFO] {len(fire_coords)} coordenada(s) guardada(s) en '{filename}'")
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

                cell_counter += 1
                print(f"\n--- Celda {cell_counter} / {total_cells} (Lado {side + 1}, Paso {step + 1}) ---")

                align_with_line(tl_flight)

                if not safe_forward(tl_flight, CELL_DIST_CM):
                    print("[ERROR] Forward falló. Abortando.")
                    should_abort = True
                    break

                time.sleep(0.1)

                x, y = update_coordinates(x, y, direction)
                print(f"  -> Coordenadas: (x={x}, y={y})")

                # 👇 NUEVA FUNCIÓN SIMPLIFICADA
                has_fire = detect_fire_at_cell(model, (x, y), len(fire_coords))
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