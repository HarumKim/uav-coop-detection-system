# mission_drone2_v1.py
# -*- coding: utf-8 -*-
"""
DRON 2: VISITA COORDENADAS DE FUEGO SIEMPRE AVANZANDO (FORWARD)

Convenciones del sistema de coordenadas (MISMO que DRON 1):

- Unidad de coordenadas: "celdas" (mismo grid logico que DRON 1).
- DRON 1 usa ~40 cm/celda en el mapa; DRON 2 avanza ~38 cm/celda (calibrado).

- Ejes en el plano:
    +X  -> hacia adelante desde el punto de despegue.
    +Y  -> hacia la IZQUIERDA respecto al eje +X.

Requisitos de movimiento:
- El dron SOLO se desplaza físicamente usando:
    - forward (adelante)
    - rotate (giro en yaw)
- NUNCA left / right / back.

- Para ir de (x1, y1) a (x2, y2):
    1) Se calcula el vector Δ = (dx, dy) en el plano.
    2) Se calcula el ángulo objetivo usando atan2(dy, dx).
    3) Se calcula el ángulo a girar en yaw (en el dron) para alinear el eje +X
       con el vector objetivo.
    4) Se avanza la distancia euclidiana en línea recta (hipotenusa).

Además:
- Incluye alineación visual sobre fuego (YOLO) antes de descender.
- Incluye lógica para compactar fuegos duplicados y ajuste de esquinas del grid.

"""

import math
import time
import socket
import subprocess

import cv2
import numpy as np
import robomaster
from robomaster import robot
from ultralytics import YOLO

# ============================
# CONFIGURACIÓN GENERAL
# ============================

CELL_DIST_MAP_CM = 40        # DRON 1 mapea cada celda a ~40 cm (solo referencia lógica)
CELL_DIST_DRONE2_CM = 38     # DRON 2 avanza ~38 cm por celda (calibrado)
HOVER_TIME_OVER_FIRE = 1.0  # tiempo en hover en la cota baja (s)

# Tamaño del grid lógico (en celdas, debe coincidir con DRON 1)
# Ejemplo: coordenadas X en [0, GRID_MAX_X], Y en [0, GRID_MAX_Y].
GRID_MAX_X = 10   # última columna (X máxima) del grid
GRID_MAX_Y = 11   # última fila (Y máxima) del grid


# Calibración extra cuando hay giros grandes (> 45 grados)
EXTRA_YAW_CALIB_DEG = 9.0    # grados extra al giro
EXTRA_DIST_CALIB_CM = 10.0   # centímetros extra al desplazamiento

# WiFi opcional para modo standalone (Windows)
TELLO2_WIFI_PROFILE = "TELLO-FE1B95"
WIFI_INTERFACE_NAME = "Wi-Fi"

# === YOLO / VISIÓN PARA DRON 2 ===
YOLO_MODEL_PATH = r"C:\Users\SONIA\Documents\Python Scripts\best.pt"
FIRE_CLASS_ID = 0
FIRE_CONF_THRES = 0.5

APPLY_MIRROR = False         # si la cámara está invertida físicamente
SHOW_WINDOWS_VISUAL = True   # mostrar ventana de OpenCV para debug

# Control visual
VISUAL_ALIGN_MAX_TIME = 5.0      # s máximo para intentar alinear
VISUAL_ALIGN_DT = 0.12           # periodo de control (s)
FIRE_ALIGN_TOL_X = 25            # tolerancia en pixeles eje X
FIRE_ALIGN_TOL_Y = 25            # tolerancia en pixeles eje Y
FIRE_ALIGN_STABLE_FRAMES = 4     # (lo dejamos aunque usemos tiempo de hold)
RC_VISUAL_MAX = 14               # velocidad máxima en RC
VISUAL_KP = 0.7                  # ganancia proporcional para control visual
FIRE_ALIGN_HOLD_TIME = 1.5       # segundos que debe permanecer centrado

# ============================
# UTILIDADES DE ÁNGULOS
# ============================

def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / math.pi


def wrap_angle_deg(angle):
    """Envuelve el ángulo a [-180, 180) grados."""
    while angle >= 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

# ============================
# CONEXIÓN WIFI (OPCIONAL)
# ============================

def connect_to_wifi(profile_name: str,
                    interface_name: str = WIFI_INTERFACE_NAME,
                    timeout: int = 25):
    """
    Conecta a una red WiFi usando un perfil de Windows (netsh).
    """
    print(f"\n[WiFi] Conectando al perfil/SSID '{profile_name}' en interfaz '{interface_name}'...")
    cmd = ["netsh", "wlan", "connect", f"name={profile_name}", f"interface={interface_name}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("[WiFi][ERROR] netsh fallo al intentar conectar:")
        print(e.stdout)
        print(e.stderr)
        return

    for _ in range(timeout):
        time.sleep(1.0)
        check_cmd = ["netsh", "wlan", "show", "interfaces"]
        try:
            result = subprocess.run(check_cmd, check=True, capture_output=True, text=True)
            output = result.stdout
            if profile_name in output and "connected" in output.lower():
                print(f"[WiFi] Conectado correctamente a '{profile_name}'.")
                return
        except subprocess.CalledProcessError:
            pass

    print("[WiFi][WARN] Tiempo de espera agotado. Verifica la conexión manualmente.")


def configure_local_ip():
    try:
        DRONE_IP = "192.168.10.1"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((DRONE_IP, 8889))
        local_ip = s.getsockname()[0]
        s.close()

        robomaster.config.LOCAL_IP_STR = local_ip
        print(f"[IP][DRON 2] LOCAL_IP_STR configurada a {local_ip}")
    except Exception as e:
        print(f"[IP][DRON 2][WARN] No fue posible determinar la IP local automáticamente: {e}")
        print("    Se usara la configuración por defecto de robomaster.")


# ============================
# UTILIDADES DE MOVIMIENTO
# ============================

def safe_takeoff(flight):
    try:
        print("[DRON 2] Despegando...")
        flight.takeoff().wait_for_completed(timeout=10)
        time.sleep(2.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en takeoff:", e)
        return False


def safe_land(flight):
    try:
        print("[DRON 2] Aterrizando...")
        flight.land().wait_for_completed(timeout=10)
        time.sleep(2.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en land:", e)
        return False


def safe_forward_cm(flight, dist_cm, speed=50):
    """
    Avanza dist_cm usando go_forward. Devuelve True si completó sin excepción.
    """
    try:
        print(f"[DRON 2] Avanzando {dist_cm:.1f} cm (speed={speed})...")
        flight.go_forward(x=dist_cm, y=0, z=0, speed=speed).wait_for_completed(timeout=10)
        time.sleep(1.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en go_forward:", e)
        return False


def safe_down_cm(flight, dist_cm, speed=40):
    """
    Desciende dist_cm centímetros (positive value).
    """
    try:
        print(f"[DRON 2] Descendiendo {dist_cm:.1f} cm...")
        flight.go_down(x=dist_cm, speed=speed).wait_for_completed(timeout=10)
        time.sleep(1.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en go_down:", e)
        return False


def safe_up_cm(flight, dist_cm, speed=40):
    """
    Sube dist_cm centímetros (positive value).
    """
    try:
        print(f"[DRON 2] Subiendo {dist_cm:.1f} cm...")
        flight.go_up(x=dist_cm, speed=speed).wait_for_completed(timeout=10)
        time.sleep(1.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en go_up:", e)
        return False


def safe_rotate_deg(flight, yaw_deg):
    """
    Gira yaw_deg grados en yaw. Positivo: izquierda; negativo: derecha.
    """
    try:
        sign = 1 if yaw_deg >= 0 else -1
        yaw_abs = abs(int(round(yaw_deg)))
        if yaw_abs == 0:
            return True

        if sign > 0:
            print(f"[DRON 2] Rotando +{yaw_abs} grados (ccw)...")
            flight.rotate_ccw(yaw_abs).wait_for_completed(timeout=10)
        else:
            print(f"[DRON 2] Rotando -{yaw_abs} grados (cw)...")
            flight.rotate_cw(yaw_abs).wait_for_completed(timeout=10)

        time.sleep(1.0)
        return True
    except Exception as e:
        print("[DRON 2][ERROR] Problema en rotate:", e)
        return False


# ============================
# LECTURA DE COORDENADAS
# ============================

def load_fire_coordinates_from_txt(path="fire_coordinates.txt"):
    """
    Lee las coordenadas de fuego desde un archivo de texto con formato:
        x,y
    por línea.
    """
    fire_coords = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    continue
                try:
                    x = int(parts[0].strip())
                    y = int(parts[1].strip())
                    fire_coords.append((x, y))
                except ValueError:
                    print(f"[DRON 2][WARN] Línea ignorada en '{path}': {line}")
        print(f"[DRON 2] 🔥 Cargadas {len(fire_coords)} coordenada(s) desde '{path}'.")
    except FileNotFoundError:
        print(f"[DRON 2][ERROR] No se encontró el archivo '{path}'.")
    except Exception as e:
        print(f"[DRON 2][ERROR] Problema al leer '{path}': {e}")

    return fire_coords


def compact_adjacent_fires(fire_coords, max_step=1):
    """
    Elimina fuegos duplicados cuando aparecen en celdas ADYACENTES
    consecutivas (|dx| + |dy| <= max_step).
    """
    if not fire_coords:
        return []

    compact = []
    for (x, y) in fire_coords:
        if not compact:
            compact.append((x, y))
            continue

        last_x, last_y = compact[-1]
        manhattan = abs(x - last_x) + abs(y - last_y)

        if manhattan <= max_step:
            # Es el mismo fuego visto desde la celda de al lado
            print(f"[DRON 2] 🔁 ({x},{y}) se considera el mismo fuego que ({last_x},{last_y}). Se omite.")
            continue

        compact.append((x, y))

    return compact


def adjust_corner_cells(fire_coords):
    """
    Ajusta fuegos detectados en las ESQUINAS del grid para meterlos
    una casilla hacia adentro por seguridad.

    Esquinas consideradas (en coordenadas de celdas):
      - (GRID_MAX_X, 0)
      - (0, GRID_MAX_Y)
      - (GRID_MAX_X, GRID_MAX_Y)

    Regla:
      - (GRID_MAX_X, 0)     -> (GRID_MAX_X - 1, 0)
      - (0, GRID_MAX_Y)     -> (0, GRID_MAX_Y - 1)
      - (GRID_MAX_X, GRID_MAX_Y) -> (GRID_MAX_X - 1, GRID_MAX_Y)

    Si quisieras media casilla, podrías cambiar -1 por -0.5, pero
    para mantener las coordenadas en enteros usamos 1 celda completa.
    """
    if not fire_coords:
        return []

    adjusted = []
    for (x, y) in fire_coords:
        original = (x, y)

        # Esquina superior derecha: (GRID_MAX_X, 0)
        if x == GRID_MAX_X and y == 0:
            x = GRID_MAX_X - 1

        # Esquina inferior izquierda: (0, GRID_MAX_Y)
        elif x == 0 and y == GRID_MAX_Y:
            y = GRID_MAX_Y - 1

        # Esquina inferior derecha: (GRID_MAX_X, GRID_MAX_Y)
        elif x == GRID_MAX_X and y == GRID_MAX_Y:
            x = GRID_MAX_X - 1

        if (x, y) != original:
            print(f"[DRON 2] 🔧 Ajustando coordenada de esquina {original} -> ({x}, {y})")

        adjusted.append((x, y))

    return adjusted


# ============================
# YOLO: CONTROL VISUAL SOBRE FUEGO
# ============================

def transform_frame(frame):
    """Aplica mirror / flips necesarios para que la vista "apunte hacia abajo" igual que en DRON 1."""
    if frame is None:
        return None
    if APPLY_MIRROR:
        frame = cv2.flip(frame, 1)
    return frame


def detect_fire_center(frame, model):
    """
    Corre YOLO y devuelve:
      - has_fire: bool
      - (err_x, err_y): error respecto al centro de la imagen (pixeles)
      - vis: imagen anotada para debug
    """
    if frame is None:
        return False, (0, 0), None

    frame = transform_frame(frame)
    h, w = frame.shape[:2]

    results = model(frame, verbose=False, imgsz=320)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return False, (0, 0), frame

    best_box = None
    best_conf = 0.0

    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls == FIRE_CLASS_ID and conf >= FIRE_CONF_THRES and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return False, (0, 0), frame

    x1, y1, x2, y2 = best_box.xyxy[0]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 0), -1)

    err_x = cx - (w // 2)
    err_y = cy - (h // 2)

    return True, (err_x, err_y), frame

def visual_align_over_fire(flight, camera, model):
    """
    Control visual:
      - Usa YOLO para encontrar fuego.
      - Mueve el dron en a (left/right) y b (forward/back)
        hasta centrar el fuego en la imagen.
      - SOLO sale con éxito cuando lleva FIRE_ALIGN_HOLD_TIME segundos centrado.
    Devuelve:
      - True  si se logró la alineación estable.
      - False si se terminó el tiempo sin lograrla.
    """
    print("[DRON 2][VISUAL] Iniciando alineacion sobre fuego...")
    start = time.time()
    stable_frames = 0
    aligned_since = None  # tiempo desde que entró a zona centrada
    aligned_success = False

    try:
        while time.time() - start < VISUAL_ALIGN_MAX_TIME:
            try:
                frame = camera.read_cv2_image(timeout=2, strategy="newest")
            except Exception as e:
                print("[DRON 2][VISUAL][WARN] Error leyendo frame:", e)
                continue

            has_fire, (err_x, err_y), vis = detect_fire_center(frame, model)

            if not has_fire:
                # No se ve fuego -> resetear alineación y parar movimiento
                stable_frames = 0
                aligned_since = None
                cmd_a = 0
                cmd_b = 0
            else:
                # ¿Está dentro de tolerancia?
                if abs(err_x) < FIRE_ALIGN_TOL_X and abs(err_y) < FIRE_ALIGN_TOL_Y:
                    if aligned_since is None:
                        aligned_since = time.time()
                        stable_frames = 0  # empezamos a contar frames estables
                    stable_frames += 1
                    cmd_a = 0
                    cmd_b = 0
                else:
                    # Se salió de la zona de tolerancia -> resetear hold
                    stable_frames = 0
                    aligned_since = None

                    # Control proporcional en a, b
                    h, w = vis.shape[:2]
                    norm_x = err_x / (w / 2.0)
                    norm_y = err_y / (h / 2.0)

                    cmd_a = int(VISUAL_KP * norm_x * 50.0)
                    cmd_b = int(VISUAL_KP * norm_y * 50.0)

                    cmd_a = max(-RC_VISUAL_MAX, min(RC_VISUAL_MAX, cmd_a))
                    cmd_b = max(-RC_VISUAL_MAX, min(RC_VISUAL_MAX, cmd_b))

            # Enviar RC (misma convención de signos que ya usabas)
            try:
                flight.rc(a=-cmd_a, b=-cmd_b, c=0, d=0)
            except Exception:
                pass

            # Calcular cuánto tiempo lleva centrado
            hold_time = 0.0
            if aligned_since is not None:
                hold_time = time.time() - aligned_since

            # Overlay de estado
            if vis is not None:
                status = f"Alineando... hold={hold_time:.1f}/{FIRE_ALIGN_HOLD_TIME:.1f}s"
                if aligned_since is not None:
                    color = (0, 255, 0) if hold_time >= FIRE_ALIGN_HOLD_TIME else (0, 255, 255)
                else:
                    color = (0, 255, 255)

                cv2.putText(vis, status, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if SHOW_WINDOWS_VISUAL:
                    cv2.imshow("DRON 2 - Visual fuego", vis)
                    cv2.waitKey(1)

            # ✅ Condición de éxito: centrado continuo al menos FIRE_ALIGN_HOLD_TIME
            #   y mínimo FIRE_ALIGN_STABLE_FRAMES frames seguidos en tolerancia
            if (aligned_since is not None
                and hold_time >= FIRE_ALIGN_HOLD_TIME
                and stable_frames >= FIRE_ALIGN_STABLE_FRAMES):
                print(f"[DRON 2][VISUAL] Fuego centrado (alineado {hold_time:.2f} s, "
                      f"{stable_frames} frames estables).")
                aligned_success = True
                break

            time.sleep(VISUAL_ALIGN_DT)

    finally:
        # Siempre detener el movimiento al salir
        try:
            flight.rc(a=0, b=0, c=0, d=0)
        except Exception:
            pass
        time.sleep(0.2)

    if not aligned_success:
        print("[DRON 2][VISUAL][WARN] No se logró una alineación estable antes del timeout.")

    return aligned_success


# ============================
# BÚSQUEDA LOCAL DE FUEGO (YAW SWEEP)
# ============================

def search_fire_yaw_sweep(flight, camera, model,
                          max_sweep_deg=60,
                          step_deg=10):
    """
    Hace un barrido en yaw alrededor del heading actual para intentar
    encontrar un fuego con YOLO.

    Devuelve:
      - found: bool (True si encontró fuego)
      - heading_offset_deg: yaw neto aplicado respecto al heading inicial.

    Si NO encuentra fuego:
      - Devuelve (False, 0.0) y regresa el dron al heading original.
    """
    print("[DRON 2][BUSQUEDA] Iniciando barrido local de yaw para encontrar fuego...")

    current_offset = 0.0  # yaw relativo al inicio (en grados)

    def check_for_fire():
        has_fire = False
        err = (0, 0)
        vis = None
        try:
            frame = camera.read_cv2_image(timeout=2, strategy="newest")
            has_fire, err, vis = detect_fire_center(frame, model)
        except Exception as e:
            print("[DRON 2][BUSQUEDA][WARN] Error leyendo frame:", e)
        return has_fire, err, vis

    # 1) Revisar primero el heading actual (offset = 0)
    has_fire, err, vis = check_for_fire()
    if has_fire:
        print("[DRON 2][BUSQUEDA] Fuego encontrado en heading actual (offset ~ 0°).")
        return True, 0.0

    # 2) Construir secuencia de offsets a explorar:
    #    0 (ya revisado), +step, +2*step, ..., +max,
    #    luego -step, -2*step, ..., -max
    offsets = []
    for k in range(step_deg, max_sweep_deg + 1, step_deg):
        offsets.append(k)
    for k in range(step_deg, max_sweep_deg + 1, step_deg):
        offsets.append(-k)

    # 3) Recorrer cada offset objetivo
    for target_offset in offsets:
        # cuánto hay que girar desde el offset actual
        delta_yaw = target_offset - current_offset

        if not safe_rotate_deg(flight, delta_yaw):
            print("[DRON 2][BUSQUEDA][ERROR] No se pudo completar un paso de rotate en el barrido.")
            # En caso de error en el rotate, intentamos al menos no seguir usando
            # un heading interno incongruente.
            return False, 0.0

        current_offset = target_offset
        time.sleep(0.5)

        has_fire, err, vis = check_for_fire()
        if has_fire:
            print(f"[DRON 2][BUSQUEDA] Fuego encontrado en barrido (offset yaw ~ {current_offset:.1f}°).")
            return True, current_offset

    # 4) Si llegamos aquí, NO encontramos fuego. Regresar al heading original.
    if abs(current_offset) > 1e-3:
        print("[DRON 2][BUSQUEDA] No se encontró fuego. Regresando al heading original...")
        safe_rotate_deg(flight, -current_offset)
        time.sleep(0.5)

    print("[DRON 2][BUSQUEDA] No se encontró fuego durante el barrido local.")
    return False, 0.0


# ============================
# NAVEGACIÓN ENTRE COORDENADAS
# ============================

def move_to_coordinate(flight, current_pos, current_heading_deg, target_pos):
    """
    Mueve el dron desde current_pos = (x,y) hasta target_pos = (x,y)
    usando SIEMPRE:
      - rotate (para alinear yaw con el vector objetivo)
      - go_forward (para avanzar en línea recta)

    Devuelve:
      - (ok, new_pos, new_heading_deg)
    """
    (x0, y0) = current_pos
    (x1, y1) = target_pos

    dx_cells = x1 - x0
    dy_cells = y1 - y0

    if dx_cells == 0 and dy_cells == 0:
        print(f"[DRON 2] Ya estamos en la coordenada {target_pos}.")
        return True, current_pos, current_heading_deg

    dist_cm = math.sqrt(dx_cells**2 + dy_cells**2) * CELL_DIST_DRONE2_CM

    angle_world_rad = math.atan2(dy_cells, dx_cells)
    angle_world_deg = rad2deg(angle_world_rad)

    yaw_change_world = wrap_angle_deg(angle_world_deg - current_heading_deg)

    yaw_change_world_cmd = yaw_change_world
    dist_cmd_cm = dist_cm

    if abs(yaw_change_world) > 90.0:
        sign = 1.0 if yaw_change_world >= 0 else -1.0
        yaw_change_world_cmd = yaw_change_world + sign * EXTRA_YAW_CALIB_DEG
        dist_cmd_cm = dist_cm + EXTRA_DIST_CALIB_CM
        print(f"[DRON 2][CALIB] Giro grande detectado ({yaw_change_world:.1f}°).")
        print(f"              -> Aplicando corrección: yaw +{sign*EXTRA_YAW_CALIB_DEG:.1f}°, dist +{EXTRA_DIST_CALIB_CM} cm")

    if not safe_rotate_deg(flight, yaw_change_world_cmd):
        return False, current_pos, current_heading_deg

    new_heading = wrap_angle_deg(current_heading_deg + yaw_change_world_cmd)

    if not safe_forward_cm(flight, dist_cmd_cm, speed=60):
        return False, current_pos, new_heading

    new_pos = (x1, y1)
    print(f"[DRON 2] Llegamos (aprox) a la celda {new_pos}, heading ~ {new_heading:.1f}°")

    return True, new_pos, new_heading


# ============================
# LÓGICA PRINCIPAL DE LA MISIÓN DEL DRON 2
# ============================

def drone2_main(fire_coordinates):
    """
    Ejecuta la misión del DRON 2 dada la lista de coordenadas de fuego.
    """
    print("========================================")
    print("        MISION DRON 2 - FOLLOW UP")
    print("  Siempre avanzando (FORWARD) + ROTATE")
    print("========================================\n")

    configure_local_ip()

    tl_drone = robot.Drone()
    print("[DRON 2] Inicializando conexión...")
    tl_drone.initialize(conn_type="sta")
    print("[DRON 2] Conexión establecida.\n")

    flight = tl_drone.flight
    camera = tl_drone.camera

    camera.start_video_stream(display=False)
    time.sleep(2.0)

    model = YOLO(YOLO_MODEL_PATH)

    try:
        bat = tl_drone.battery.get_battery()
        print(f"[DRON 2] Batería: {bat}%")
    except Exception:
        print("[DRON 2] No se pudo leer la batería (no es crítico).")

    if not safe_takeoff(flight):
        print("[DRON 2] Abortando misión (falló el despegue).")
        camera.stop_video_stream()
        tl_drone.close()
        return

    current_pos = (0, 0)
    current_heading_deg = 0.0

    for idx, (fx, fy) in enumerate(fire_coordinates):
        print("\n----------------------------------------")
        print(f"[DRON 2] 🔥 Visitando fuego #{idx+1} en coordenada ({fx}, {fy})")
        print("----------------------------------------")

        ok, current_pos, current_heading_deg = move_to_coordinate(
            flight,
            current_pos=current_pos,
            current_heading_deg=current_heading_deg,
            target_pos=(fx, fy)
        )

        if not ok:
            print("[DRON 2][ERROR] Fallo al llegar a un fuego. Abortando misión.")
            break

        found_fire, yaw_offset = search_fire_yaw_sweep(flight, camera, model)

        if found_fire:
            current_heading_deg = wrap_angle_deg(current_heading_deg + yaw_offset)
            print(f"[DRON 2] Ajustando heading interno tras barrido: {current_heading_deg:.1f}°")

            aligned_ok = visual_align_over_fire(flight, camera, model)

            if not aligned_ok:
                print("[DRON 2][WARN] No se logró alineación visual estable. "
                    "Se omite la bajada en este fuego.")
            else:
                if not safe_down_cm(flight, 40):
                    print("[DRON 2][WARN] No se pudo bajar 40 cm sobre el fuego.")
                else:
                    time.sleep(HOVER_TIME_OVER_FIRE)
                    if not safe_up_cm(flight, 40):
                        print("[DRON 2][WARN] No se pudo subir de nuevo tras revisar el fuego.")
                    else:
                        time.sleep(1.0)
        else:
            print("[DRON 2] No se encontró fuego visible en esta coordenada (solo inspección desde altura actual).")

        print("[DRON 2] 🔁 Preparándose para pasar al siguiente fuego...")

    print("\n[DRON 2] Misión de fuegos completada (o abortada por error).")

    print("[DRON 2] Regresando al origen aproximado (0,0)...")
    ok_back, _, _ = move_to_coordinate(
        flight,
        current_pos=current_pos,
        current_heading_deg=current_heading_deg,
        target_pos=(0, 0)
    )
    if not ok_back:
        print("[DRON 2][WARN] No se pudo regresar (0,0) de forma precisa, aterrizando donde está.")

    safe_land(flight)

    camera.stop_video_stream()
    tl_drone.close()
    print("[DRON 2] Misión finalizada. Conexión cerrada.")


def main(fire_coordinates=None):
    print("=" * 60)
    print("    DRON 2: NAVEGACIÓN A FUEGOS DETECTADOS")
    print("    Sistema de coordenadas: X+ forward, Y+ izquierda")
    print("=" * 60)

    if fire_coordinates is None:
        fire_coordinates = load_fire_coordinates_from_txt()

    # 1) Compactar fuegos cercanos (misma zona física)
    fire_coordinates = compact_adjacent_fires(fire_coordinates, max_step=1)

    # 2) Ajustar fuegos que caen justo en las esquinas del grid
    fire_coordinates = adjust_corner_cells(fire_coordinates)

    print(f"[DRON 2] Coordenadas finales a visitar: {fire_coordinates}")

    if not fire_coordinates:
        print("[DRON 2] No hay coordenadas finales a visitar. Saliendo.")
        return

    drone2_main(fire_coordinates)


if __name__ == "__main__":
    print("[DRON 2] Modo standalone: leyendo 'fire_coordinates.txt'\n")

    try:
        connect_to_wifi(TELLO2_WIFI_PROFILE)
        time.sleep(4.0)
    except Exception as e:
        print("[DRON 2][WARN] No se pudo conectar automáticamente al WiFi:", e)
        print("          Asegúrate que la laptop ya está conectada a la red del DRON 2.")

    main()
