# mission_drone2_v1.py
# -*- coding: utf-8 -*-
"""
DRON 2: VISITA COORDENADAS DE FUEGO SIEMPRE AVANZANDO (FORWARD)

Convenciones del sistema de coordenadas (MISMO que DRON 1):

- Unidad de coordenadas: "celdas".
- CELL_DIST_CM = 40 cm por celda.

- Ejes en el plano:
    +X  -> hacia adelante desde el punto de despegue.
    +Y  -> hacia la IZQUIERDA respecto al eje +X.

Requisitos de movimiento:
- El dron SOLO se desplaza físicamente usando:
    - forward (adelante)
    - rotate (giro en yaw)
- NO se usan left / right / back para traslación.
- Para ir de (x1, y1) a (x2, y2):
    1) Se calcula el vector Δ = (dx, dy) en el plano.
    2) Se calcula el ángulo objetivo usando atan2(dy, dx).
    3) Se gira ese ángulo relativo al heading actual.
    4) Se avanza la distancia de la hipotenusa.
"""

import os
import time
import math
import socket
import subprocess

import robomaster
from robomaster import robot

# ============================
# CONFIGURACIÓN GENERAL
# ============================

CELL_DIST_CM = 40           # cm por celda (igual que DRON 1)
HOVER_TIME_OVER_FIRE = 1.0  # tiempo en hover en la cota baja (s)

# WiFi opcional para modo standalone (Windows)
TELLO2_WIFI_PROFILE = "TELLO-FE1B95"
WIFI_INTERFACE_NAME = "Wi-Fi"


# ============================
# UTILIDADES WiFi / IP
# ============================

def connect_to_wifi(profile_name: str,
                    interface_name: str = WIFI_INTERFACE_NAME,
                    timeout: int = 25):
    print(f"\n[WiFi] Conectando al perfil/SSID '{profile_name}' en interfaz '{interface_name}'...")
    cmd = [
        "netsh", "wlan", "connect",
        f"name={profile_name}",
        f"ssid={profile_name}",
        f"interface={interface_name}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[WiFi][ERROR] al intentar conectar:")
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("No se pudo lanzar el comando de conexión WiFi. Revisa el nombre del perfil/SSID.")

    start = time.time()
    while time.time() - start < timeout:
        status = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
        )
        if status.returncode == 0 and profile_name in status.stdout:
            print(f"[WiFi] Conectado correctamente a '{profile_name}'.")
            return
        time.sleep(1.0)

    print("[WiFi][WARN] Tiempo de espera agotado. Verifica la conexión manualmente.")


def configure_local_ip():
    try:
        DRONE_IP = "192.168.10.1"
        DRONE_PORT = 8889

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((DRONE_IP, DRONE_PORT))
        local_ip = s.getsockname()[0]
        s.close()

        robomaster.config.LOCAL_IP_STR = local_ip
        print(f"[IP][DRON 2] LOCAL_IP_STR configurada a {local_ip}")
    except Exception as e:
        print("[IP][DRON 2][ERROR] No se pudo detectar IP local, usando 0.0.0.0:", e)
        robomaster.config.LOCAL_IP_STR = "0.0.0.0"


# ============================
# MOVIMIENTOS SEGUROS
# ============================

def safe_takeoff(flight):
    print("[DRON 2][SAFE] Despegando...")
    try:
        flight.takeoff()
        time.sleep(4.0)
        print("[DRON 2][SAFE] Takeoff completado (asumido por tiempo).")
        return True
    except Exception as e:
        print("[DRON 2][SAFE][ERROR] Error en takeoff:", e)
        return False


def safe_land(flight):
    print("[DRON 2][SAFE] Aterrizando...")
    try:
        flight.land()
        time.sleep(5.0)
        print("[DRON 2][SAFE] Land completado (asumido por tiempo).")
        return True
    except Exception as e:
        print("[DRON 2][SAFE][ERROR] Error en land:", e)
        return False


def safe_rotate_yaw(flight, yaw_deg):
    """
    Gira 'yaw_deg' grados:
      - En tu setup, valor NEGATIVO = giro a la IZQUIERDA.
    """
    yaw_cmd = int(round(yaw_deg))
    if abs(yaw_cmd) < 2:
        return True

    print(f"[DRON 2][SAFE] Rotando {yaw_cmd} grados (yaw)...")
    try:
        flight.rc(a=0, b=0, c=0, d=0)
    except Exception:
        pass

    time.sleep(0.2)

    try:
        flight.rotate(yaw_cmd)
        turn_speed_deg = 60.0   # deg/s aprox
        est_time = abs(yaw_cmd) / turn_speed_deg + 1.0

        print(f"[DRON 2][SAFE] Esperando {est_time:.1f} s para completar giro...")
        time.sleep(est_time)

        print("[DRON 2][SAFE] Rotación completada (asumida por tiempo).")
        return True
    except Exception as e:
        print("[DRON 2][SAFE][ERROR] Error en rotate:", e)
        return False


def safe_forward_cm(flight, dist_cm):
    """
    Avanza dist_cm hacia ADELANTE (forward) con un solo comando,
    siempre que la distancia no exceda el rango típico del SDK.
    Solo si es MUY grande se divide en segmentos.
    """
    dist_cm = float(dist_cm)
    if dist_cm <= 0:
        return True

    MAX_CMD_CM = 480.0  # margen por debajo del límite típico (500 cm)
    print(f"[DRON 2][SAFE] Avanzando {dist_cm:.1f} cm hacia adelante...")

    # Detener RC antes de mover
    try:
        for _ in range(3):
            flight.rc(a=0, b=0, c=0, d=0)
            time.sleep(0.03)
    except Exception:
        pass
    time.sleep(0.2)

    # Caso normal: un solo comando forward
    if dist_cm <= MAX_CMD_CM:
        try:
            cmd_dist = int(round(dist_cm))
            flight.forward(cmd_dist)
            # velocidad estimada ~30 cm/s + colchón
            speed_est = 30.0
            est_time = dist_cm / speed_est + 1.5
            # límite razonable de espera (por si las dudas)
            est_time = min(est_time, 25.0)

            print(f"[DRON 2][SAFE]     Esperando {est_time:.1f} s para forward...")
            time.sleep(est_time)

            print("[DRON 2][SAFE] Forward completado (asumido por tiempo).")
            return True
        except Exception as e:
            print(f"[DRON 2][SAFE][ERROR] Error en forward de {dist_cm:.1f} cm: {e}")
            return False

    # Caso extremo: distancia > MAX_CMD_CM, dividir en segmentos (raro en tu grid)
    print("[DRON 2][SAFE][WARN] Distancia muy grande; se dividira en segmentos.")
    remaining = dist_cm
    while remaining > 0:
        step = min(remaining, MAX_CMD_CM)
        try:
            cmd_dist = int(round(step))
            flight.forward(cmd_dist)
            speed_est = 30.0
            est_time = step / speed_est + 1.5
            est_time = min(est_time, 12.0)
            print(f"[DRON 2][SAFE]  -> Segmento forward {step:.1f} cm (esperando {est_time:.1f} s)")
            time.sleep(est_time)
        except Exception as e:
            print(f"[DRON 2][SAFE][ERROR] Error en segmento forward de {step:.1f} cm: {e}")
            return False
        remaining -= step

    print("[DRON 2][SAFE] Forward total completado (segmentado por límite de comando).")
    return True


def safe_down_cm(flight, dist_cm):
    dist_cm = float(dist_cm)
    if dist_cm <= 0:
        return True

    print(f"[DRON 2][SAFE] Bajando {dist_cm:.1f} cm...")
    try:
        flight.rc(a=0, b=0, c=0, d=0)
    except Exception:
        pass
    time.sleep(0.2)

    try:
        flight.down(int(round(dist_cm)))
        speed_est = 20.0  # cm/s aprox descendiendo
        est_time = dist_cm / speed_est + 1.0
        print(f"[DRON 2][SAFE]     Esperando {est_time:.1f} s para bajar...")
        time.sleep(est_time)
        print("[DRON 2][SAFE] Descenso completado (asumido por tiempo).")
        return True
    except Exception as e:
        print(f"[DRON 2][SAFE][ERROR] Error en down: {e}")
        return False


def safe_up_cm(flight, dist_cm):
    dist_cm = float(dist_cm)
    if dist_cm <= 0:
        return True

    print(f"[DRON 2][SAFE] Subiendo {dist_cm:.1f} cm...")
    try:
        flight.rc(a=0, b=0, c=0, d=0)
    except Exception:
        pass
    time.sleep(0.2)

    try:
        flight.up(int(round(dist_cm)))
        speed_est = 20.0  # cm/s aprox subiendo
        est_time = dist_cm / speed_est + 1.0
        print(f"[DRON 2][SAFE]     Esperando {est_time:.1f} s para subir...")
        time.sleep(est_time)
        print("[DRON 2][SAFE] Ascenso completado (asumido por tiempo).")
        return True
    except Exception as e:
        print(f"[DRON 2][SAFE][ERROR] Error en up: {e}")
        return False


# ============================
# LECTURA DE COORDENADAS (TXT)
# ============================

def load_fire_coordinates_from_txt(path: str = None):
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "fire_coordinates.txt")

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


# ============================
# NAVEGACIÓN SIEMPRE FORWARD
# ============================

def move_to_coordinate(flight, curr_x, curr_y, curr_heading_deg,
                       target_x, target_y):
    dx_cells = target_x - curr_x
    dy_cells = target_y - curr_y

    print(f"\n[DRON 2] Objetivo: ({target_x}, {target_y}) "
          f"desde ({curr_x}, {curr_y}), heading actual = {curr_heading_deg:.1f}°")
    print(f"[DRON 2] Δx = {dx_cells} celdas, Δy = {dy_cells} celdas")

    if dx_cells == 0 and dy_cells == 0:
        print("[DRON 2] Ya estamos en esa coordenada. No se mueve.")
        return curr_x, curr_y, curr_heading_deg, True

    dx_cm = dx_cells * CELL_DIST_CM
    dy_cm = dy_cells * CELL_DIST_CM

    desired_heading_deg = math.degrees(math.atan2(dy_cm, dx_cm))
    print(f"[DRON 2] Heading deseado (mundo) = {desired_heading_deg:.1f}°")

    yaw_change_world = desired_heading_deg - curr_heading_deg

    while yaw_change_world > 180.0:
        yaw_change_world -= 360.0
    while yaw_change_world < -180.0:
        yaw_change_world += 360.0

    yaw_cmd = -yaw_change_world
    print(f"[DRON 2] Δheading mundo = {yaw_change_world:.1f}°  -> yaw_cmd = {yaw_cmd:.1f}° (sdk)")

    if not safe_rotate_yaw(flight, yaw_cmd):
        print("[DRON 2][ERROR] Fallo al girar hacia la coordenada.")
        return curr_x, curr_y, curr_heading_deg, False

    dist_cm = math.hypot(dx_cm, dy_cm)
    print(f"[DRON 2] Distancia a recorrer = {dist_cm:.1f} cm")

    if not safe_forward_cm(flight, dist_cm):
        print("[DRON 2][ERROR] Fallo al avanzar hacia la coordenada.")
        return curr_x, curr_y, curr_heading_deg, False

    curr_x = target_x
    curr_y = target_y
    curr_heading_deg = desired_heading_deg

    while curr_heading_deg > 180.0:
        curr_heading_deg -= 360.0
    while curr_heading_deg < -180.0:
        curr_heading_deg += 360.0

    print(f"[DRON 2] Llegó a ({curr_x}, {curr_y}) con heading ≈ {curr_heading_deg:.1f}°")
    return curr_x, curr_y, curr_heading_deg, True


# ============================
# MISIÓN PRINCIPAL DRON 2
# ============================

def drone2_main(fire_coordinates):
    if fire_coordinates is None:
        fire_coordinates = []

    print("========================================")
    print("        MISION DRON 2 - FOLLOW UP       ")
    print("  Siempre avanzando (FORWARD) + ROTATE  ")
    print("========================================\n")

    print(f"[DRON 2] Coordenadas de fuego recibidas: {fire_coordinates}")

    if not fire_coordinates:
        print("[DRON 2][WARN] No hay coordenadas de fuego. Saliendo sin volar.")
        return

    configure_local_ip()

    print("[DRON 2] Inicializando dron...")
    tl_drone = robot.Drone()
    tl_drone.initialize()
    flight = tl_drone.flight

    curr_x, curr_y = 0, 0
    curr_heading_deg = 0.0

    try:
        try:
            bat = tl_drone.battery.get_battery()
            print(f"[DRON 2] Batería: {bat}%")
        except Exception:
            print("[DRON 2] No se pudo leer la batería.")

        if not safe_takeoff(flight):
            print("[DRON 2][ERROR] No se pudo despegar. Abortando.")
            return

        time.sleep(1.5)

        # Recorrer cada coordenada de fuego en ORDEN
        for idx, (fx, fy) in enumerate(fire_coordinates, start=1):
            print(f"\n[DRON 2] ***** OBJETIVO {idx}/{len(fire_coordinates)}: ({fx}, {fy}) *****")

            curr_x, curr_y, curr_heading_deg, ok = move_to_coordinate(
                flight, curr_x, curr_y, curr_heading_deg, fx, fy
            )
            if not ok:
                print("[DRON 2][ERROR] Fallo al llegar a un fuego. Abortando misión.")
                break

            # Maniobra sobre el fuego: bajar 40 cm y volver a altura base
            print(f"[DRON 2] Maniobra vertical sobre fuego en ({fx}, {fy}): bajar/subir 40 cm...")
            try:
                flight.rc(a=0, b=0, c=0, d=0)
            except Exception:
                pass
            time.sleep(0.3)

            if not safe_down_cm(flight, 40):
                print("[DRON 2][WARN] No se pudo bajar 40 cm sobre el fuego.")
            else:
                time.sleep(HOVER_TIME_OVER_FIRE)
                if not safe_up_cm(flight, 40):
                    print("[DRON 2][WARN] No se pudo subir de regreso a la altura base.")
                else:
                    time.sleep(0.3)

        # Regresar a origen (0,0)
        print("\n[DRON 2] Regresando al origen (0,0)...")
        curr_x, curr_y, curr_heading_deg, ok = move_to_coordinate(
            flight, curr_x, curr_y, curr_heading_deg, 0, 0
        )
        if not ok:
            print("[DRON 2][WARN] Fallo al regresar al origen. Aterrizando donde está.")

        safe_land(flight)
        print("[DRON 2] Misión completada, dron en el suelo.")

    except Exception as e:
        print("[DRON 2][ERROR] EXCEPCION durante la misión:", e)
        try:
            safe_land(flight)
        except Exception:
            pass

    finally:
        try:
            tl_drone.close()
        except Exception:
            pass
        print("[DRON 2] Conexión cerrada.")


# ============================
# MAIN GENÉRICO / STANDALONE
# ============================

def main(fire_coordinates=None):
    print("=" * 60)
    print("    DRON 2: NAVEGACIÓN A FUEGOS DETECTADOS")
    print("    Sistema de coordenadas: X+ forward, Y+ izquierda")
    print("=" * 60)

    if fire_coordinates is None:
        fire_coordinates = load_fire_coordinates_from_txt()

    if not fire_coordinates:
        print("[DRON 2][WARN] No hay coordenadas de fuego válidas. No se despega.")
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
