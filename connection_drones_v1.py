# -*- coding: utf-8 -*-
"""
Orquestador de 2 drones usando scripts separados, pero
importándolos como módulos (SIN subprocess).

Flujo:
 1) Conectarse al WiFi del DRON 1.
 2) Llamar mission_drone1.main().
 3) Conectarse al WiFi del DRON 2.
 4) Llamar mission_drone2.main().
"""

import sys
import os
import time
import subprocess

# ========= CONFIG WiFi (AJUSTA NOMBRES) =========
TELLO1_WIFI_PROFILE = "TELLO-FE1A04"   # SSID / perfil del dron 1
TELLO2_WIFI_PROFILE = "TELLO-FE1B95"   # SSID / perfil del dron 2
WIFI_INTERFACE_NAME = "Wi-Fi"         # Nombre de la interfaz en Windows

# ========= CONFIG IMPORTS (aseguramos que pueda ver los scripts) =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import mission_drone1_v1   # <-- tu script del dron 1 (el que pegaste arriba)
import mission_drone2_v1   # <-- tu script simple del dron 2


def connect_to_wifi(profile_name: str,
                    interface_name: str = WIFI_INTERFACE_NAME,
                    timeout: int = 25):
    """
    Conecta a una red WiFi usando 'netsh wlan connect'.
    Necesitas haber guardado antes el perfil conectándote al Tello al menos una vez.
    """
    print(f"\n[WiFi] Conectando al perfil/SSID '{profile_name}' en interfaz '{interface_name}'...")
    cmd = [
        "netsh", "wlan", "connect",
        f"name={profile_name}",
        f"ssid={profile_name}",
        f"interface={interface_name}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[WiFi] ERROR al intentar conectar:")
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("No se pudo lanzar el comando de conexión WiFi. Revisa el nombre del perfil/SSID.")

    # Esperar a que se establezca la conexión
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

    print("[WiFi] Tiempo de espera agotado. Verifica la conexión manualmente.")


def main():
    print("==========================================================")
    print("   ORQUESTADOR 2 DRONES (IMPORTANDO mission_droneX)       ")
    print("==========================================================")
    print("\nFlujo:")
    print(" 1) Conectarse al WiFi del DRON 1.")
    print(" 2) Ejecutar mission_drone1.main().")
    print(" 3) Conectarse al WiFi del DRON 2.")
    print(" 4) Ejecutar mission_drone2.main().")
    print("==========================================================\n")

    # -------- DRON 1 --------
    connect_to_wifi(TELLO1_WIFI_PROFILE)
    print("[ORQ] Esperando 8 segundos para que la red del DRON 1 se estabilice...")
    time.sleep(8.0)

    print("\n[ORQ] Lanzando misión del DRON 1 (mission_drone1.main())...\n")
    mission_drone1_v1.main()   # <- aquí se ejecuta TODO tu código del dron 1

    # -------- DRON 2 --------
    connect_to_wifi(TELLO2_WIFI_PROFILE)
    print("[ORQ] Esperando 8 segundos para que la red del DRON 2 se estabilice...")
    time.sleep(8.0)

    print("\n[ORQ] Lanzando misión del DRON 2 (mission_drone2.main())...\n")
    mission_drone2_v1.main()   # <- aquí se ejecuta la misión simple del dron 2

    print("\n[INFO] Flujo del orquestador terminado.")


if __name__ == "__main__":
    main()
