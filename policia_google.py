import multiprocessing
import time
import cv2
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime
import io


class MotoMonitor:
    def __init__(self, key_path, folder_id, spreadsheet_id, sheet_name, intervalo_minimo=5):
        # Inicializa credenciales y servicios
        self.credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
        )
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
        
        self.folder_id = folder_id
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name

        # Intervalo mínimo entre ejecuciones por moto
        self.intervalo_minimo = intervalo_minimo
        self.ultima_ejecucion = {}

        # Cola para las tareas
        self.task_queue = multiprocessing.Queue()

        # Proceso separado para manejar las cargas
        self.worker_process = multiprocessing.Process(target=self.procesar_cola, daemon=True)
        self.worker_process.start()

    def verificar_ejecucion(self, moto_id):
        """
        Verifica si se puede ejecutar el procesamiento para una moto específica
        según el tiempo mínimo entre ejecuciones.
        """
        ahora = time.time()
        if moto_id not in self.ultima_ejecucion:
            self.ultima_ejecucion[moto_id] = ahora
            return True

        tiempo_desde_ultima = ahora - self.ultima_ejecucion[moto_id]
        if tiempo_desde_ultima >= self.intervalo_minimo:
            self.ultima_ejecucion[moto_id] = ahora
            return True
        return False

    def subir_imagen(self, image):
        """
        Subir una imagen a Google Drive.
        """
        _, image_encoded = cv2.imencode('.jpg', image)
        image_io = io.BytesIO(image_encoded.tobytes())
        fecha_actual = datetime.now().strftime("%d_%m_%Y-%H_%M_%S_%f")[:-3] + ".jpg"
        file_metadata = {'name': fecha_actual, 'parents': [self.folder_id]}
        media = MediaIoBaseUpload(image_io, mimetype='image/jpeg')
        file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return fecha_actual, file.get('id')

    def registrar_datos(self, id_imagen, nombre_imagen, cantidad_personas, cantidad_moto, tupla_objeto):
        """
        Registrar datos en Google Sheets.
        """
        result = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f'{self.sheet_name}!A:A'
        ).execute()
        values = result.get('values', [])
        siguiente_fila = len(values) + 1
        rango = f'{self.sheet_name}!A{siguiente_fila}'
        fecha_actual = datetime.now()
        datos_a_agregar = [
            [
                nombre_imagen.split(".")[0],
                fecha_actual.strftime("%d-%m-%Y"),
                fecha_actual.strftime("%H:%M:%S"),
                nombre_imagen,
                float(cantidad_personas),
                float(cantidad_moto),
                tupla_objeto
            ]
        ]
        self.sheets_service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range=rango,
            valueInputOption='USER_ENTERED',
            body={'values': datos_a_agregar}
        ).execute()

    def procesar_cola(self):
        """
        Proceso separado para manejar las tareas en la cola.
        """
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # Terminar proceso si se encuentra un marcador None
                    break

                image, cantidad_personas, cantidad_moto, tupla_objeto_json = task
                nombre_imagen, id_archivo = self.subir_imagen(image)
                self.registrar_datos(id_archivo, nombre_imagen, cantidad_personas, cantidad_moto, tupla_objeto_json)
                print(f"Procesado en segundo plano: {nombre_imagen} con ID {id_archivo}")
            except Exception as e:
                print(f"Error al procesar la tarea: {e}")

    def procesar_moto_sospechosa(self, image, tupla_objeto):
        """
        Método principal para agregar tareas de procesamiento a la cola.
        """
        rompen_regla = [x for x in tupla_objeto if x[1] > 1]
        if not len(rompen_regla):
            return

        for i, moto in enumerate(rompen_regla):
            moto_id = f"moto{i+1}"
            if self.verificar_ejecucion(moto_id):
                cantidad_moto = len(tupla_objeto)
                cantidad_personas = sum([x[1] for x in tupla_objeto])
                tupla_objeto_json = self.transformar_motos_con_personas(tupla_objeto)

                # Agregar tarea a la cola
                self.task_queue.put((image, cantidad_personas, cantidad_moto, tupla_objeto_json))

    def transformar_motos_con_personas(self, motos_con_personas):
        """
        Transforma una lista de detecciones en el formato requerido para 'tupla_objeto'.
        """
        objetos = []
        for i, (_, cantidad_personas, _) in enumerate(motos_con_personas):
            moto_data = {
                "moto": f"moto{i+1}",
                "personas": [f"persona{j+1}" for j in range(cantidad_personas)]
            }
            objetos.append(moto_data)
        return json.dumps(objetos, indent=2)
