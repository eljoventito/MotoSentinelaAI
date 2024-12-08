import cv2
import time
import numpy as np
from policia_google import MotoMonitor

from datetime import datetime
from ultralytics import YOLO
import sys
import os

class_colors = {0: (0, 165, 255), 1: (255, 191, 0)}  # Moto: naranja, Persona: celeste
output_dir = ''
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)


import random
moto_colors = [
    (0, 165, 255),   # Naranja
    (0, 128, 255),   # Azul claro
    (0, 255, 128),   # Verde neón
    (128, 0, 255),   # Fucsia
    (255, 255, 0),   # Amarillo
    (255, 128, 0)    # Naranja intenso
]
persona_colors = [
    (255, 191, 0),   # Celeste
    (255, 0, 128),   # Rosa intenso
    (128, 255, 0),   # Verde lima
    (0, 255, 255),   # Cian
    (255, 0, 255),   # Magenta
    (0, 0, 255)      # Rojo intenso
]

def assign_color_by_index(index, object_type):
    """
    Asigna un color único a un objeto basado en su índice y tipo.

    Args:
        index: Índice de la instancia.
        object_type: Tipo de objeto ('moto' o 'persona').

    Returns:
        Un color en formato BGR.
    """
    if object_type == "moto":
        return moto_colors[index % len(moto_colors)]
    elif object_type == "persona":
        return persona_colors[index % len(persona_colors)]
    else:
        return (255, 255, 255)  # Color por defecto (blanco)
    


def resize_with_aspect_and_padding(image, target_size):
    """
    Redimensiona una imagen manteniendo la relación de aspecto y añade relleno blanco.

    Args:
        image: Imagen original.
        target_size: Tamaño objetivo (ancho, alto).

    Returns:
        Imagen redimensionada y rellenada, junto con los parámetros de escala y offset.
    """
    h, w = image.shape[:2]
    scale = min(target_size[1] / h, target_size[0] / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return padded_image, (scale, x_offset, y_offset)


def detect_people_on_motorbikes(results, threshold=0.5):
    """
    Identifica cuántas personas están asociadas a cada moto.

    Args:
        results: Resultado del modelo YOLO (objeto ultralytics).
        threshold: Umbral para considerar una detección válida basada en confianza.

    Returns:
        Listado de motos con el número de personas asociadas.
    """
    motos = []
    personas = []

    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = box.conf.item()
        if confidence > threshold:
            bbox = box.xyxy.cpu().numpy().flatten()
            if class_id == 0:  # ID de la clase moto
                motos.append(bbox)
            elif class_id == 1:  # ID de la clase persona
                personas.append(bbox)

    motos_con_personas = []
    for moto in motos:
        moto_xmin, moto_ymin, moto_xmax, moto_ymax = moto
        personas_cercanas = 0
        for persona in personas:
            px_min, py_min, px_max, py_max = persona
            if not (px_max < moto_xmin or px_min > moto_xmax or py_max < moto_ymin or py_min > moto_ymax):
                personas_cercanas += 1
        motos_con_personas.append((moto, personas_cercanas))

    return motos_con_personas

def detect_people_on_motorbikes(results, threshold=0.5):
    """
    Identifica cuántas personas están asociadas a cada moto y calcula un cuadro extendido que englobe
    la moto y las personas asociadas.

    Args:
        results: Resultado del modelo YOLO (objeto ultralytics).
        threshold: Umbral para considerar una detección válida basada en confianza.

    Returns:
        Listado de motos con:
        - Coordenadas de la moto.
        - Número de personas asociadas.
        - Cuadro extendido que engloba moto y personas asociadas.
    """
    motos = []
    personas = []

    # Separar detecciones de motos y personas
    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = box.conf.item()
        if confidence > threshold:
            bbox = box.xyxy.cpu().numpy().flatten()
            if class_id == 0:  # ID de la clase moto
                motos.append(bbox)
            elif class_id == 1:  # ID de la clase persona
                personas.append(bbox)

    motos_con_personas = []
    for moto in motos:
        moto_xmin, moto_ymin, moto_xmax, moto_ymax = moto
        personas_cercanas = 0

        # Inicializar cuadro extendido con las coordenadas de la moto
        extended_xmin, extended_ymin = moto_xmin, moto_ymin
        extended_xmax, extended_ymax = moto_xmax, moto_ymax

        # Calcular personas asociadas y extender el cuadro
        for persona in personas:
            px_min, py_min, px_max, py_max = persona
            if not (px_max < moto_xmin or px_min > moto_xmax or py_max < moto_ymin or py_min > moto_ymax):
                personas_cercanas += 1

                # Expandir el cuadro extendido para incluir a la persona
                extended_xmin = min(extended_xmin, px_min)
                extended_ymin = min(extended_ymin, py_min)
                extended_xmax = max(extended_xmax, px_max)
                extended_ymax = max(extended_ymax, py_max)

        # Agregar la moto, el número de personas asociadas y el cuadro extendido
        motos_con_personas.append((moto, personas_cercanas, [extended_xmin, extended_ymin, extended_xmax, extended_ymax]))

    return motos_con_personas



def process_and_draw_segmentations(image, results, class_colors, scale, x_offset, y_offset, original_size, draw_box_object=False):
    """
    Dibuja las segmentaciones y cajas, ajustando las coordenadas a las dimensiones originales.
    Asigna un color único y más distintivo para cada instancia de objeto detectado.

    Args:
        image: Imagen procesada (con padding).
        results: Resultados del modelo YOLO.
        class_colors: Diccionario base de colores para cada clase.
        scale: Escala usada para redimensionar.
        x_offset, y_offset: Offset aplicado durante el redimensionado.
        original_size: Dimensiones originales (ancho, alto).
        draw_box_object: Si True, dibuja también los cuadros delimitadores alrededor de los objetos detectados.
    """
    original_height, original_width = original_size

    for i, box in enumerate(results[0].boxes):
        # Convertir coordenadas de vuelta a la resolución original
        bbox = box.xyxy.cpu().numpy().flatten()
        bbox[0] = int((bbox[0] - x_offset) / scale)
        bbox[1] = int((bbox[1] - y_offset) / scale)
        bbox[2] = int((bbox[2] - x_offset) / scale)
        bbox[3] = int((bbox[3] - y_offset) / scale)

        conf = box.conf.item()
        class_id = int(box.cls)
        # Generar un color completamente aleatorio para cada instancia
        if class_id == 0:  # Clase "moto"
            instance_color = assign_color_by_index(i, "moto")
        elif class_id == 1:  # Clase "persona"
            instance_color = assign_color_by_index(i, "persona")
        else:
            instance_color = (255, 255, 255)  # Color por defecto

        # Dibujar el cuadro con el color único
        if draw_box_object:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), instance_color, 2)
        
            # Etiqueta
            label = f"{class_id}: {conf:.2f}"
            cv2.putText(image, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, instance_color, 2)

    if results[0].masks is not None:
        for i, (mask, box) in enumerate(zip(results[0].masks.data, results[0].boxes)):
            class_id = int(box.cls)
            if class_id in class_colors:
                mask = mask.cpu().numpy()
                # Redimensionar la máscara de vuelta a la resolución original
                resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                # Convertir la máscara a binaria
                binary_mask = resized_mask > 0.5

                class_id = int(box.cls)
                # Generar un color completamente aleatorio para cada instancia
                if class_id == 0:  # Clase "moto"
                    instance_color = assign_color_by_index(i, "moto")
                elif class_id == 1:  # Clase "persona"
                    instance_color = assign_color_by_index(i, "persona")
                else:
                    instance_color = (255, 255, 255)  # Color por defecto

                overlay = image.copy()
                color = np.array(instance_color, dtype=np.uint8)
                image[binary_mask] = (
                    color * 0.5 + image[binary_mask] * 0.5
                ).astype(np.uint8)

    return image


def process_and_save(image, model, target_size, filename_base, save_original, save_with_box, save_without_box):
    """
    Procesa y guarda imágenes con opciones de segmentación y cajas delimitadoras.

    Args:
        image: Imagen original.
        model: Modelo YOLO.
        target_size: Tamaño objetivo de la imagen.
        filename_base: Nombre base para guardar imágenes.
    """
    try:
        if save_original:
            original_filename = f"{filename_base}_ORI.jpg"
            cv2.imwrite(original_filename, image)

        processed_image, _ = resize_with_aspect_and_padding(image, target_size)
        results = model(processed_image)

        if save_with_box and results[0].masks is not None:
            result_image = results[0].plot()
            modified_filename = f"{filename_base}_BOX.jpg"
            cv2.imwrite(modified_filename, result_image)

        if save_without_box and results[0].masks is not None:
            modified_image = process_and_draw_segmentations(image.copy(), results, class_colors, draw_boxes=False)
            modified_filename = f"{filename_base}_SEG.jpg"
            cv2.imwrite(modified_filename, modified_image)

    except Exception as e:
        print(f"Error al procesar y guardar las imágenes: {e}")



def live_detection(model, nivel_confianza, target_size=(864, 480), save_original=True, save_with_box=True, save_without_box=True):
    ultima_deteccion = 0
    intervalo_guardado = 0.5
    """
    Realiza detección en vivo usando una cámara.
    Args:
        model: Modelo YOLO.
        nivel_confianza: Nivel de confianza mínimo.
    """
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame del video.")
            break
        frame_original = frame.copy()
        processed_frame, (scale, x_offset, y_offset) = resize_with_aspect_and_padding(frame, target_size)
        results = model(processed_frame, conf=nivel_confianza)
    
        # Detectar motos y personas asociadas
        motos_con_personas = detect_people_on_motorbikes(results)
    
        # Dibujar cuadros y segmentaciones ajustadas a la resolución original
        frame = process_and_draw_segmentations(frame, results, class_colors, scale, x_offset, y_offset, original_size=(frame.shape[0], frame.shape[1])      )
    
        for moto, count in motos_con_personas:
            # Extraer y escalar las coordenadas de la moto a la resolución original
            moto_xmin = int((moto[0] - x_offset) / scale)
            moto_ymin = int((moto[1] - y_offset) / scale)
            moto_xmax = int((moto[2] - x_offset) / scale)
            moto_ymax = int((moto[3] - y_offset) / scale)
        
            # Definir color: verde para 1 persona, rojo para 2 o más personas
            color = (0, 255, 0) if count == 1 else (0, 0, 255)
        
            # Dibujar cuadro alrededor de la moto
            cv2.rectangle(frame, (moto_xmin, moto_ymin), (moto_xmax, moto_ymax), color, 3)
        
            # Añadir etiqueta indicando el número de personas
            cv2.putText( frame, f"{count} personas", (moto_xmin, moto_ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2            )
    
        # Mostrar la transmisión en vivo
        cv2.imshow("YOLOv8 Detección - Transmisión en Vivo", frame)
    
        # Guardar imágenes si hay detección y pasa el intervalo
        tiempo_actual = time.time()
        if motos_con_personas and tiempo_actual - ultima_deteccion > intervalo_guardado:
            ultima_deteccion = tiempo_actual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{output_dir}/deteccion_{timestamp}"
            process_and_save(frame_original, model, target_size, filename_base, save_original, save_with_box, save_without_box)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_extended_boxes(image, motos_con_personas, scale, x_offset, y_offset):
    """
    Dibuja las cajas extendidas que engloban motos y personas asociadas en la imagen.

    Args:
        image: Imagen sobre la que se dibujarán las cajas.
        motos_con_personas: Lista de tuplas con información de motos, número de personas asociadas y cuadro extendido.
        scale: Escala utilizada para redimensionar las coordenadas.
        x_offset, y_offset: Offset aplicado durante el redimensionado.

    Returns:
        La imagen con las cajas extendidas dibujadas.
    """
    for moto, count, extended_box in motos_con_personas:
        # Extraer las coordenadas del cuadro extendido
        extended_xmin = int((extended_box[0] - x_offset) / scale)
        extended_ymin = int((extended_box[1] - y_offset) / scale)
        extended_xmax = int((extended_box[2] - x_offset) / scale)
        extended_ymax = int((extended_box[3] - y_offset) / scale)

        # Definir color: verde para 1 persona, rojo para 2 o más personas
        color = (0, 255, 0) if count <= 1 else (0, 0, 255)
        text_label = "Uso correcto" if count <= 1 else 'Uso sospechoso'
        # Dibujar cuadro extendido
        cv2.rectangle(image, (extended_xmin, extended_ymin), (extended_xmax, extended_ymax), color, 3)
        
        # Añadir etiqueta indicando el número de personas
        cv2.putText(
            image,
            f"{text_label}",
            (extended_xmin, extended_ymax + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    return image



def process_video(video_path, model, target_size, nivel_confianza, max_distance, output_path=None , monitor = None):
    """
    Procesa un video para detectar motos y personas asociadas y genera un video con las detecciones dibujadas.

    Args:
        video_path: Ruta del video de entrada.
        model: Modelo YOLO cargado.
        target_size: Tamaño objetivo para redimensionar los fotogramas.
        nivel_confianza: Nivel de confianza mínimo para las detecciones.
        max_distance: Distancia máxima permitida entre una persona y una moto.
        output_path: Ruta para guardar el video procesado (opcional).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Configurar el video de salida (si se especifica una ruta)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el fotograma con padding
        processed_frame, (scale, x_offset, y_offset) = resize_with_aspect_and_padding(frame, target_size)

        # Realizar predicción
        results = model(processed_frame, conf=nivel_confianza)

        # Detectar motos y personas
        motos_con_personas = detect_people_on_motorbikes(results, threshold=nivel_confianza, max_distance=max_distance)

        # Dibujar segmentaciones y cajas
        frame = process_and_draw_segmentations(
            frame, results, class_colors, scale, x_offset, y_offset, original_size=(frame.shape[0], frame.shape[1]), draw_box_object=False
        )

        # Dibujar las cajas extendidas
        frame = draw_extended_boxes(frame, motos_con_personas, scale, x_offset, y_offset)
        if monitor:
            monitor.procesar_moto_sospechosa(frame, motos_con_personas)
        # Mostrar el fotograma con detecciones
        cv2.imshow("Detección en Video", frame)

        # Escribir el fotograma procesado en el video de salida
        if output_path:
            out.write(frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


def detect_people_on_motorbikes(results, threshold=0.5, max_distance=50):
    """
    Identifica cuántas personas están asociadas a cada moto y calcula un cuadro extendido que englobe
    la moto y las personas asociadas.

    Args:
        results: Resultado del modelo YOLO (objeto ultralytics).
        threshold: Umbral para considerar una detección válida basada en confianza.
        max_distance: Distancia máxima permitida entre la moto y las personas asociadas.

    Returns:
        Listado de motos con:
        - Coordenadas de la moto.
        - Número de personas asociadas.
        - Cuadro extendido que engloba moto y personas asociadas.
    """
    motos = []
    personas = []

    # Separar detecciones de motos y personas
    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = box.conf.item()
        if confidence > threshold:
            bbox = box.xyxy.cpu().numpy().flatten()
            if class_id == 0:  # ID de la clase moto
                motos.append(bbox)
            elif class_id == 1:  # ID de la clase persona
                personas.append(bbox)

    motos_con_personas = []
    print(f"Cantidad de Motos: {len(motos)}")
    for moto in motos:
        moto_xmin, moto_ymin, moto_xmax, moto_ymax = moto
        personas_cercanas = 0

        # Inicializar cuadro extendido con las coordenadas de la moto
        extended_xmin, extended_ymin = moto_xmin, moto_ymin
        extended_xmax, extended_ymax = moto_xmax, moto_ymax

        # Calcular personas asociadas y extender el cuadro
        for persona in personas:
            px_min, py_min, px_max, py_max = persona

            # Calcular la distancia entre el centro de la persona y el centro de la moto
            moto_center_x = (moto_xmin + moto_xmax) / 2
            moto_center_y = (moto_ymin + moto_ymax) / 2
            persona_center_x = (px_min + px_max) / 2
            persona_center_y = (py_min + py_max) / 2
            distance = ((moto_center_x - persona_center_x) ** 2 + (moto_center_y - persona_center_y) ** 2) ** 0.5
            print(distance)
            # Si la persona está dentro de la distancia máxima, asociarla a la moto
            if distance <= max_distance:
                personas_cercanas += 1

                # Expandir el cuadro extendido para incluir a la persona
                extended_xmin = min(extended_xmin, px_min)
                extended_ymin = min(extended_ymin, py_min)
                extended_xmax = max(extended_xmax, px_max)
                extended_ymax = max(extended_ymax, py_max)
            else:
                print(f"Distancia superada: {distance}")
        print(f"Cantidad de personas: {personas_cercanas}")

        # Agregar la moto, el número de personas asociadas y el cuadro extendido
        motos_con_personas.append((moto, personas_cercanas, [extended_xmin, extended_ymin, extended_xmax, extended_ymax]))

    return motos_con_personas
