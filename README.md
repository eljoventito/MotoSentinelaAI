# MotoSentinelaAI

## Descripción del Proyecto
**MotoSentinelaAI** es un sistema basado en inteligencia artificial diseñado para detectar y analizar el uso sospechoso de motocicletas. El proyecto utiliza técnicas de aprendizaje profundo, específicamente la arquitectura YOLOv8, para identificar motocicletas con dos o más pasajeros, ayudando a cumplir las regulaciones locales de seguridad vial en Lima, Perú.

El sistema permite la segmentación de imágenes, el entrenamiento de modelos personalizados y la implementación en tiempo real para generar alertas en escenarios urbanos.

---

## Características Principales
- **Segmentación de Objetos**: Precisión en la detección de motocicletas y pasajeros.
- **Generación de Dataset**: Creación de datasets personalizados a partir de videos en vivo.
- **Entrenamiento Personalizado**: Uso de YOLOv8 para entrenar modelos específicos.
- **Aplicación en Tiempo Real**: Detección desde cámaras en vivo.
- **Escalabilidad y Adaptabilidad**: Configuración modular para diversos contextos.

---

## Requisitos Previos
- **Python 3.10 o superior**
- Entorno virtual (recomendado)
- Dependencias listadas en `requirements.txt`

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/eljoventito/MotoSentinelaAI.git
cd MotoSentinelaAI
```

### 2. Crear un entorno virtual
```bash
git clone https://github.com/eljoventito/MotoSentinelaAI.git
cd MotoSentinelaAI
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales (opcional)
Si usas servicios de Google Cloud, coloca tu archivo de credenciales en la ruta:
```bash
motos_detectadas/googleapp/config_google.json
```
## Uso

### 1. Recolección de Imágenes

Ejecuta el cuaderno Jupyter para capturar imágenes desde video:
```bash
jupyter notebook P1.RecoleccionImagenes.ipynb
```
### 2. Generación del Dataset

Crea un dataset etiquetado desde tus imágenes:
```bash
jupyter notebook P2.GenerarDataset.ipynb
```
### 3. Entrenamiento del Modelo

Entrena el modelo YOLOv8 con el dataset:
```bash
jupyter notebook P3.Entrenamiento.ipynb
```

### 4. Aplicación en Tiempo Real

Implementa la detección en tiempo real desde cámaras:
```bash
jupyter notebook P4.AplicacionTiempoReal.ipynb
```


## Contribuciones

Contribuciones son bienvenidas. Si deseas aportar mejoras, abre un pull request o crea un issue. Todas las ideas son valoradas.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto

- **Autores**: Luis Villacorta, Josemanuel Cañari, Christopher Panana. 
- **Correo**: luis.villacorta.tito@gmail.com, canaripalanterossy@gmail.com, christopher271413@gmail.com
- **LinkedIn**: [Luis Villacorta](https://www.linkedin.com/in/luisvillacorta/), [Josemanuel Cañari](https://www.linkedin.com/in/josemanuel-ca%C3%B1ari-palante-015504251/), [Christopher Panana](https://www.linkedin.com/in/christopher-panana-estadistico/)
