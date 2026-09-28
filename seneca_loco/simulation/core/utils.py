import cv2
from mujoco import Renderer, mj_forward
from pathlib import Path

def image_renderer(model, data, camera=-1, titulo="MuJoCo Render"):
    """
    Renders a single frame and displays it in a window.
    Press any key to cerrar la ventana.
    """
    renderer = Renderer(model,height=720, width=1280)
    mj_forward(model, data)
    renderer.update_scene(data, camera=camera)

    # MuJoCo devuelve RGB, cv2 necesita BGR
    imagen_rgb = renderer.render()
    imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow(titulo, imagen_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    renderer.close()

def video_renderer(frames, fps=60, titulo="MuJoCo Video", output_path=None):
    """
    Muestra un video y lo guarda en un archivo si se proporciona output_path.
    
    args:
        frames: lista de imágenes RGB (numpy arrays)
        fps: frames por segundo
        titulo: nombre de la ventana
        output_path: ruta completa incluyendo el nombre del archivo (ej. .mp4)
    """
    if not frames:
        print("No frames to render.")
        return

    # --- Setup Video Writer ---
    if output_path:
        height, width, _ = frames[0].shape
        # 'mp4v' is a standard codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving video to: {output_path}")

    # --- Display and Write Loop ---
    for frame in frames:
        # MuJoCo gives RGB, OpenCV wants BGR for both display and saving
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save frame to file
        if output_path:
            out.write(bgr_frame)
        
        # Show frame on screen
        cv2.imshow(titulo, bgr_frame)
        
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    # --- Cleanup ---
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

def extract_frame(
    video_path,
    output_dir=None,
    file_name="frame.png",
    time_seconds=0,
    show=False
):
    """
    Extrae un frame de un video y lo guarda como imagen.

    Parámetros
    ----------
    video_path : str o Path
        Ruta del video.

    output_dir : str o Path o None
        Carpeta donde guardar la imagen.
        Si es None, guarda en el directorio actual.

    file_name : str
        Nombre de la imagen de salida.

    time_seconds : float
        Tiempo exacto del frame a capturar.

    show : bool
        Si True, muestra la imagen extraída.
    """

    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / file_name

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = int(fps * time_seconds)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        cap.release()
        raise ValueError("No se pudo extraer el frame")

    cv2.imwrite(str(output_path), frame)

    print(f"Imagen guardada en: {output_path}")

    if show:
        cv2.imshow("Extracted Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

    return output_path