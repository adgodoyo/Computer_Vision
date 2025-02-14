import cv2
import numpy as np

def main():
    # Dimensiones de la imagen
    width, height = 512, 512

    # Centro de la esfera y su radio
    cx, cy = width // 2, height // 2
    r = 100

    # Parámetros de iluminación
    Ia = 0.2   # Intensidad ambiente
    Il = 1.0   # Intensidad de la luz puntual o direccional
    ka = 0.1   # Coeficiente de reflexión ambiente
    kd = 0.7   # Coeficiente de reflexión difusa
    ks = 0.5   # Coeficiente de reflexión especular
    alpha = 32 # Exponente especular (dureza del brillo)

    # Vectores de luz y cámara (normalizados)
    # Asumimos luz y cámara en la dirección +Z
    L = np.array([1, 1, 1], dtype=np.float32)  # Dirección de la luz
    L /= np.linalg.norm(L)

    V = np.array([0, 0, 1], dtype=np.float32)  # Dirección de la cámara/observador
    V /= np.linalg.norm(V)

    # Creamos la imagen en escala de grises (float32 para calcular con decimales)
    image = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Coordenadas relativas al centro de la esfera
            dx = x - cx
            dy = y - cy
            dist2 = dx*dx + dy*dy

            # Verificamos si (x,y) está dentro de la esfera proyectada
            if dist2 <= r*r:
                # Calculamos z asumiendo una semiesfera 'mirando' hacia la cámara (z >= 0)
                z = np.sqrt(r*r - dist2)

                # Vector normal en ese punto de la esfera (N)
                N = np.array([dx, dy, z], dtype=np.float32)
                N /= np.linalg.norm(N)  # Normalizamos

                # 1) Iluminación ambiente
                I_amb = ka * Ia

                # 2) Iluminación difusa (Lambert)
                ndotl = np.dot(N, L)  # producto punto N·L
                I_dif = kd * Il * max(ndotl, 0.0)

                # 3) Iluminación especular (Phong clásico)
                R = 2.0 * ndotl * N - L  # vector reflejado
                R /= np.linalg.norm(R)  # normalizamos
                rdotv = np.dot(R, V)    # R·V
                I_spec = ks * Il * (max(rdotv, 0.0)**alpha)

                # Suma de los componentes
                I = I_amb + I_dif + I_spec

                # 4) (Opcional) Oclusión de ambiente muy simplificada
                #    Factor que atenúa un poco la luz en el centro de la esfera
                #    (Truco rápido: cuanto mayor es z, menos oclusión)
                AO_factor = 1.0 - (z / float(r)) * 0.3  
                # Aseguramos no bajar más de 70% (puedes modificarlo)
                AO_factor = max(0.7, AO_factor)
                I *= AO_factor

                # Guardamos la intensidad en la imagen
                image[y, x] = I

    # Normalizamos la imagen a [0..1] y luego a [0..255] para poder mostrarla
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # Mostramos la imagen resultante
    cv2.imshow("Shaded Sphere", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
