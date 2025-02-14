import cv2
import numpy as np

def render_sphere(width, height, r, ka, kd, ks, alpha, Ia, Il, use_ao=False):
    """
    Renderiza una esfera en una imagen de tamaño (width x height) 
    aplicando iluminación ambiente, difusa y especular Phong.
    
    Parámetros:
        width, height : Dimensiones de la imagen en pixeles.
        r             : Radio de la esfera.
        ka            : Coef. reflexión ambiente [0..1].
        kd            : Coef. reflexión difusa [0..1].
        ks            : Coef. reflexión especular [0..1].
        alpha         : Exponente especular (1..100+).
        Ia            : Intensidad ambiente (0..10, aprox).
        Il            : Intensidad de la luz principal (0..10, aprox).
        use_ao        : Si True, aplica una oclusión ambiente muy sencilla.
        
    Retorna:
        image         : Imagen numpy uint8 (escala de grises).
    """
    # Convertimos r a int por si nos llega flotante
    r = int(r)
    
    # Centro de la esfera
    cx, cy = width // 2, height // 2

    # Creamos la imagen en escala de grises (float32) para los cálculos
    image = np.zeros((height, width), dtype=np.float32)

    # Vectores de luz y cámara (normalizados) 
    # Asumimos dirección +Z para ambos
    L = np.array([0, 0, 1], dtype=np.float32)
    L /= np.linalg.norm(L)

    V = np.array([0, 0, 1], dtype=np.float32)
    V /= np.linalg.norm(V)

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist2 = dx*dx + dy*dy

            # Comprobamos si (x,y) está dentro de la esfera
            if dist2 <= r*r:
                # Coordenada z asumiendo semiesfera visible
                z = np.sqrt(r*r - dist2)
                
                # Normal en ese punto de la esfera
                N = np.array([dx, dy, z], dtype=np.float32)
                N /= np.linalg.norm(N)

                # Iluminación ambiente
                I_amb = ka * Ia

                # Iluminación difusa (Lambert)
                ndotl = np.dot(N, L)
                I_dif = kd * Il * max(ndotl, 0.0)

                # Iluminación especular (Phong)
                R = 2.0 * ndotl * N - L
                R /= np.linalg.norm(R)
                rdotv = np.dot(R, V)
                I_spec = ks * Il * (max(rdotv, 0.0) ** alpha)

                # Suma de los componentes
                I = I_amb + I_dif + I_spec

                if use_ao:
                    # Oclusión ambiente muy sencilla: 
                    # factor que atenúa un poco la luz en base a z (altura)
                    AO_factor = 1.0 - (z / float(r)) * 0.3
                    # evitamos atenuar más de un 30%
                    AO_factor = max(0.7, AO_factor)
                    I *= AO_factor

                image[y, x] = I

    # Normalizar a [0..255] para visualización
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def main():
    # Parámetros iniciales y rangos
    width, height = 512, 512

    # Rango y valores iniciales
    r_init     = 100  # radio inicial
    ka_init    = 10   # -> 0.10
    kd_init    = 70   # -> 0.70
    ks_init    = 50   # -> 0.50
    alpha_init = 32
    Ia_init    = 20   # -> 2.0
    Il_init    = 100  # -> 1.0
    ao_init    = 0    # 0 = off, 1 = on

    cv2.namedWindow("Shaded Sphere", cv2.WINDOW_AUTOSIZE)

    # Trackbars:
    # 1. Radius (de 10 a 250)
    cv2.createTrackbar("Radius", "Shaded Sphere", r_init, 250, lambda x: None)
    # 2. ka (0..100) => real = x/100
    cv2.createTrackbar("ka", "Shaded Sphere", ka_init, 100, lambda x: None)
    # 3. kd (0..100) => real = x/100
    cv2.createTrackbar("kd", "Shaded Sphere", kd_init, 100, lambda x: None)
    # 4. ks (0..100) => real = x/100
    cv2.createTrackbar("ks", "Shaded Sphere", ks_init, 100, lambda x: None)
    # 5. alpha (1..100)
    cv2.createTrackbar("alpha", "Shaded Sphere", alpha_init, 100, lambda x: None)
    # 6. Ia (0..100) => real = x/10
    cv2.createTrackbar("Ia", "Shaded Sphere", Ia_init, 100, lambda x: None)
    # 7. Il (0..300) => real = x/100
    cv2.createTrackbar("Il", "Shaded Sphere", Il_init, 300, lambda x: None)
    # 8. Ambient Occlusion On/Off
    cv2.createTrackbar("AO", "Shaded Sphere", ao_init, 1, lambda x: None)

    while True:
        # Leemos valores de trackbars
        r_t     = cv2.getTrackbarPos("Radius", "Shaded Sphere")
        ka_t    = cv2.getTrackbarPos("ka", "Shaded Sphere") / 100.0
        kd_t    = cv2.getTrackbarPos("kd", "Shaded Sphere") / 100.0
        ks_t    = cv2.getTrackbarPos("ks", "Shaded Sphere") / 100.0
        alpha_t = max(cv2.getTrackbarPos("alpha", "Shaded Sphere"), 1)
        Ia_t    = cv2.getTrackbarPos("Ia", "Shaded Sphere") / 10.0
        Il_t    = cv2.getTrackbarPos("Il", "Shaded Sphere") / 100.0
        ao_t    = cv2.getTrackbarPos("AO", "Shaded Sphere")

        # Renderizamos con los valores actuales
        image = render_sphere(
            width, height,
            r_t,
            ka_t, kd_t, ks_t,
            alpha_t,
            Ia_t, Il_t,
            use_ao=bool(ao_t)
        )

        # Mostramos la imagen
        cv2.imshow("Shaded Sphere", image)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC para salir
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()