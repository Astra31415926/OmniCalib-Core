import numpy as np


class OmniCalibEnvironment:
    """Модуль симуляции 'кривой среды' (OmniCalib-Environment).

    Отвечает за генерацию идеальных спектров, их искажение сенсором
    и внесение паразитных помех (Crosstalk).
    """

    def __init__(self):
        # 1. Сетка данных: от 380 до 780 нм с шагом 10 нм (41 элемент)
        self.wavelengths = np.arange(380, 781, 10)
        self.num_points = len(self.wavelengths)

        # Инициализация заглушек для калибровочных данных
        self.sensor_qe = None  # Спектральная чувствительность сенсора (RGB)
        self.violet_limiter_ref = None  # Эталонная кривая фиолетового лимитера

    def load_sensor_profile(self, r_curve, g_curve, b_curve):
        """Загрузка реальных спектральных кривых сенсора (например, Sony IMX).

        Каждая кривая должна быть вектором из 41 элемента.
        """
        # Формируем матрицу чувствительности сенсора (3 строки x 41 колонка)
        self.sensor_qe = np.vstack([r_curve, g_curve, b_curve])

    def load_violet_limiter(self, reference_spectrum):
        """Загрузка эталонного спектра отражения фиолетового лимитера."""
        self.violet_limiter_ref = np.array(reference_spectrum)

    def apply_distortions(self, ideal_spectrum, matrix_a, vector_b):
        """Искусственное загрязнение и сдвиг спектра.

        АРХИТЕКТУРНАЯ ЗАГЛУШКА: Сейчас реализована линейная модель (Y = A*X +
        B).
        В будущем здесь будет нелинейная модель Кубелки-Мунка без изменения
        интерфейса функции.
        """
        # Простая линейная трансформация спектра (срез, пожелтение и т.д.)
        distorted_spectrum = np.dot(matrix_a, ideal_spectrum) + vector_b

        # Ограничиваем значения физическими рамками [0.0, 1.0]
        return np.clip(distorted_spectrum, 0.0, 1.0)

    def capture_by_sensor(self, spectrum):
        """Симуляция съемки спектра камерой.

        Перемножает спектр излучения на квантовую эффективность сенсора.
        На выходе получаем 'грязный' RGB-сигнал с перекрестными помехами
        (Crosstalk).
        """
        if self.sensor_qe is None:
            raise ValueError(
                "Профиль сенсора не загружен. Сначала вызовите load_sensor_profile."
            )

        # Интегрирование (скалярное произведение) спектра с кривыми RGB
        rgb_signal = np.dot(self.sensor_qe, spectrum)
        return rgb_signal


# =====================================================================
# ДЕМОНСТРАЦИЯ РАБОТЫ СКРИПТА (ВЕРИФИКАЦИЯ ЛОГИКИ)
# =====================================================================
if __name__ == "__main__":
    print("--- Инициализация ядра OmniCalib-Environment ---")
    sim = OmniCalibEnvironment()

    # 1. Генерируем синтетический эталонный спектр (например, плоский белый = 1.0)
    ideal_test_spectrum = np.ones(sim.num_points)

    # 2. Создаем фейковые (для теста) кривые сенсора Sony (широкие, перекрывающиеся гауссианы)
    # В реальном коде сюда будут загружаться данные из баз вроде Imageval
    mock_r = np.exp(-((sim.wavelengths - 600) ** 2) / (2 * 40**2))
    mock_g = np.exp(-((sim.wavelengths - 530) ** 2) / (2 * 40**2))
    mock_b = np.exp(-((sim.wavelengths - 450) ** 2) / (2 * 40**2))
    sim.load_sensor_profile(mock_r, mock_g, mock_b)

    # 3. Задаем матрицу искажения среды (Матрица А) и аддитивный шум (Веткор B)
    # Например, среда сильно 'желтит' (гасит синюю компоненту на краях спектра)
    matrix_a = np.eye(sim.num_points)
    matrix_a[0:10, 0:10] *= 0.5  # Глушим фиолетово-синюю зону (380-480 нм)
    vector_b = (
        np.zeros(sim.num_points) + 0.02
    )  # Небольшая паразитная засветка (дымка)

    # 4. Запуск симуляции
    dirty_spectrum = sim.apply_distortions(
        ideal_test_spectrum, matrix_a, vector_b
    )
    rgb_output = sim.capture_by_sensor(dirty_spectrum)

    print(f"Размерность сетки: {sim.num_points} точек.")
    print(f"Искаженный спектр (первые 5 точек): {dirty_spectrum[:5]}")
    print(f"Выходной сигнал с камеры (RGB): {rgb_output}")
    print("--- Симуляция успешно завершена ---")
