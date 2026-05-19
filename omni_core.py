import numpy as np


class OmniCalibCore:
    """Математическое ядро очистки и восстановления спектра (OmniCalib-Core)."""

    def __init__(self):
        self.wavelengths = np.arange(380, 781, 10)
        self.num_points = len(self.wavelengths)

    def reconstruct_spectrum(
        self, captured_rgb, sensor_qe, violet_limiter_ref, measured_violet_rgb
    ) -> np.ndarray:
        """Алгоритм деконволюции (очистки) спектра.

        Использует фиолетовый лимитер как маркер границы для нормирования
        передаточной функции среды и разделения каналов (Crosstalk).
        """
        # 1. Математический якорь: вычисляем коэффициент искажения на границе (380-400 нм)
        # Сравниваем эталонный фиолетовый RGB с тем, что реально замерил сенсор
        expected_violet_rgb = np.dot(sensor_qe, violet_limiter_ref)

        # Вычисляем вектор ошибки среды для граничной зоны (простая матрица коррекции)
        # Избегаем деления на ноль с помощью клиппинга
        env_correction_factor = expected_violet_rgb / np.clip(
            measured_violet_rgb, 1e-5, None
        )

        # 2. Коррекция входного RGB-сигнала
        corrected_rgb = captured_rgb * env_correction_factor

        # 3. Обращение матрицы сенсора (Деконволюция)
        # Так как матрица сенсора (3 x 41) не квадратная, применяем псевдообратную матрицу Мура-Пенроуза
        pinv_sensor = np.linalg.pinv(sensor_qe)

        # Восстанавливаем спектральный вектор (41 точка) из 3-х каналов RGB
        reconstructed = np.dot(pinv_sensor, corrected_rgb)

        # Ограничиваем физический диапазон спектра [0.0, 1.0]
        return np.clip(reconstructed, 0.0, 1.0)


# =====================================================================
# ТЕСТ СВЯЗКИ: СИМУЛЯТОР + ЯДРО ВЫЧИСЛЕНИЯ
# =====================================================================
if __name__ == "__main__":
    from omni_sim import OmniCalibEnvironment

    print("--- Тестирование сквозной калибровки OmniCalib ---")

    # Инициализируем среду и ядро
    sim = OmniCalibEnvironment()
    core = OmniCalibCore()

    # Создаем тестовые кривые сенсора
    mock_r = np.exp(-((sim.wavelengths - 600) ** 2) / (2 * 40**2))
    mock_g = np.exp(-((sim.wavelengths - 530) ** 2) / (2 * 40**2))
    mock_b = np.exp(-((sim.wavelengths - 450) ** 2) / (2 * 40**2))
    sim.load_sensor_profile(mock_r, mock_g, mock_b)

    # Создаем эталон фиолетового лимитера (пик в зоне 380-410 нм)
    violet_ref = np.exp(-((sim.wavelengths - 390) ** 2) / (2 * 10**2))

    # Среда дает искажение (желтизна: гасит синий край спектра на 50%)
    matrix_a = np.eye(sim.num_points)
    matrix_a[0:5, 0:5] *= 0.5
    vector_b = np.zeros(sim.num_points)

    # Симулируем замер фиолетового лимитера в «кривой среде»
    dirty_violet_spectrum = sim.apply_distortions(violet_ref, matrix_a, vector_b)
    measured_violet_rgb = sim.capture_by_sensor(dirty_violet_spectrum)

    # Симулируем съемку объекта (например, идеального белого листа)
    ideal_object = np.ones(sim.num_points)
    dirty_object_spectrum = sim.apply_distortions(
        ideal_object, matrix_a, vector_b
    )
    captured_rgb = sim.capture_by_sensor(dirty_object_spectrum)

    # ЯДРО ВОССТАНАВЛИВАЕТ ИСХОДНЫЙ СПЕКТР ОБЪЕКТА
    restored_spectrum = core.reconstruct_spectrum(
        captured_rgb=captured_rgb,
        sensor_qe=sim.sensor_qe,
        violet_limiter_ref=violet_ref,
        measured_violet_rgb=measured_violet_rgb,
    )

    print("Восстановление завершено.")
    print(f"Исходный спектр (первые 3 точки): {ideal_object[:3]}")
    print(f"Восстановленный спектр (первые 3 точки): {restored_spectrum[:3]}")
