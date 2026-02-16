import pyaudio
import numpy as np
from constants import SAMPLE_RATE, GOOSE_FREQ_MIN, GOOSE_FREQ_MAX
import time


class GooseAudioDetector:
    def __init__(self, mic_count=4):
        self.sample_rate = SAMPLE_RATE
        self.chunk = 1024
        self.mics = mic_count
        self.audio = pyaudio.PyAudio()
        self.streams = []
        self.buffers = [[] for _ in range(mic_count)]

        # TDoA микрофоны (по кругу 1м)
        self.mic_positions = np.array(
            [[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]]  # квадратура
        )

    def start_listening(self):
        """Запуск всех микрофонов"""
        for i in range(self.mics):
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback(i),
            )
            stream.start_stream()
            self.streams.append(stream)

    def audio_callback(self, mic_id):
        def callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            # Фильтр гусиных частот 1-4кГц
            filtered = self.bandpass_filter(audio_data, GOOSE_FREQ_MIN, GOOSE_FREQ_MAX)
            self.buffers[mic_id].extend(filtered)
            if len(self.buffers[mic_id]) > self.sample_rate:
                self.buffers[mic_id] = self.buffers[mic_id][-self.sample_rate :]
            return (in_data, pyaudio.paContinue)

        return callback

    def get_direction(self):
        """TDoA → азимут гуся в градусах"""
        signals = [np.array(buf) for buf in self.buffers]

        # Корреляция между микрофонами
        tdoas = []
        for i in range(self.mics):
            for j in range(i + 1, self.mics):
                corr = np.correlate(signals[i], signals[j], mode="full")
                delay = (np.argmax(corr) - len(signals[i]) + 1) / self.sample_rate
                tdoas.append((i, j, delay))

        if not tdoas:
            return None

        # Гиперболическая локализация → азимут
        azimuth = self.tdoa_to_azimuth(tdoas)
        volume = np.max([np.max(np.abs(s)) for s in signals])

        if volume > 0.1 and azimuth is not None:  # Порог шума
            return {
                "azimuth": azimuth,
                "volume": volume,
                "confidence": min(1.0, volume),
            }
        return None

    def tdoa_to_azimuth(self, tdoas):
        """TDoA → угол в градусах (-180..180)"""
        avg_tdoa = np.mean([tdoa[2] for tdoa in tdoas])
        speed_sound = 343  # м/с
        baseline = 0.707  # расстояние между микрофонами

        angle = np.arcsin(avg_tdoa * speed_sound / baseline) * 180 / np.pi
        return angle

    def stop(self):
        for stream in self.streams:
            stream.stop_stream()
            stream.close()
        self.audio.terminate()
